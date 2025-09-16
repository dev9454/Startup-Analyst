from __future__ import annotations
import json, json5, logging
from typing import Optional, Dict, Any
import boto3
from botocore.exceptions import ClientError
from tenacity import retry, stop_after_attempt, wait_exponential
from config import BEDROCK_REGION, BEDROCK_MODEL_ID

logger = logging.getLogger(__name__)
client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

SYSTEM_GUARD = ("You are an expert investment analyst AI. "
                "Always respond ONLY with valid JSON matching the requested schema. "
                "No prose, no markdown. Use nulls if unknown.")

def _wrap_prompt(user_prompt: str, context: str = "", system: str = SYSTEM_GUARD) -> str:
    merged = f"{system}\n\nContext:\n{context}\n\nTask:\n{user_prompt}"
    return ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{merged}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def call_bedrock_llm(user_prompt: str, context: str = "", temperature: float = 0.0, max_gen_len: int = 2048) -> str:
    formatted_prompt = _wrap_prompt(user_prompt, context=context)
    try:
        resp = client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({"prompt": formatted_prompt, "max_gen_len": max_gen_len, "temperature": temperature})
        )
        return json.loads(resp["body"].read())["generation"]
    except (ClientError, Exception) as e:
        logger.error(f"Bedrock LLM invoke failed: {e}")
        raise

def call_bedrock_json(user_prompt: str, context: str = "", temperature: float = 0.0, schema_hint: Optional[str] = None) -> Dict[str, Any]:
    hint = f"\n\nJSON Schema (hint): {schema_hint}" if schema_hint else ""
    instr = "Return ONLY JSON. No commentary. Ensure keys exist; null if unknown."
    raw = call_bedrock_llm(user_prompt + "\n\n" + instr + hint, context=context, temperature=temperature)
    try:
        return json.loads(raw)
    except Exception:
        try:
            return json5.loads(raw)
        except Exception:
            s, e = raw.find("{"), raw.rfind("}")
            if s >= 0 and e > s:
                return json.loads(raw[s:e+1])
            raise ValueError(f"Invalid JSON from LLM: {raw[:400]}")
