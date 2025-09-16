# tools/llm.py
import json
import logging
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---- REQUIRED: your exact Bedrock setup ----
BEDROCK_REGION = "us-east-1"
client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

# ---- Minimal stubs so your function has these names ----
class _DummyRegistry:
    def render_for_prompt(self) -> str:
        # return a short capability list; swap with your real tool catalog later
        return "No tools available; analysis only."

REGISTRY = _DummyRegistry()

# NO SPACES in variable names — this must be a valid identifier
SUPERVISOR_PROMPT_TEMPLATE = """You are an expert VC analyst.
Use the following action catalog (if any) and the provided context to complete the task.

[ACTION_CATALOG]
{action_catalog}

[CONTEXT]
{context}

[TASK]
{user_prompt}
"""

# ---- Your function, unchanged in behavior, just fixed identifiers/imports ----
def call_bedrock_llm(user_prompt: str, context: str = "") -> str:
    action_catalog = REGISTRY.render_for_prompt()
    prompt = SUPERVISOR_PROMPT_TEMPLATE.format(
        action_catalog=action_catalog,
        user_prompt=user_prompt,
        context=context,
    )

    formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
    native_request = {
        "prompt": formatted_prompt,
        "max_gen_len": 2048,
        "temperature": 0.0,
    }
    request_body = json.dumps(native_request)
    try:
        response = client.invoke_model(
            modelId="meta.llama3-70b-instruct-v1:0",
            body=request_body
        )
        model_response = json.loads(response["body"].read())
        return model_response["generation"]
    except (ClientError, Exception) as e:
        logger.error(f"ERROR: Could not invoke LLM. Reason: {e}")
        raise
