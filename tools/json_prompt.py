# tools/json_prompt.py
def build_json_prompt(schema: str, task_hint: str = "") -> str:
    """
    Returns instructions that coerce a JSON-only response.
    schema: a compact JSON literal showing required keys.
    task_hint: short plain text reminding what to output.
    """
    return (
        (task_hint or "Complete the task") + "\n\n"
        "Rules:\n"
        "1) Return EXACTLY ONE JSON object (no markdown, no prose).\n"
        "2) Use null if a value is unknown.\n"
        "3) Keep strings short; avoid newlines in values.\n"
        "4) DO NOT include any text before or after the JSON object.\n\n"
        f"JSON schema (informal): {schema}\n"
        "Respond now with ONLY the JSON object."
    )
