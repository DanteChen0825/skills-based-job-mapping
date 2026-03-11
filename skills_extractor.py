"""
Skills-based job mapping — extraction pipeline.

Main entry point:
    extract_skills(business_unit, function, sub_function, job_description)
        -> dict  (parsed JSON result)

Batch entry point:
    extract_skills_batch(records: list[dict])
        -> list[dict]
"""

import json
from bedrock_client import invoke_claude

# ── System prompt (role + rules) ──────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a workforce skills analyst. Your job is to analyse EDF job descriptions
and extract a structured set of skills with required proficiency levels.

Rules:
- Return ONLY valid JSON — no preamble, no commentary, no markdown fences.
- Each skill must appear only once (deduplicate).
- Minimum 8 skills per record; capture all meaningful skills present.
- evidence must be a direct quote or close paraphrase from the JD, under 20 words.
- If the job description is empty or too short, return the error JSON shown below.

Proficiency scale:
  1 – Awareness    : "desirable", "exposure to", "familiar with"
  2 – Foundation   : "basic", "some experience", "working towards"
  3 – Practitioner : "experience in", "competent", "able to", "knowledge of"
  4 – Advanced     : "significant experience", "proven track record", "expert knowledge"
  5 – Expert/Lead  : "lead", "set standards", "head of", "strategic", "accountable for"

Skill categories (use exactly these names):
  - Technical / Domain Skills
  - Digital & IT Skills
  - Analytical & Problem-Solving Skills
  - Communication & Stakeholder Skills
  - Leadership & Management Skills
  - Compliance & Regulatory Skills
  - Project & Programme Management Skills
  - Interpersonal & Behavioural Skills

Output schema for a successful record:
{
  "business_unit": "<Business Unit>",
  "function": "<Function>",
  "sub_function": "<Sub-function>",
  "job_title": "<Inferred job title>",
  "seniority_level": "<Entry | Junior | Mid | Senior | Lead | Head>",
  "skills": [
    {
      "skill_name": "<skill name>",
      "category": "<one of the 8 categories>",
      "proficiency_required": <1–5>,
      "evidence": "<short quote from JD>"
    }
  ]
}

Output schema for an empty/insufficient JD:
{
  "error": "Insufficient job description content",
  "business_unit": "...",
  "function": "...",
  "sub_function": "..."
}
"""


def build_prompt(
    business_unit: str,
    function: str,
    sub_function: str,
    job_description: str,
) -> str:
    """
    Build the user-turn prompt for a single job record.
    """
    return f"""\
Extract skills from the following EDF job record and return the result as JSON.

Business Unit: {business_unit}
Function: {function}
Sub-function: {sub_function}
Job Description:
{job_description}
"""


def build_batch_prompt(records: list[dict]) -> str:
    """
    Build the user-turn prompt for multiple job records.
    Each dict must have keys: business_unit, function, sub_function, job_description.
    Returns a JSON array — one result object per input row, in order.
    """
    blocks = []
    for i, r in enumerate(records, 1):
        blocks.append(
            f"--- Record {i} ---\n"
            f"Business Unit: {r.get('business_unit', '')}\n"
            f"Function: {r.get('function', '')}\n"
            f"Sub-function: {r.get('sub_function', '')}\n"
            f"Job Description:\n{r.get('job_description', '')}\n"
        )

    return (
        "Extract skills from each of the following EDF job records. "
        "Return a JSON array with one result object per record, in order.\n\n"
        + "\n".join(blocks)
    )


def _parse_json_response(raw: str) -> dict | list:
    """Strip any accidental markdown fences and parse JSON."""
    text = raw.strip()
    # Remove ```json ... ``` or ``` ... ``` wrappers if model adds them anyway
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())


def extract_skills(
    business_unit: str,
    function: str,
    sub_function: str,
    job_description: str,
) -> dict:
    """
    Extract skills from a single job record.

    Args:
        business_unit:    Organisational unit (e.g. "Hinkley Point C")
        function:         Department / function name
        sub_function:     Team / sub-department name
        job_description:  Full text of the job advertisement

    Returns:
        Parsed dict matching the output schema defined in the system prompt.
    """
    prompt = build_prompt(business_unit, function, sub_function, job_description)
    raw = invoke_claude(prompt, system=_SYSTEM_PROMPT)
    return _parse_json_response(raw)


def extract_skills_batch(records: list[dict]) -> list[dict]:
    """
    Extract skills from multiple job records in a single API call.

    Args:
        records: List of dicts, each with keys:
                   business_unit, function, sub_function, job_description

    Returns:
        List of parsed result dicts, one per input record, in order.
    """
    if not records:
        return []
    prompt = build_batch_prompt(records)
    raw = invoke_claude(prompt, system=_SYSTEM_PROMPT)
    result = _parse_json_response(raw)
    if isinstance(result, dict):
        # Model returned a single object instead of array (e.g. 1-record batch)
        return [result]
    return result


# ── Quick CLI test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_result = extract_skills(
        business_unit="Customers (64013735)",
        function="EDF Business & Wholesale Services",
        sub_function="Mid Market (65010009)",
        job_description=(
            "Customer Service Advisor. A passion for delivering great customer "
            "service – no experience needed, just a willingness to learn! "
            "Strong communication skills – speaking, listening, and writing with "
            "confidence. A problem-solver who thrives in a fast-moving environment. "
            "Comfortable using IT systems and navigating multiple platforms. "
            "Take ownership of customer queries, resolving issues efficiently."
        ),
    )
    print(json.dumps(sample_result, indent=2))
