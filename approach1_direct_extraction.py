"""
Approach 1 – Direct LLM Extraction
====================================
Reads job descriptions from the Excel file, sends each one to Claude via
Amazon Bedrock, and asks it to extract skills with proficiency level,
importance, and confidence score directly from the text.
No external catalogue required.

Improvements over v1:
  • Anchored canonical skill names against a reference list to reduce naming drift
  • Requests a confidence score (0.0–1.0) per skill
  • Asks Claude to deduplicate overlapping skills before returning
  • Captures function / sub_function in output for downstream grouping
  • Retries on transient AWS / JSON errors (up to 2 retries)
  • JD truncation raised to 8 000 chars

Output: approach1_results_test.csv / approach1_results_full.csv (and .json)

Run:
    python approach1_direct_extraction.py
"""

import json
import re
import time
import pandas as pd
from bedrock_client import invoke_claude

# ── Configuration ──────────────────────────────────────────────────────────────
TEST_MODE    = True              # ← True = process TEST_LIMIT records only; False = all

EXCEL_FILE   = "050326 SR Job Description Details LE.xlsx"
OUTPUT_CSV   = "approach1_results_test.csv" if TEST_MODE else "approach1_results_full.csv"
OUTPUT_JSON  = "approach1_results_test.json" if TEST_MODE else "approach1_results_full.json"
TEST_LIMIT   = 3
MAX_RETRIES  = 2                 # retry on transient errors
JD_CHAR_LIMIT = 8000             # characters to send to Claude per JD
# ───────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert HR analyst specialising in skills-based job mapping for an energy company.
Your task is to extract structured, deduplicated skill data from job descriptions.
Always respond with valid JSON only — no prose, no markdown fences.
""".strip()

# A canonical reference list is embedded in the prompt so Claude uses consistent
# naming. It may also identify skills not on this list — that is expected.
EXTRACTION_PROMPT = """
Analyse the job description below and extract every skill that is clearly
mentioned or strongly implied. Follow these rules strictly:

RULES
1. Use canonical skill names. Where a skill matches one in the REFERENCE LIST
   below, use exactly that name. For skills not on the list, invent a short,
   consistent name (Title Case, 3–5 words max).
2. Do NOT include vague filler phrases such as "Willingness to Learn",
   "Enthusiasm", "Positive Attitude", or any skill that every employee at
   every company is expected to have by default.
3. Deduplicate: if two skills are near-identical (e.g. "Written Communication"
   and "Written Communication Skills"), merge them into one entry.
4. For each skill return EXACTLY these fields:
   - "skill_name":        canonical short name
   - "category":          one of ["Technical", "Soft", "Domain", "Tool/System", "Compliance/Regulatory"]
   - "proficiency_level": one of ["Awareness", "Working", "Practitioner", "Expert"]
   - "importance":        one of ["Core", "Important", "Nice to Have"]
   - "confidence":        float 0.0–1.0 — how confident you are this skill is genuinely required
   - "evidence":          a short verbatim quote (≤ 20 words) from the JD that supports this skill

PROFICIENCY GUIDANCE
  Awareness    – candidate has heard of / understands the concept
  Working      – applies the skill with some supervision
  Practitioner – applies independently and can guide others
  Expert       – recognised authority, can set direction

IMPORTANCE GUIDANCE
  Core         – role cannot be performed without this skill
  Important    – significantly improves performance
  Nice to Have – helpful but not essential

REFERENCE LIST (use these exact names where applicable)
Stakeholder Engagement, Stakeholder Management, Financial Analysis, Financial Reporting,
Budget Management, Forecasting, Risk Management, Compliance Management, Quality Assurance,
Project Management, Programme Management, Change Management, Data Analysis, Report Writing,
Presentation Skills, Communication Skills, Verbal Communication, Written Communication,
Problem Solving, Critical Thinking, Decision Making, Leadership, Team Management,
People Development, Coaching and Mentoring, Negotiation, Influencing, Conflict Resolution,
Customer Service, Customer Relationship Management, Safety Management, Health and Safety,
CDM Regulations, Nuclear Safety, Radiation Protection, Security Clearance Awareness,
Mechanical Engineering, Electrical Engineering, Civil Engineering, Welding,
Pipework Installation, Quality Control, Non-Conformance Reporting, Inspection and Testing,
IT Service Delivery, Microsoft Office, SAP, SharePoint, Power BI, Excel Advanced,
Procurement, Supply Chain Management, Contract Management, Vendor Management,
Training Delivery, Instructional Design, Systematic Approach to Training,
Emergency Planning, Environmental Management, Sustainability, Net Zero Knowledge,
Analytical Skills, Attention to Detail, Time Management, Planning and Organising,
Commercial Acumen, Business Partnering, Audit and Assurance, Regulatory Affairs,
Active Directory, Network Administration, LAN/WAN Networking, Cyber Security

Return a JSON object with a single key "skills" containing the array.

Job Title: {job_title}
Business Function: {function}
Job Description:
{job_description}
"""


def extract_json(text: str) -> dict:
    """Strip any markdown fences and parse JSON from model response."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(text)


def process_row(row: pd.Series) -> dict:
    job_ref      = row["Job Ref ID"]
    job_title    = row["Job Title"]
    job_status   = row["Job Status"]
    function_    = str(row.get("Function", "") or "")
    sub_function = str(row.get("Sub-function", "") or "")
    business_unit= str(row.get("Business unit", "") or "")
    jd_text      = str(row["Default Job Ad Job Description"] or "").strip()

    print(f"  Processing [{job_ref}] {job_title} ({job_status}) ...", end=" ", flush=True)

    if not jd_text or jd_text.lower() == "nan":
        print("SKIPPED (no JD text)")
        return {
            "job_ref_id":   job_ref,
            "job_title":    job_title,
            "job_status":   job_status,
            "function":     function_,
            "sub_function": sub_function,
            "business_unit":business_unit,
            "skills":       [],
            "error":        "No job description text available",
        }

    prompt = EXTRACTION_PROMPT.format(
        job_title=job_title,
        function=f"{function_} / {sub_function}".strip(" /"),
        job_description=jd_text[:JD_CHAR_LIMIT],
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            raw_response = invoke_claude(prompt, system=SYSTEM_PROMPT)
            parsed       = extract_json(raw_response)
            skills       = parsed.get("skills", [])
            print(f"OK ({len(skills)} skills)")
            return {
                "job_ref_id":   job_ref,
                "job_title":    job_title,
                "job_status":   job_status,
                "function":     function_,
                "sub_function": sub_function,
                "business_unit":business_unit,
                "skills":       skills,
            }
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            print(f"JSON PARSE ERROR (attempt {attempt}): {e}", end=" ")
            if attempt <= MAX_RETRIES:
                time.sleep(2 ** attempt)
        except Exception as e:
            last_error = str(e)
            print(f"ERROR (attempt {attempt}): {e}", end=" ")
            if attempt <= MAX_RETRIES:
                time.sleep(2 ** attempt)

    print(f"FAILED after {MAX_RETRIES + 1} attempts")
    return {
        "job_ref_id":   job_ref,
        "job_title":    job_title,
        "job_status":   job_status,
        "function":     function_,
        "sub_function": sub_function,
        "business_unit":business_unit,
        "skills":       [],
        "error":        last_error,
    }


def main():
    print(f"\n{'='*60}")
    print("  Approach 1 – Direct LLM Skill Extraction")
    print(f"{'='*60}\n")

    # Load Excel
    print(f"Loading: {EXCEL_FILE}")
    df = pd.read_excel(EXCEL_FILE)
    print(f"Total rows: {len(df)}")

    if TEST_MODE:
        df = df.head(TEST_LIMIT)
        print(f"⚠️  TEST MODE — processing first {TEST_LIMIT} records only\n")
    else:
        print(f"🚀 FULL MODE — processing all {len(df)} records\n")

    # Process each role
    results = []
    for _, row in df.iterrows():
        result = process_row(row)
        results.append(result)

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ JSON saved to: {OUTPUT_JSON}")

    # Flatten to CSV for easy viewing
    rows_flat = []
    for role in results:
        if role["skills"]:
            for skill in role["skills"]:
                rows_flat.append({
                    "job_ref_id":        role["job_ref_id"],
                    "job_title":         role["job_title"],
                    "job_status":        role["job_status"],
                    "function":          role.get("function", ""),
                    "sub_function":      role.get("sub_function", ""),
                    "business_unit":     role.get("business_unit", ""),
                    "skill_name":        skill.get("skill_name", ""),
                    "category":          skill.get("category", ""),
                    "proficiency_level": skill.get("proficiency_level", ""),
                    "importance":        skill.get("importance", ""),
                    "confidence":        skill.get("confidence", ""),
                    "evidence":          skill.get("evidence", ""),
                })
        else:
            rows_flat.append({
                "job_ref_id":    role["job_ref_id"],
                "job_title":     role["job_title"],
                "job_status":    role["job_status"],
                "function":      role.get("function", ""),
                "sub_function":  role.get("sub_function", ""),
                "business_unit": role.get("business_unit", ""),
                "skill_name":    "",
                "error":         role.get("error", ""),
            })

    df_out = pd.DataFrame(rows_flat)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"✅ CSV  saved to: {OUTPUT_CSV}")

    # Summary
    total_skills = sum(len(r["skills"]) for r in results)
    print(f"\n📊 Summary: {len(results)} roles processed, {total_skills} skills extracted")


if __name__ == "__main__":
    main()
