"""
Approach 1 – Direct LLM Extraction
====================================
Reads job descriptions from the Excel file, sends each one to Claude via
Amazon Bedrock, and asks it to extract skills with proficiency level and
importance directly from the text. No external catalogue required.

Output: approach1_results.json  (and a summary CSV)

Run:
    python approach1_direct_extraction.py
"""

import json
import re
import pandas as pd
from bedrock_client import invoke_claude

# ── Configuration ──────────────────────────────────────────────────────────────
TEST_MODE    = True              # ← True = process 10 records only; False = process all

EXCEL_FILE   = "050326 SR Job Description Details LE.xlsx"
OUTPUT_CSV   = "approach1_results_test.csv" if TEST_MODE else "approach1_results_full.csv"
OUTPUT_JSON  = "approach1_results_test.json" if TEST_MODE else "approach1_results_full.json"
TEST_LIMIT   = 3
# ───────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert HR analyst specialising in skills-based job mapping.
Your task is to extract structured skill data from job descriptions.
Always respond with valid JSON only — no prose, no markdown fences.
""".strip()

EXTRACTION_PROMPT = """
Analyse the following job description and extract all skills mentioned or implied.

For each skill return:
- "skill_name": short canonical name (e.g. "Stakeholder Engagement")
- "category": one of ["Technical", "Soft", "Domain", "Tool/System", "Compliance/Regulatory"]
- "proficiency_level": one of ["Awareness", "Working", "Practitioner", "Expert"]
- "importance": one of ["Core", "Important", "Nice to Have"]
- "evidence": a short quote or phrase from the JD that supports this skill

Return a JSON object with a single key "skills" containing an array of skill objects.

Job Title: {job_title}
Job Description:
{job_description}
"""


def extract_json(text: str) -> dict:
    """Strip any markdown fences and parse JSON from model response."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(text)


def process_row(row: pd.Series) -> dict:
    job_ref    = row["Job Ref ID"]
    job_title  = row["Job Title"]
    job_status = row["Job Status"]
    jd_text    = str(row["Default Job Ad Job Description"] or "").strip()

    print(f"  Processing [{job_ref}] {job_title} ({job_status}) ...", end=" ", flush=True)

    if not jd_text or jd_text.lower() == "nan":
        print("SKIPPED (no JD text)")
        return {
            "job_ref_id": job_ref,
            "job_title":  job_title,
            "job_status": job_status,
            "skills":     [],
            "error":      "No job description text available",
        }

    prompt = EXTRACTION_PROMPT.format(
        job_title=job_title,
        job_description=jd_text[:6000],  # Truncate very long JDs to stay within token limits
    )

    try:
        raw_response = invoke_claude(prompt, system=SYSTEM_PROMPT)
        parsed       = extract_json(raw_response)
        skills       = parsed.get("skills", [])
        print(f"OK ({len(skills)} skills)")
        return {
            "job_ref_id": job_ref,
            "job_title":  job_title,
            "job_status": job_status,
            "skills":     skills,
        }
    except json.JSONDecodeError as e:
        print(f"JSON PARSE ERROR: {e}")
        return {
            "job_ref_id": job_ref,
            "job_title":  job_title,
            "job_status": job_status,
            "skills":     [],
            "error":      f"JSON parse error: {e}",
            "raw_response": raw_response if "raw_response" in dir() else "",
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "job_ref_id": job_ref,
            "job_title":  job_title,
            "job_status": job_status,
            "skills":     [],
            "error":      str(e),
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
                    "job_ref_id":       role["job_ref_id"],
                    "job_title":        role["job_title"],
                    "job_status":       role["job_status"],
                    "skill_name":       skill.get("skill_name", ""),
                    "category":         skill.get("category", ""),
                    "proficiency_level":skill.get("proficiency_level", ""),
                    "importance":       skill.get("importance", ""),
                    "evidence":         skill.get("evidence", ""),
                })
        else:
            rows_flat.append({
                "job_ref_id": role["job_ref_id"],
                "job_title":  role["job_title"],
                "job_status": role["job_status"],
                "skill_name": "",
                "error":      role.get("error", ""),
            })

    df_out = pd.DataFrame(rows_flat)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"✅ CSV  saved to: {OUTPUT_CSV}")

    # Summary
    total_skills = sum(len(r["skills"]) for r in results)
    print(f"\n📊 Summary: {len(results)} roles processed, {total_skills} skills extracted")


if __name__ == "__main__":
    main()
