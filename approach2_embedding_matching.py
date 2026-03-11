"""
Approach 2 – Embedding-Based Matching with ESCO Catalogue  (Statistical Edition)
==================================================================================
Pipeline:
  1. Fetch ESCO skill catalogue from EU API (or built-in seed if offline)
  2. Build TF-IDF index over skill titles + descriptions
  3. For each JD: retrieve top-K candidates by cosine similarity
  4. Claude validates candidates → confirms skill, assigns proficiency + importance
  5. Optionally enrich confirmed skills with full ESCO detail (skill_type, reuse_level)

Statistical outputs added:
  - Per-role: candidates_retrieved, confirmed, acceptance_rate, cosine score stats
  - Cross-run: printed summary table — min/mean/max cosine, acceptance rate per role
  - approach2_stats.csv — machine-readable statistics file

Output:
  approach2_results_test.json / approach2_results_test.csv
  approach2_stats_test.csv

Run:
    python approach2_embedding_matching.py
"""

import json
import re
import time
import statistics
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from bedrock_client import invoke_claude

# ── Configuration ──────────────────────────────────────────────────────────────
TEST_MODE         = True              # ← True = process TEST_LIMIT records; False = all

EXCEL_FILE        = "050326 SR Job Description Details LE.xlsx"
OUTPUT_CSV        = "approach2_results_test.csv"  if TEST_MODE else "approach2_results_full.csv"
OUTPUT_JSON       = "approach2_results_test.json" if TEST_MODE else "approach2_results_full.json"
STATS_CSV         = "approach2_stats_test.csv"    if TEST_MODE else "approach2_stats_full.csv"
TEST_LIMIT        = 5           # ← number of records in TEST_MODE
TOP_K_SKILLS      = 15          # candidates retrieved per JD from TF-IDF
ESCO_SKILLS_LIMIT = 500         # how many ESCO skills to pull from the API
ESCO_API_BASE     = "https://esco.ec.europa.eu/api"
ESCO_LANGUAGE     = "en"
# ───────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert HR analyst specialising in skills-based job mapping.
Your task is to validate and enrich a shortlist of candidate skills against a job description.
Always respond with valid JSON only — no prose, no markdown fences.
""".strip()

ENRICHMENT_PROMPT = """
You are given a job description and a shortlist of candidate skills retrieved from the ESCO taxonomy.
For each skill, decide:
1. Is it genuinely relevant to this role? (true/false)
2. If relevant, what proficiency level is required? ["Awareness", "Working", "Practitioner", "Expert"]
3. If relevant, what is its importance? ["Core", "Important", "Nice to Have"]
4. Provide a short evidence quote (≤20 words) from the JD.

Return a JSON object with key "skills" — an array. Only include skills where "relevant" is true.

Each item:
{{
  "skill_name": "...",
  "esco_uri": "...",
  "proficiency_level": "...",
  "importance": "...",
  "evidence": "..."
}}

Job Title: {job_title}
Business Function: {function}

Job Description (excerpt):
{job_description}

Candidate skills from ESCO (title | URI | cosine similarity | description):
{candidate_skills}
"""


# ── ESCO helpers ───────────────────────────────────────────────────────────────

def _parse_esco_item(item: dict) -> dict:
    """
    Extract uri, title and description from a single ESCO search result item.
    The /search quick-mode response returns fragments:
      { "uri": "...", "title": "...", "className": "Skill", ... }
    The full-mode response nests labels under preferredLabel / description.
    We handle both shapes.
    """
    uri   = item.get("uri", "")
    # Quick-mode: flat "title" field
    title = item.get("title", "")
    # Full-mode fallback: preferredLabel.<lang>
    if not title:
        pl = item.get("preferredLabel", {})
        title = pl.get(ESCO_LANGUAGE) or next(iter(pl.values()), "") if pl else ""
    # Description: try flat key first, then nested object
    desc = item.get("description", "")
    if not desc:
        desc_obj = item.get("description", {})
        if isinstance(desc_obj, dict):
            desc = desc_obj.get(ESCO_LANGUAGE, "") or next(iter(desc_obj.values()), "")
    return {"uri": uri, "title": str(title), "description": str(desc)[:300]}


def fetch_esco_skills(limit: int = 500) -> list[dict]:
    """
    Pull skills from the ESCO REST API using the /search endpoint (quick mode).
    Paginates automatically up to `limit` results.
    Falls back to a small built-in seed list if the API is unreachable.

    ESCO API ref: GET /search
      ?text=       – omit to get all skills ordered by label
      ?type=skill  – restrict to Skill class
      ?language=en
      ?limit=100   – max page size supported by ESCO
      ?offset=N    – zero-based page offset
      ?full=false  – quick mode (lower latency)
    """
    print(f"  Fetching up to {limit} skills from ESCO API ({ESCO_API_BASE}/search) ...",
          end=" ", flush=True)
    try:
        url    = f"{ESCO_API_BASE}/search"
        page_size = min(limit, 100)   # ESCO caps at 100 per page
        params = {
            "language": ESCO_LANGUAGE,
            "type":     "skill",
            "limit":    page_size,
            "offset":   0,
            "full":     "false",       # quick mode — returns partial fragments
        }
        skills: list[dict] = []

        while len(skills) < limit:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            # ESCO /search response shape:
            # { "total": N, "_embedded": { "results": [ ... ] }, ... }
            items = data.get("_embedded", {}).get("results", [])
            if not items:
                break

            for item in items:
                parsed = _parse_esco_item(item)
                if parsed["uri"] and parsed["title"]:
                    skills.append(parsed)

            total_available = data.get("total", 0)
            if len(skills) >= total_available or len(skills) >= limit:
                break

            params["offset"] += page_size
            time.sleep(0.15)   # be polite to the public API

        print(f"OK ({len(skills)} skills fetched)")
        return skills[:limit]

    except Exception as e:
        print(f"FAILED ({e}) — using built-in seed list")
        return _builtin_seed_skills()


def fetch_esco_skills_for_query(query: str, limit: int = 50) -> list[dict]:
    """
    Search ESCO skills matching a specific text query using /search.
    Useful for targeted retrieval when you know a domain keyword.

    ESCO API ref: GET /search  (searchGet / searchQuickMode)
    """
    try:
        url    = f"{ESCO_API_BASE}/search"
        params = {
            "text":     query,
            "language": ESCO_LANGUAGE,
            "type":     "skill",
            "limit":    min(limit, 100),
            "offset":   0,
            "full":     "false",
        }
        resp  = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("_embedded", {}).get("results", [])
        return [_parse_esco_item(i) for i in items if i.get("uri")]
    except Exception:
        return []


def get_esco_skill_detail(uri: str) -> dict:
    """
    Fetch full details for a single ESCO skill by URI.
    ESCO API ref: GET /resource/skill?uri=<uri>  (resourceSkillGet)
    Returns dict with uri, title, description, skill_type, reuse_level.
    """
    try:
        url    = f"{ESCO_API_BASE}/resource/skill"
        params = {"uri": uri, "language": ESCO_LANGUAGE}
        resp   = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data   = resp.json()
        parsed = _parse_esco_item(data)
        # Extract skill type and reuse level from _links if present
        links  = data.get("_links", {})
        parsed["skill_type"]   = links.get("hasSkillType",  [{}])[0].get("title", "") if links.get("hasSkillType")  else ""
        parsed["reuse_level"]  = links.get("hasReuseLevel", [{}])[0].get("title", "") if links.get("hasReuseLevel") else ""
        return parsed
    except Exception:
        return {"uri": uri, "title": "", "description": "", "skill_type": "", "reuse_level": ""}


def _builtin_seed_skills() -> list[dict]:
    """Minimal fallback skill catalogue if ESCO API is unavailable."""
    return [
        {"uri": "esco:1",  "title": "Customer Service",             "description": "Providing support and assistance to customers"},
        {"uri": "esco:2",  "title": "Communication Skills",         "description": "Expressing ideas clearly in writing and speech"},
        {"uri": "esco:3",  "title": "Problem Solving",              "description": "Identifying and resolving issues effectively"},
        {"uri": "esco:4",  "title": "Teamwork",                     "description": "Collaborating effectively with others"},
        {"uri": "esco:5",  "title": "Financial Management",         "description": "Managing budgets, forecasts and financial reporting"},
        {"uri": "esco:6",  "title": "Stakeholder Engagement",       "description": "Building relationships with internal and external stakeholders"},
        {"uri": "esco:7",  "title": "Quality Assurance",            "description": "Ensuring outputs meet defined quality standards"},
        {"uri": "esco:8",  "title": "Project Management",           "description": "Planning and delivering projects on time and within scope"},
        {"uri": "esco:9",  "title": "Data Analysis",                "description": "Interpreting data to support decision-making"},
        {"uri": "esco:10", "title": "Health and Safety",            "description": "Ensuring compliance with health and safety regulations"},
        {"uri": "esco:11", "title": "Microsoft Office",             "description": "Using Word, Excel, PowerPoint and Outlook"},
        {"uri": "esco:12", "title": "Report Writing",               "description": "Producing clear written reports and documentation"},
        {"uri": "esco:13", "title": "Risk Management",              "description": "Identifying, assessing and mitigating risks"},
        {"uri": "esco:14", "title": "Leadership",                   "description": "Guiding and motivating teams to achieve goals"},
        {"uri": "esco:15", "title": "Mechanical Engineering",       "description": "Design and maintenance of mechanical systems"},
        {"uri": "esco:16", "title": "Welding",                      "description": "Joining metal components using welding techniques"},
        {"uri": "esco:17", "title": "Nuclear Safety",               "description": "Applying safety principles in nuclear environments"},
        {"uri": "esco:18", "title": "Budgeting and Forecasting",    "description": "Preparing and monitoring financial budgets"},
        {"uri": "esco:19", "title": "Compliance Management",        "description": "Ensuring adherence to regulatory requirements"},
        {"uri": "esco:20", "title": "IT Proficiency",               "description": "Using software tools and digital platforms effectively"},
    ]


# ── Matching helpers ───────────────────────────────────────────────────────────

def build_tfidf_index(skills: list[dict]):
    """Build a TF-IDF matrix over skill titles + descriptions."""
    corpus = [f"{s['title']} {s['description']}" for s in skills]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix     = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def retrieve_top_k(jd_text: str, skills: list[dict],
                   vectorizer, matrix, k: int = 15) -> list[dict]:
    """Return the top-k ESCO skills most similar to the JD via TF-IDF cosine."""
    jd_vec = vectorizer.transform([jd_text])
    sims   = cosine_similarity(jd_vec, matrix).flatten()
    top_idx = sims.argsort()[::-1][:k]
    return [
        {**skills[i], "similarity_score": round(float(sims[i]), 4)}
        for i in top_idx
    ]


def extract_json(text: str) -> dict:
    """Strip markdown fences and parse JSON."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(text)


# ── Per-role processing ────────────────────────────────────────────────────────

def process_row(row: pd.Series, skills: list[dict], vectorizer, matrix) -> dict:
    job_ref       = row["Job Ref ID"]
    job_title     = row["Job Title"]
    job_status    = row["Job Status"]
    function_     = str(row.get("Function", "") or "")
    sub_function  = str(row.get("Sub-function", "") or "")
    business_unit = str(row.get("Business unit", "") or "")
    jd_text       = str(row["Default Job Ad Job Description"] or "").strip()

    print(f"  [{job_ref}] {job_title} ...", end=" ", flush=True)

    if not jd_text or jd_text.lower() == "nan":
        print("SKIPPED (no JD)")
        return {
            "job_ref_id": job_ref, "job_title": job_title,
            "job_status": job_status, "function": function_,
            "sub_function": sub_function, "business_unit": business_unit,
            "skills": [], "error": "No job description text",
            "candidates_retrieved": 0, "candidates": [],
            "stats": {},
        }

    # Step 1a — TF-IDF retrieval from the bulk catalogue
    tfidf_candidates = retrieve_top_k(jd_text, skills, vectorizer, matrix, k=TOP_K_SKILLS)

    # Step 1b — additionally query ESCO /search with the job title for domain-specific skills
    api_candidates = fetch_esco_skills_for_query(job_title, limit=10)

    # Merge and deduplicate by URI
    seen_uris      = {c["uri"] for c in tfidf_candidates}
    all_candidates = list(tfidf_candidates)
    for c in api_candidates:
        if c["uri"] and c["uri"] not in seen_uris:
            c.setdefault("similarity_score", 0.0)
            all_candidates.append(c)
            seen_uris.add(c["uri"])

    # ── Cosine score statistics for this role ──────────────────────────────────
    cosine_scores = [c.get("similarity_score", 0.0) for c in all_candidates if c.get("similarity_score") is not None]
    score_stats = {
        "n_candidates": len(all_candidates),
        "cosine_min":   round(min(cosine_scores), 4)  if cosine_scores else 0,
        "cosine_max":   round(max(cosine_scores), 4)  if cosine_scores else 0,
        "cosine_mean":  round(statistics.mean(cosine_scores), 4)   if cosine_scores else 0,
        "cosine_median":round(statistics.median(cosine_scores), 4) if cosine_scores else 0,
        "cosine_stdev": round(statistics.stdev(cosine_scores), 4)  if len(cosine_scores) > 1 else 0,
    }

    candidate_list = "\n".join(
        f"- {c['title']} | {c['uri']} | sim={c.get('similarity_score', 0):.3f} | {c['description']}"
        for c in all_candidates
    )

    # Step 2 — ask Claude to validate and enrich
    prompt = ENRICHMENT_PROMPT.format(
        job_title=job_title,
        function=f"{function_} / {sub_function}".strip(" /"),
        job_description=jd_text[:5000],
        candidate_skills=candidate_list,
    )

    try:
        raw     = invoke_claude(prompt, system=SYSTEM_PROMPT)
        parsed  = extract_json(raw)
        matched = parsed.get("skills", [])

        # Step 3 — enrich each confirmed skill with full ESCO detail
        for skill in matched:
            uri = skill.get("esco_uri", "")
            if uri and uri.startswith("http"):
                detail = get_esco_skill_detail(uri)
                skill["skill_type"]  = detail.get("skill_type", "")
                skill["reuse_level"] = detail.get("reuse_level", "")

        n_confirmed    = len(matched)
        acceptance_rate= round(n_confirmed / max(len(all_candidates), 1), 4)

        # Add acceptance rate to stats
        score_stats["n_confirmed"]      = n_confirmed
        score_stats["acceptance_rate"]  = acceptance_rate

        print(
            f"OK  candidates={len(all_candidates)}  confirmed={n_confirmed}"
            f"  accept={acceptance_rate:.0%}"
            f"  cosine_mean={score_stats['cosine_mean']:.3f}"
        )
        return {
            "job_ref_id":           job_ref,
            "job_title":            job_title,
            "job_status":           job_status,
            "function":             function_,
            "sub_function":         sub_function,
            "business_unit":        business_unit,
            "skills":               matched,
            "candidates_retrieved": len(all_candidates),
            "candidates":           all_candidates,   # kept for frontend stats view
            "stats":                score_stats,
        }
    except json.JSONDecodeError as e:
        print(f"JSON PARSE ERROR: {e}")
        score_stats.update({"n_confirmed": 0, "acceptance_rate": 0})
        return {
            "job_ref_id": job_ref, "job_title": job_title,
            "job_status": job_status, "function": function_,
            "sub_function": sub_function, "business_unit": business_unit,
            "skills": [], "error": f"JSON parse error: {e}",
            "candidates_retrieved": len(all_candidates), "candidates": all_candidates,
            "stats": score_stats,
        }
    except Exception as e:
        print(f"ERROR: {e}")
        score_stats.update({"n_confirmed": 0, "acceptance_rate": 0})
        return {
            "job_ref_id": job_ref, "job_title": job_title,
            "job_status": job_status, "function": function_,
            "sub_function": sub_function, "business_unit": business_unit,
            "skills": [], "error": str(e),
            "candidates_retrieved": len(all_candidates), "candidates": all_candidates,
            "stats": score_stats,
        }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print("  Approach 2 – Embedding-Based Matching (TF-IDF + ESCO)  [Statistical]")
    print(f"{'='*70}\n")

    # Load ESCO catalogue
    print("Step 1: Load ESCO skill catalogue")
    esco_skills = fetch_esco_skills(limit=ESCO_SKILLS_LIMIT)

    # Build TF-IDF index
    print("Step 2: Build TF-IDF index over ESCO skills ...", end=" ", flush=True)
    vectorizer, matrix = build_tfidf_index(esco_skills)
    print("OK")

    # Load Excel
    print(f"\nStep 3: Load job descriptions from {EXCEL_FILE}")
    df = pd.read_excel(EXCEL_FILE)
    print(f"Total rows: {len(df)}")
    if TEST_MODE:
        df = df.head(TEST_LIMIT)
        print(f"⚠️  TEST MODE — processing first {TEST_LIMIT} records only\n")
    else:
        print(f"🚀 FULL MODE — processing all {len(df)} records\n")

    # Process each role
    print("Step 4: Match and enrich skills per role")
    print(f"  {'Role':<45} {'Cands':>6} {'Conf':>5} {'Accept':>7} {'CosMean':>8}")
    print(f"  {'-'*45} {'-'*6} {'-'*5} {'-'*7} {'-'*8}")
    results = []
    for _, row in df.iterrows():
        result = process_row(row, esco_skills, vectorizer, matrix)
        results.append(result)

    # ── Statistical summary table ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Statistical Summary")
    print(f"{'='*70}")
    print(f"  {'Role':<40} {'Cands':>6} {'Conf':>5} {'Accept%':>8} {'CosMean':>8} {'CosMin':>7} {'CosMax':>7}")
    print(f"  {'-'*40} {'-'*6} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")

    all_cosines     = []
    all_acceptances = []
    for r in results:
        s = r.get("stats", {})
        nc = s.get("n_candidates", r.get("candidates_retrieved", 0))
        nk = s.get("n_confirmed",  len(r.get("skills", [])))
        ar = s.get("acceptance_rate", 0)
        cm = s.get("cosine_mean", 0)
        cmin= s.get("cosine_min", 0)
        cmax= s.get("cosine_max", 0)
        print(f"  {r['job_title'][:40]:<40} {nc:>6} {nk:>5} {ar:>8.0%} {cm:>8.3f} {cmin:>7.3f} {cmax:>7.3f}")
        all_cosines.extend([c.get("similarity_score", 0) for c in r.get("candidates", [])])
        if ar > 0:
            all_acceptances.append(ar)

    print(f"\n  {'OVERALL':}")
    if all_cosines:
        print(f"    Cosine scores  — min={min(all_cosines):.3f}  mean={statistics.mean(all_cosines):.3f}"
              f"  median={statistics.median(all_cosines):.3f}  max={max(all_cosines):.3f}"
              f"  n={len(all_cosines)}")
    if all_acceptances:
        print(f"    Acceptance rate — min={min(all_acceptances):.0%}  mean={statistics.mean(all_acceptances):.0%}"
              f"  max={max(all_acceptances):.0%}")
    total_skills = sum(len(r.get("skills", [])) for r in results)
    print(f"    Roles processed: {len(results)}   Skills confirmed: {total_skills}")
    print(f"{'='*70}\n")

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON saved to: {OUTPUT_JSON}")

    # Flatten to skills CSV
    rows_flat = []
    for role in results:
        if role.get("skills"):
            for skill in role["skills"]:
                rows_flat.append({
                    "job_ref_id":        role["job_ref_id"],
                    "job_title":         role["job_title"],
                    "job_status":        role["job_status"],
                    "function":          role.get("function", ""),
                    "sub_function":      role.get("sub_function", ""),
                    "business_unit":     role.get("business_unit", ""),
                    "skill_name":        skill.get("skill_name", ""),
                    "esco_uri":          skill.get("esco_uri", ""),
                    "skill_type":        skill.get("skill_type", ""),
                    "reuse_level":       skill.get("reuse_level", ""),
                    "proficiency_level": skill.get("proficiency_level", ""),
                    "importance":        skill.get("importance", ""),
                    "evidence":          skill.get("evidence", ""),
                })
        else:
            rows_flat.append({
                "job_ref_id": role["job_ref_id"],
                "job_title":  role["job_title"],
                "job_status": role["job_status"],
                "error":      role.get("error", ""),
            })

    df_out = pd.DataFrame(rows_flat)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"✅ CSV  saved to: {OUTPUT_CSV}")

    # Save stats CSV
    stats_rows = []
    for r in results:
        s = r.get("stats", {})
        stats_rows.append({
            "job_ref_id":       r["job_ref_id"],
            "job_title":        r["job_title"],
            "job_status":       r["job_status"],
            "function":         r.get("function", ""),
            "candidates":       s.get("n_candidates", 0),
            "confirmed":        s.get("n_confirmed",  len(r.get("skills", []))),
            "acceptance_rate":  s.get("acceptance_rate", 0),
            "cosine_min":       s.get("cosine_min",  0),
            "cosine_max":       s.get("cosine_max",  0),
            "cosine_mean":      s.get("cosine_mean", 0),
            "cosine_median":    s.get("cosine_median", 0),
            "cosine_stdev":     s.get("cosine_stdev", 0),
            "error":            r.get("error", ""),
        })
    pd.DataFrame(stats_rows).to_csv(STATS_CSV, index=False, encoding="utf-8")
    print(f"✅ Stats CSV saved to: {STATS_CSV}")


if __name__ == "__main__":
    main()
