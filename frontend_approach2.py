"""
Streamlit Frontend – Approach 2: Embedding-Based Matching (TF-IDF + ESCO)
==========================================================================
Run:
    streamlit run frontend_approach2.py
"""

import json
import re
import time
import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bedrock_client import invoke_claude

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Approach 2 – Embedding Matching",
    page_icon="🔍",
    layout="wide",
)

EXCEL_FILE    = "050326 SR Job Description Details LE.xlsx"
ESCO_API_BASE = "https://esco.ec.europa.eu/api"

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
4. Provide a short evidence quote from the JD.

Return a JSON object with key "skills" — an array. Only include skills where relevant is true.

Each item:
{{
  "skill_name": "...",
  "esco_uri": "...",
  "proficiency_level": "...",
  "importance": "...",
  "evidence": "..."
}}

Job Title: {job_title}

Job Description (excerpt):
{job_description}

Candidate skills from ESCO:
{candidate_skills}
"""

IMPORTANCE_COLOURS = {"Core": "🔴", "Important": "🟠", "Nice to Have": "🟢"}
PROFICIENCY_COLOURS = {"Awareness": "⚪", "Working": "🔵", "Practitioner": "🟣", "Expert": "🟡"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(text)


@st.cache_data(show_spinner=False)
def load_excel():
    return pd.read_excel(EXCEL_FILE)


@st.cache_data(show_spinner="Fetching ESCO skills catalogue…")
def fetch_esco_skills(limit: int = 300) -> list[dict]:
    """Pull skills from ESCO API; fall back to built-in list if unavailable."""
    try:
        url    = f"{ESCO_API_BASE}/search"
        params = {"language": "en", "type": "skill", "limit": 100, "offset": 0, "full": "false"}
        skills = []
        while len(skills) < limit:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data  = resp.json()
            items = data.get("_embedded", {}).get("results", [])
            if not items:
                break
            for item in items:
                skills.append({
                    "uri":         item.get("uri", ""),
                    "title":       item.get("title", ""),
                    "description": item.get("description", "")[:300],
                })
            if len(skills) >= data.get("total", 0) or len(skills) >= limit:
                break
            params["offset"] += 100
            time.sleep(0.2)
        return skills
    except Exception:
        return _builtin_seed_skills()


def _builtin_seed_skills() -> list[dict]:
    return [
        {"uri": "esco:1",  "title": "Customer Service",            "description": "Providing support and assistance to customers"},
        {"uri": "esco:2",  "title": "Communication Skills",        "description": "Expressing ideas clearly in writing and speech"},
        {"uri": "esco:3",  "title": "Problem Solving",             "description": "Identifying and resolving issues effectively"},
        {"uri": "esco:4",  "title": "Teamwork",                    "description": "Collaborating effectively with others"},
        {"uri": "esco:5",  "title": "Financial Management",        "description": "Managing budgets, forecasts and financial reporting"},
        {"uri": "esco:6",  "title": "Stakeholder Engagement",      "description": "Building relationships with internal and external stakeholders"},
        {"uri": "esco:7",  "title": "Quality Assurance",           "description": "Ensuring outputs meet defined quality standards"},
        {"uri": "esco:8",  "title": "Project Management",          "description": "Planning and delivering projects on time and within scope"},
        {"uri": "esco:9",  "title": "Data Analysis",               "description": "Interpreting data to support decision-making"},
        {"uri": "esco:10", "title": "Health and Safety",           "description": "Ensuring compliance with health and safety regulations"},
        {"uri": "esco:11", "title": "Microsoft Office",            "description": "Using Word, Excel, PowerPoint and Outlook"},
        {"uri": "esco:12", "title": "Report Writing",              "description": "Producing clear written reports and documentation"},
        {"uri": "esco:13", "title": "Risk Management",             "description": "Identifying, assessing and mitigating risks"},
        {"uri": "esco:14", "title": "Leadership",                  "description": "Guiding and motivating teams to achieve goals"},
        {"uri": "esco:15", "title": "Mechanical Engineering",      "description": "Design and maintenance of mechanical systems"},
        {"uri": "esco:16", "title": "Welding",                     "description": "Joining metal components using welding techniques"},
        {"uri": "esco:17", "title": "Nuclear Safety",              "description": "Applying safety principles in nuclear environments"},
        {"uri": "esco:18", "title": "Budgeting and Forecasting",   "description": "Preparing and monitoring financial budgets"},
        {"uri": "esco:19", "title": "Compliance Management",       "description": "Ensuring adherence to regulatory requirements"},
        {"uri": "esco:20", "title": "IT Proficiency",              "description": "Using software tools and digital platforms effectively"},
        {"uri": "esco:21", "title": "Contract Management",         "description": "Managing contractual obligations and performance"},
        {"uri": "esco:22", "title": "Analytical Thinking",         "description": "Breaking down complex problems into components"},
        {"uri": "esco:23", "title": "Technical Drawing",           "description": "Reading and interpreting engineering drawings"},
        {"uri": "esco:24", "title": "Team Leadership",             "description": "Leading and developing high-performing teams"},
        {"uri": "esco:25", "title": "Environmental Compliance",    "description": "Adhering to environmental regulations and standards"},
    ]


@st.cache_resource(show_spinner=False)
def build_tfidf_index(skills_tuple):
    """Build TF-IDF matrix. Accepts a tuple of (title, description) for hashability."""
    corpus = [f"{t} {d}" for t, d in skills_tuple]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def retrieve_top_k(jd_text: str, skills: list[dict], vectorizer, matrix, k: int) -> list[dict]:
    jd_vec  = vectorizer.transform([jd_text])
    sims    = cosine_similarity(jd_vec, matrix).flatten()
    top_idx = sims.argsort()[::-1][:k]
    return [{**skills[i], "similarity_score": round(float(sims[i]), 4)} for i in top_idx]


def run_enrichment(job_title: str, jd_text: str, candidates: list[dict]) -> list[dict]:
    candidate_list = "\n".join(
        f"- {c['title']} (URI: {c['uri']}): {c['description']}" for c in candidates
    )
    prompt = ENRICHMENT_PROMPT.format(
        job_title=job_title,
        job_description=jd_text[:5000],
        candidate_skills=candidate_list,
    )
    raw    = invoke_claude(prompt, system=SYSTEM_PROMPT)
    parsed = extract_json(raw)
    return parsed.get("skills", [])


def flatten_results(results: list[dict]) -> pd.DataFrame:
    rows = []
    for role in results:
        for skill in role.get("skills", []):
            rows.append({
                "Job Ref ID":        role["job_ref_id"],
                "Job Title":         role["job_title"],
                "Job Status":        role["job_status"],
                "Skill":             skill.get("skill_name", ""),
                "ESCO URI":          skill.get("esco_uri", ""),
                "Proficiency Level": skill.get("proficiency_level", ""),
                "Importance":        skill.get("importance", ""),
                "Evidence":          skill.get("evidence", ""),
            })
    return pd.DataFrame(rows)


# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("🔍 Approach 2 — Embedding-Based Matching (ESCO Catalogue)")
st.caption("Retrieves candidate skills from the ESCO taxonomy via TF-IDF similarity, then uses Claude to validate relevance and assign proficiency levels.")

with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
    1. Loads the **ESCO skill catalogue** (EU reference taxonomy — 13,890 skills)
    2. Builds a **TF-IDF index** over skill titles and descriptions
    3. For each job description, retrieves the **top-K most similar ESCO skills** using cosine similarity
    4. Sends the shortlist to **Claude Sonnet** on **Amazon Bedrock** to validate relevance and assign proficiency + importance
    5. Results are displayed with the originating ESCO URI for traceability

    **Strengths:** Consistent vocabulary, governed catalogue, traceable  
    **Watch-outs:** Catalogue quality is the bottleneck; skills not in ESCO may be missed
    """)

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading Excel file…"):
    df_raw = load_excel()

st.success(f"Loaded **{len(df_raw):,}** roles from the dataset")

# ── Load ESCO catalogue ────────────────────────────────────────────────────────
esco_skills = fetch_esco_skills(limit=300)

catalogue_source = "ESCO API" if any(not s["uri"].startswith("esco:") for s in esco_skills) else "built-in seed list"
st.info(f"📚 Skill catalogue: **{len(esco_skills)} skills** loaded from {catalogue_source}")

# ── Build TF-IDF index ─────────────────────────────────────────────────────────
skills_tuple = tuple((s["title"], s["description"]) for s in esco_skills)
with st.spinner("Building TF-IDF index…"):
    vectorizer, matrix = build_tfidf_index(skills_tuple)

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    sample_size = st.slider(
        "Number of roles to process",
        min_value=1, max_value=50, value=20, step=1,
    )

    top_k = st.slider(
        "Top-K candidate skills per role (TF-IDF retrieval)",
        min_value=5, max_value=30, value=15, step=5,
    )

    status_options    = sorted(df_raw["Job Status"].dropna().unique().tolist())
    selected_statuses = st.multiselect(
        "Filter by Job Status",
        options=status_options,
        default=status_options,
    )

    run_btn = st.button("▶️ Run Matching", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**Catalogue preview**")
    st.dataframe(
        pd.DataFrame(esco_skills)[["title", "description"]].head(10),
        use_container_width=True,
        height=220,
    )

# ── Filter and preview ─────────────────────────────────────────────────────────
df_filtered = df_raw[df_raw["Job Status"].isin(selected_statuses)].head(sample_size)

st.subheader(f"📋 Roles selected for processing ({len(df_filtered)})")
st.dataframe(
    df_filtered[["Job Ref ID", "Job Title", "Job Status", "Business unit", "Function"]],
    use_container_width=True,
    height=220,
)

# ── Run matching ───────────────────────────────────────────────────────────────
if run_btn:
    results = []
    total   = len(df_filtered)

    progress_bar = st.progress(0, text="Starting…")
    status_text  = st.empty()

    for i, (_, row) in enumerate(df_filtered.iterrows()):
        job_ref   = row["Job Ref ID"]
        job_title = row["Job Title"]
        job_status= row["Job Status"]
        jd_text   = str(row.get("Default Job Ad Job Description", "") or "").strip()

        progress_bar.progress(i / total, text=f"Processing {i+1}/{total}: {job_title}")
        status_text.info(f"🔄 **[{job_ref}]** {job_title}")

        if not jd_text or jd_text.lower() == "nan":
            results.append({
                "job_ref_id": job_ref, "job_title": job_title,
                "job_status": job_status, "skills": [],
                "error": "No JD text",
            })
            continue

        try:
            # Step 1 — TF-IDF retrieval
            candidates = retrieve_top_k(jd_text, esco_skills, vectorizer, matrix, k=top_k)
            # Step 2 — Claude enrichment
            skills = run_enrichment(job_title, jd_text, candidates)
            results.append({
                "job_ref_id":            job_ref,
                "job_title":             job_title,
                "job_status":            job_status,
                "skills":                skills,
                "candidates_retrieved":  len(candidates),
            })
        except Exception as e:
            results.append({
                "job_ref_id": job_ref, "job_title": job_title,
                "job_status": job_status, "skills": [], "error": str(e),
            })

    progress_bar.progress(1.0, text="Done!")
    status_text.empty()
    st.session_state["a2_results"] = results

# ── Display results ────────────────────────────────────────────────────────────
if "a2_results" in st.session_state:
    results      = st.session_state["a2_results"]
    total_skills = sum(len(r["skills"]) for r in results)
    errors       = sum(1 for r in results if r.get("error"))

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Roles Processed", len(results))
    col2.metric("Total Skills Confirmed", total_skills)
    col3.metric("Errors / Skipped", errors)

    # ── Per-role expanders ─────────────────────────────────────────────────────
    st.subheader("📂 Results by Role")
    for role in results:
        label = f"[{role['job_ref_id']}] {role['job_title']}  —  {len(role['skills'])} skills confirmed"
        with st.expander(label):
            if role.get("error") and not role["skills"]:
                st.warning(f"⚠️ {role['error']}")
                continue

            for skill in role["skills"]:
                imp  = skill.get("importance", "")
                prof = skill.get("proficiency_level", "")
                uri  = skill.get("esco_uri", "")
                st.markdown(
                    f"{IMPORTANCE_COLOURS.get(imp,'⚪')} **{skill.get('skill_name','')}** "
                    f"&nbsp;|&nbsp; {PROFICIENCY_COLOURS.get(prof,'⚪')} {prof} "
                    f"&nbsp;|&nbsp; {imp}"
                )
                if uri:
                    st.caption(f"🔗 ESCO URI: `{uri}`")
                st.caption(f"📎 *\"{skill.get('evidence','')}\"*")

    # ── Flat table + download ──────────────────────────────────────────────────
    st.subheader("📊 Full Results Table")
    df_out = flatten_results(results)
    st.dataframe(df_out, use_container_width=True, height=400)

    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name="approach2_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Charts ─────────────────────────────────────────────────────────────────
    if not df_out.empty:
        st.subheader("📈 Analytics")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Skills by Importance**")
            imp_counts = df_out["Importance"].value_counts().reset_index()
            imp_counts.columns = ["Importance", "Count"]
            st.bar_chart(imp_counts.set_index("Importance"))

        with c2:
            st.markdown("**Skills by Proficiency Level**")
            prof_counts = df_out["Proficiency Level"].value_counts().reset_index()
            prof_counts.columns = ["Proficiency Level", "Count"]
            st.bar_chart(prof_counts.set_index("Proficiency Level"))

        st.markdown("**Top 20 most common skills across all roles**")
        top_skills = df_out["Skill"].value_counts().head(20).reset_index()
        top_skills.columns = ["Skill", "Count"]
        st.bar_chart(top_skills.set_index("Skill"))
