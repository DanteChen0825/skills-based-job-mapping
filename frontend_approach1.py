"""
Streamlit Frontend – Approach 1: Direct LLM Skill Extraction
=============================================================
Run:
    streamlit run frontend_approach1.py
"""

import json
import re
import io
import pandas as pd
import streamlit as st
from bedrock_client import invoke_claude

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Approach 1 – Direct LLM Extraction",
    page_icon="🧠",
    layout="wide",
)

EXCEL_FILE = "050326 SR Job Description Details LE.xlsx"

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

IMPORTANCE_COLOURS = {
    "Core":           "🔴",
    "Important":      "🟠",
    "Nice to Have":   "🟢",
}
PROFICIENCY_COLOURS = {
    "Awareness":    "⚪",
    "Working":      "🔵",
    "Practitioner": "🟣",
    "Expert":       "🟡",
}
CATEGORY_BADGE = {
    "Technical":            "🔧",
    "Soft":                 "💬",
    "Domain":               "📚",
    "Tool/System":          "🖥️",
    "Compliance/Regulatory":"⚖️",
}


def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(text)


@st.cache_data(show_spinner=False)
def load_excel():
    return pd.read_excel(EXCEL_FILE)


def run_extraction(job_ref, job_title, jd_text: str) -> dict:
    prompt = EXTRACTION_PROMPT.format(
        job_title=job_title,
        job_description=jd_text[:6000],
    )
    raw = invoke_claude(prompt, system=SYSTEM_PROMPT)
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
                "Category":          skill.get("category", ""),
                "Proficiency Level": skill.get("proficiency_level", ""),
                "Importance":        skill.get("importance", ""),
                "Evidence":          skill.get("evidence", ""),
            })
    return pd.DataFrame(rows)


# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("🧠 Approach 1 — Direct LLM Skill Extraction")
st.caption("Sends job descriptions to Claude via Amazon Bedrock and extracts skills directly — no external catalogue needed.")

with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
    1. Loads job descriptions from the Excel file
    2. For each selected role, sends the JD to **Claude Sonnet** on **Amazon Bedrock**
    3. Claude identifies skills, assigns proficiency level and importance, and provides evidence from the text
    4. Results are displayed as an interactive table and can be downloaded as CSV

    **Strengths:** Fast, no external catalogue, easy to iterate  
    **Watch-outs:** Inconsistency across similar roles, no shared vocabulary enforced
    """)

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading Excel file…"):
    df_raw = load_excel()

st.success(f"Loaded **{len(df_raw):,}** roles from the dataset")

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    sample_size = st.slider(
        "Number of roles to process",
        min_value=1, max_value=50, value=20, step=1,
    )

    status_options = sorted(df_raw["Job Status"].dropna().unique().tolist())
    selected_statuses = st.multiselect(
        "Filter by Job Status",
        options=status_options,
        default=status_options,
    )

    run_btn = st.button("▶️ Run Extraction", type="primary", use_container_width=True)

# ── Filter and preview ─────────────────────────────────────────────────────────
df_filtered = df_raw[df_raw["Job Status"].isin(selected_statuses)].head(sample_size)

st.subheader(f"📋 Roles selected for processing ({len(df_filtered)})")
st.dataframe(
    df_filtered[["Job Ref ID", "Job Title", "Job Status", "Business unit", "Function"]],
    use_container_width=True,
    height=220,
)

# ── Run extraction ─────────────────────────────────────────────────────────────
if run_btn:
    results = []
    total = len(df_filtered)

    progress_bar  = st.progress(0, text="Starting…")
    status_text   = st.empty()
    results_area  = st.container()

    for i, (_, row) in enumerate(df_filtered.iterrows()):
        job_ref   = row["Job Ref ID"]
        job_title = row["Job Title"]
        job_status= row["Job Status"]
        jd_text   = str(row.get("Default Job Ad Job Description", "") or "").strip()

        progress_bar.progress((i) / total, text=f"Processing {i+1}/{total}: {job_title}")
        status_text.info(f"🔄 **[{job_ref}]** {job_title}")

        if not jd_text or jd_text.lower() == "nan":
            results.append({
                "job_ref_id": job_ref, "job_title": job_title,
                "job_status": job_status, "skills": [],
                "error": "No JD text",
            })
            continue

        try:
            skills = run_extraction(job_ref, job_title, jd_text)
            results.append({
                "job_ref_id": job_ref, "job_title": job_title,
                "job_status": job_status, "skills": skills,
            })
        except Exception as e:
            results.append({
                "job_ref_id": job_ref, "job_title": job_title,
                "job_status": job_status, "skills": [],
                "error": str(e),
            })

    progress_bar.progress(1.0, text="Done!")
    status_text.empty()

    st.session_state["a1_results"] = results

# ── Display results ────────────────────────────────────────────────────────────
if "a1_results" in st.session_state:
    results = st.session_state["a1_results"]
    total_skills = sum(len(r["skills"]) for r in results)
    errors       = sum(1 for r in results if r.get("error"))

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Roles Processed", len(results))
    col2.metric("Total Skills Extracted", total_skills)
    col3.metric("Errors / Skipped", errors)

    # ── Per-role expanders ─────────────────────────────────────────────────────
    st.subheader("📂 Results by Role")
    for role in results:
        label = f"[{role['job_ref_id']}] {role['job_title']}  —  {len(role['skills'])} skills"
        with st.expander(label):
            if role.get("error") and not role["skills"]:
                st.warning(f"⚠️ {role['error']}")
                continue

            for skill in role["skills"]:
                imp  = skill.get("importance", "")
                prof = skill.get("proficiency_level", "")
                cat  = skill.get("category", "")
                st.markdown(
                    f"{IMPORTANCE_COLOURS.get(imp,'⚪')} **{skill.get('skill_name','')}** "
                    f"&nbsp;|&nbsp; {CATEGORY_BADGE.get(cat,'🔹')} {cat} "
                    f"&nbsp;|&nbsp; {PROFICIENCY_COLOURS.get(prof,'⚪')} {prof} "
                    f"&nbsp;|&nbsp; {imp}"
                )
                st.caption(f"📎 *\"{skill.get('evidence','')}\"*")

    # ── Flat table + download ──────────────────────────────────────────────────
    st.subheader("📊 Full Results Table")
    df_out = flatten_results(results)
    st.dataframe(df_out, use_container_width=True, height=400)

    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name="approach1_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Charts ─────────────────────────────────────────────────────────────────
    if not df_out.empty:
        st.subheader("📈 Analytics")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Skills by Category**")
            cat_counts = df_out["Category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            st.bar_chart(cat_counts.set_index("Category"))

        with c2:
            st.markdown("**Skills by Importance**")
            imp_counts = df_out["Importance"].value_counts().reset_index()
            imp_counts.columns = ["Importance", "Count"]
            st.bar_chart(imp_counts.set_index("Importance"))
