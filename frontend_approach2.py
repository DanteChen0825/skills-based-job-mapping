# Approach 2 Frontend -- ESCO Embedding Matching (Statistical Edition)
# Run: python -m streamlit run frontend_approach2.py --server.port 8502 --server.headless true
#
# TEST_MODE = True  => process only TEST_LIMIT (5) records from selection
# TEST_MODE = False => process all selected records

import json
import re
import time
import statistics
import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bedrock_client import invoke_claude

st.set_page_config(
    page_title="Approach 2 - ESCO Matching",
    page_icon="🔍",
    layout="wide",
)

EXCEL_FILE    = "050326 SR Job Description Details LE.xlsx"
ESCO_API_BASE = "https://esco.ec.europa.eu/api"
TEST_LIMIT    = 5

IMPORTANCE_COLOUR  = {"Core": "🔴", "Important": "🟠", "Nice to Have": "🟢"}
PROFICIENCY_COLOUR = {"Awareness": "⚪", "Working": "🔵", "Practitioner": "🟣", "Expert": "🟡"}
IMPORTANCE_ORDER   = {"Core": 0, "Important": 1, "Nice to Have": 2}
PROFICIENCY_ORDER  = {"Awareness": 0, "Working": 1, "Practitioner": 2, "Expert": 3}

SYSTEM_PROMPT = (
    "You are an expert HR analyst specialising in skills-based job mapping. "
    "Validate a shortlist of ESCO skills against a job description. "
    "Always respond with valid JSON only - no prose, no markdown fences."
)

ENRICHMENT_PROMPT = """You are given a job description and a shortlist of candidate skills retrieved from the ESCO taxonomy via TF-IDF cosine similarity.

For each skill decide:
1. Is it genuinely relevant to this role?
2. If relevant - proficiency level: ["Awareness","Working","Practitioner","Expert"]
3. If relevant - importance: ["Core","Important","Nice to Have"]
4. Short evidence quote (20 words max) from the JD.

Return JSON with key "skills" - array of confirmed skills only.
Each item: {{"skill_name":"...","esco_uri":"...","proficiency_level":"...","importance":"...","evidence":"..."}}

Job Title: {job_title}
Function: {function}

Job Description (excerpt):
{job_description}

Candidate ESCO skills (name | URI | cosine sim | description):
{candidate_skills}"""


# ── Data helpers ────────────────────────────────────────────────────────────────

def extract_json(text):
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(text)


@st.cache_data(show_spinner=False)
def load_excel():
    return pd.read_excel(EXCEL_FILE)


@st.cache_data(show_spinner="Fetching ESCO skills catalogue...")
def fetch_esco_skills(limit=500):
    try:
        url    = f"{ESCO_API_BASE}/search"
        params = {"language": "en", "type": "skill", "limit": 100, "offset": 0, "full": "false"}
        skills = []
        while len(skills) < limit:
            resp  = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data  = resp.json()
            items = data.get("_embedded", {}).get("results", [])
            if not items:
                break
            for item in items:
                uri   = item.get("uri", "")
                title = item.get("title", "")
                desc  = str(item.get("description", ""))[:300]
                if uri and title:
                    skills.append({"uri": uri, "title": title, "description": desc})
            if len(skills) >= data.get("total", 0) or len(skills) >= limit:
                break
            params["offset"] += 100
            time.sleep(0.15)
        return skills[:limit]
    except Exception:
        return _builtin_seed()


def _builtin_seed():
    return [
        {"uri": "esco:1",  "title": "Customer Service",          "description": "Providing support and assistance to customers"},
        {"uri": "esco:2",  "title": "Communication Skills",      "description": "Expressing ideas clearly in writing and speech"},
        {"uri": "esco:3",  "title": "Problem Solving",           "description": "Identifying and resolving issues effectively"},
        {"uri": "esco:4",  "title": "Teamwork",                  "description": "Collaborating effectively with others"},
        {"uri": "esco:5",  "title": "Financial Management",      "description": "Managing budgets, forecasts and financial reporting"},
        {"uri": "esco:6",  "title": "Stakeholder Engagement",    "description": "Building relationships with stakeholders"},
        {"uri": "esco:7",  "title": "Quality Assurance",         "description": "Ensuring outputs meet defined quality standards"},
        {"uri": "esco:8",  "title": "Project Management",        "description": "Planning and delivering projects on time and scope"},
        {"uri": "esco:9",  "title": "Data Analysis",             "description": "Interpreting data to support decision-making"},
        {"uri": "esco:10", "title": "Health and Safety",         "description": "Ensuring compliance with health and safety regulations"},
        {"uri": "esco:11", "title": "Microsoft Office",          "description": "Using Word, Excel, PowerPoint and Outlook"},
        {"uri": "esco:12", "title": "Report Writing",            "description": "Producing clear written reports and documentation"},
        {"uri": "esco:13", "title": "Risk Management",           "description": "Identifying, assessing and mitigating risks"},
        {"uri": "esco:14", "title": "Leadership",                "description": "Guiding and motivating teams to achieve goals"},
        {"uri": "esco:15", "title": "Mechanical Engineering",    "description": "Design and maintenance of mechanical systems"},
        {"uri": "esco:16", "title": "Welding",                   "description": "Joining metal components using welding techniques"},
        {"uri": "esco:17", "title": "Nuclear Safety",            "description": "Applying safety principles in nuclear environments"},
        {"uri": "esco:18", "title": "Budgeting and Forecasting", "description": "Preparing and monitoring financial budgets"},
        {"uri": "esco:19", "title": "Compliance Management",     "description": "Ensuring adherence to regulatory requirements"},
        {"uri": "esco:20", "title": "IT Proficiency",            "description": "Using software tools and digital platforms effectively"},
        {"uri": "esco:21", "title": "Contract Management",       "description": "Managing contractual obligations and performance"},
        {"uri": "esco:22", "title": "Analytical Thinking",       "description": "Breaking down complex problems into components"},
        {"uri": "esco:23", "title": "Technical Drawing",         "description": "Reading and interpreting engineering drawings"},
        {"uri": "esco:24", "title": "Team Leadership",           "description": "Leading and developing high-performing teams"},
        {"uri": "esco:25", "title": "Environmental Compliance",  "description": "Adhering to environmental regulations and standards"},
    ]


@st.cache_resource(show_spinner=False)
def build_tfidf_index(skills_tuple):
    corpus     = [f"{t} {d}" for t, d in skills_tuple]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix     = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def retrieve_top_k(jd_text, skills, vectorizer, matrix, k):
    jd_vec  = vectorizer.transform([jd_text])
    sims    = cosine_similarity(jd_vec, matrix).flatten()
    top_idx = sims.argsort()[::-1][:k]
    return [{**skills[i], "similarity_score": round(float(sims[i]), 4)} for i in top_idx]


def run_enrichment(job_title, function_, jd_text, candidates):
    candidate_list = "\n".join(
        f"- {c['title']} | {c['uri']} | sim={c.get('similarity_score',0):.3f} | {c['description']}"
        for c in candidates
    )
    prompt = ENRICHMENT_PROMPT.format(
        job_title=job_title,
        function=function_,
        job_description=jd_text[:5000],
        candidate_skills=candidate_list,
    )
    raw    = invoke_claude(prompt, system=SYSTEM_PROMPT)
    parsed = extract_json(raw)
    return parsed.get("skills", [])


def compute_role_stats(candidates, confirmed):
    scores = [c.get("similarity_score", 0.0) for c in candidates if c.get("similarity_score") is not None]
    nc, nk = len(candidates), len(confirmed)
    return {
        "n_candidates":    nc,
        "n_confirmed":     nk,
        "acceptance_rate": round(nk / max(nc, 1), 4),
        "cosine_min":      round(min(scores), 4)               if scores else 0,
        "cosine_max":      round(max(scores), 4)               if scores else 0,
        "cosine_mean":     round(statistics.mean(scores), 4)   if scores else 0,
        "cosine_median":   round(statistics.median(scores), 4) if scores else 0,
        "cosine_stdev":    round(statistics.stdev(scores), 4)  if len(scores) > 1 else 0,
    }


def flatten_results(results):
    rows = []
    for role in results:
        for skill in role.get("skills", []):
            rows.append({
                "Job Ref ID":        role["job_ref_id"],
                "Job Title":         role["job_title"],
                "Job Status":        role["job_status"],
                "Function":          role.get("function", ""),
                "Sub-function":      role.get("sub_function", ""),
                "Skill":             skill.get("skill_name", ""),
                "ESCO URI":          skill.get("esco_uri", ""),
                "Proficiency Level": skill.get("proficiency_level", ""),
                "Importance":        skill.get("importance", ""),
                "Evidence":          skill.get("evidence", ""),
            })
    return pd.DataFrame(rows)


def refresh_live_charts(results, slots):
    df = flatten_results(results)
    if df.empty:
        return
    imp_counts = (
        df["Importance"].value_counts()
        .reindex(["Core", "Important", "Nice to Have"], fill_value=0)
        .rename_axis("Importance").reset_index(name="Count")
    )
    slots["importance"].bar_chart(imp_counts.set_index("Importance"), height=200)
    prof_counts = (
        df["Proficiency Level"].value_counts()
        .reindex(["Awareness", "Working", "Practitioner", "Expert"], fill_value=0)
        .rename_axis("Level").reset_index(name="Count")
    )
    slots["proficiency"].bar_chart(prof_counts.set_index("Level"), height=200)
    top15 = df["Skill"].value_counts().head(15).rename_axis("Skill").reset_index(name="Count")
    slots["top_skills"].bar_chart(top15.set_index("Skill"), height=220)
    acc_rows = [
        {"Role": r["job_title"][:28], "Candidates": r.get("candidates_retrieved", 0),
         "Confirmed": len(r.get("skills", []))}
        for r in results if r.get("candidates_retrieved", 0) > 0
    ]
    if acc_rows:
        slots["acceptance"].bar_chart(pd.DataFrame(acc_rows).set_index("Role"), height=200)


# ══════════════════════════════════════════════════════════════════════════════
# Page header
# ══════════════════════════════════════════════════════════════════════════════
st.title("Approach 2 - ESCO Catalogue Matching (TF-IDF + Claude)")
st.caption(
    "Retrieves candidate skills from the ESCO taxonomy via TF-IDF cosine similarity, "
    "then uses Claude Sonnet to validate relevance and assign proficiency + importance."
)

with st.expander("How it works", expanded=False):
    st.markdown("""
**Step 1 - Catalogue retrieval (TF-IDF)**
Up to 500 ESCO skills are indexed. For each JD, cosine similarity finds the top-K closest candidates.

**Step 2 - LLM validation (Claude)**
Claude reviews the candidate shortlist, rejects irrelevant matches, and assigns
proficiency level + importance + a verbatim evidence quote.

**Statistical outputs:**
- TF-IDF cosine score per candidate (accepted vs rejected)
- Acceptance rate per role (confirmed / candidates)
- Cosine score distribution across all retrieved candidates
- Per-role statistics table with min/mean/max cosine, stdev
    """)

# ══════════════════════════════════════════════════════════════════════════════
# Load data + ESCO catalogue
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading Excel file..."):
    df_raw = load_excel()
st.success(f"Loaded **{len(df_raw):,}** roles from the dataset")

esco_skills      = fetch_esco_skills(limit=500)
catalogue_source = "ESCO API" if any(not s["uri"].startswith("esco:") for s in esco_skills) else "built-in seed"
st.info(f"ESCO catalogue: **{len(esco_skills):,} skills** loaded from {catalogue_source}")

skills_tuple = tuple((s["title"], s["description"]) for s in esco_skills)
with st.spinner("Building TF-IDF index..."):
    vectorizer, matrix = build_tfidf_index(skills_tuple)

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Settings")

    test_mode = st.toggle(
        "Test Mode (5 records only)",
        value=True,
        help="ON = process only 5 roles from your selection. OFF = process all selected roles.",
    )
    if test_mode:
        st.caption(f"Test Mode ON - will process first {TEST_LIMIT} of your selection")
    else:
        st.caption("Full Mode ON - will process ALL selected roles")

    st.divider()
    top_k = st.slider("Top-K candidates per role (TF-IDF)", min_value=5, max_value=40, value=20, step=5)

    status_options    = sorted(df_raw["Job Status"].dropna().unique().tolist())
    selected_statuses = st.multiselect("Filter by Job Status", status_options, default=status_options)

    function_options  = ["All"] + sorted(df_raw["Function"].dropna().unique().tolist())
    selected_function = st.selectbox("Filter by Function", function_options)

    st.divider()
    st.markdown("**Select roles to process**")
    st.caption("Search by title, ref ID, or function below.")

# ── Filter dataframe ──────────────────────────────────────────────────────────
df_filtered = df_raw[df_raw["Job Status"].isin(selected_statuses)].copy()
if selected_function != "All":
    df_filtered = df_filtered[df_filtered["Function"] == selected_function]

# ── Role search + multi-select ─────────────────────────────────────────────────
search_query = st.text_input("Search roles", placeholder="e.g. Finance, Nuclear, 31961 ...")

if search_query.strip():
    mask = (
        df_filtered["Job Title"].str.contains(search_query, case=False, na=False)
        | df_filtered["Job Ref ID"].astype(str).str.contains(search_query, na=False)
        | df_filtered["Function"].str.contains(search_query, case=False, na=False)
        | df_filtered["Sub-function"].str.contains(search_query, case=False, na=False)
    )
    df_display = df_filtered[mask].head(100)
else:
    df_display = df_filtered.head(100)

role_labels = {
    f"[{row['Job Ref ID']}] {row['Job Title']} ({row['Job Status']})": idx
    for idx, row in df_display.iterrows()
}

with st.sidebar:
    btn1, btn2 = st.columns(2)
    if btn1.button("Select first 5"):
        st.session_state["a2_sel"] = list(role_labels.keys())[:5]
    if btn2.button("Clear all"):
        st.session_state["a2_sel"] = []

    selected_labels = st.multiselect(
        "Roles",
        options=list(role_labels.keys()),
        default=st.session_state.get("a2_sel", []),
        key="a2_sel",
        label_visibility="collapsed",
    )

    st.divider()
    run_btn = st.button("Run Matching", type="primary", use_container_width=True)

    st.divider()
    with st.expander("ESCO catalogue sample"):
        st.dataframe(pd.DataFrame(esco_skills)[["title", "description"]].head(15), height=260)

selected_indices = [role_labels[lbl] for lbl in selected_labels if lbl in role_labels]
df_selected      = df_filtered.loc[selected_indices] if selected_indices else pd.DataFrame()
df_to_process    = df_selected.head(TEST_LIMIT) if test_mode else df_selected

if not df_selected.empty:
    st.subheader(f"{len(df_selected)} role(s) selected  /  {len(df_to_process)} will be processed")
    st.dataframe(
        df_selected[["Job Ref ID", "Job Title", "Job Status", "Business unit", "Function", "Sub-function"]],
        height=200,
    )
else:
    st.info("Use the sidebar to search and select roles, then click Run Matching.")
    st.subheader("Available roles (first 100 shown)")
    st.dataframe(
        df_display[["Job Ref ID", "Job Title", "Job Status", "Business unit", "Function"]],
        height=300,
    )

# ══════════════════════════════════════════════════════════════════════════════
# Processing with live charts
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    if df_to_process.empty:
        st.warning("No roles to process. Select at least one role in the sidebar.")
        st.stop()

    results = []
    total   = len(df_to_process)

    progress_bar = st.progress(0, text="Starting...")
    status_text  = st.empty()

    st.divider()
    st.markdown("### Live Statistics  (updates after each role)")
    lm1, lm2, lm3, lm4, lm5 = st.columns(5)
    slot_done   = lm1.empty(); slot_done.metric("Roles done",          "0")
    slot_conf   = lm2.empty(); slot_conf.metric("Skills confirmed",    "0")
    slot_cands  = lm3.empty(); slot_cands.metric("Candidates checked", "0")
    slot_accept = lm4.empty(); slot_accept.metric("Avg acceptance",    "-")
    slot_err    = lm5.empty(); slot_err.metric("Errors",               "0")

    st.markdown("#### Live charts")
    live_c1, live_c2 = st.columns(2)
    with live_c1:
        st.markdown("**Importance**");         slot_imp  = st.empty()
        st.markdown("**Candidates vs Confirmed**"); slot_acc  = st.empty()
    with live_c2:
        st.markdown("**Proficiency**");        slot_prof = st.empty()
        st.markdown("**Top 15 Skills**");      slot_top  = st.empty()

    live_slots = {"importance": slot_imp, "proficiency": slot_prof,
                  "top_skills": slot_top, "acceptance": slot_acc}

    for i, (_, row) in enumerate(df_to_process.iterrows()):
        job_ref       = row["Job Ref ID"]
        job_title     = row["Job Title"]
        job_status    = row["Job Status"]
        function_     = str(row.get("Function", "")      or "")
        sub_function  = str(row.get("Sub-function", "")  or "")
        business_unit = str(row.get("Business unit", "") or "")
        jd_text       = str(row.get("Default Job Ad Job Description", "") or "").strip()

        progress_bar.progress(i / total, text=f"Processing {i+1}/{total}: {job_title}")
        status_text.info(f"[{job_ref}] {job_title} - TF-IDF retrieval...")

        if not jd_text or jd_text.lower() == "nan":
            results.append({
                "job_ref_id": job_ref, "job_title": job_title,
                "job_status": job_status, "function": function_,
                "sub_function": sub_function, "business_unit": business_unit,
                "skills": [], "error": "No JD text",
                "candidates_retrieved": 0, "candidates": [], "stats": {},
            })
        else:
            try:
                candidates = retrieve_top_k(jd_text, esco_skills, vectorizer, matrix, k=top_k)
                status_text.info(f"[{job_ref}] {job_title} - {len(candidates)} candidates -> Claude validating...")
                skills = run_enrichment(
                    job_title,
                    f"{function_} / {sub_function}".strip(" /"),
                    jd_text,
                    candidates,
                )
                role_stats = compute_role_stats(candidates, skills)
                results.append({
                    "job_ref_id":           job_ref,
                    "job_title":            job_title,
                    "job_status":           job_status,
                    "function":             function_,
                    "sub_function":         sub_function,
                    "business_unit":        business_unit,
                    "skills":               skills,
                    "candidates_retrieved": len(candidates),
                    "candidates":           candidates,
                    "stats":                role_stats,
                    "jd_text":              jd_text,
                })
            except Exception as e:
                results.append({
                    "job_ref_id": job_ref, "job_title": job_title,
                    "job_status": job_status, "function": function_,
                    "sub_function": sub_function, "business_unit": business_unit,
                    "skills": [], "error": str(e),
                    "candidates_retrieved": 0, "candidates": [], "stats": {},
                    "jd_text": jd_text,
                })

        # Update live stats after every role
        n_done   = len(results)
        n_conf   = sum(len(r.get("skills", [])) for r in results)
        n_cands  = sum(r.get("candidates_retrieved", 0) for r in results)
        n_err    = sum(1 for r in results if r.get("error"))
        acc_list = [r["stats"]["acceptance_rate"] for r in results
                    if r.get("stats", {}).get("acceptance_rate", 0) > 0]
        avg_acc  = f"{statistics.mean(acc_list):.0%}" if acc_list else "-"

        slot_done.metric("Roles done",          n_done)
        slot_conf.metric("Skills confirmed",    n_conf)
        slot_cands.metric("Candidates checked", n_cands)
        slot_accept.metric("Avg acceptance",    avg_acc)
        slot_err.metric("Errors",               n_err)
        refresh_live_charts(results, live_slots)

    progress_bar.progress(1.0, text="Done!")
    status_text.empty()
    st.session_state["a2_results"] = results

# ══════════════════════════════════════════════════════════════════════════════
# Results display
# ══════════════════════════════════════════════════════════════════════════════
if "a2_results" in st.session_state:
    results  = st.session_state["a2_results"]
    n_conf   = sum(len(r.get("skills", [])) for r in results)
    n_cands  = sum(r.get("candidates_retrieved", 0) for r in results)
    n_err    = sum(1 for r in results if r.get("error"))
    acc_list = [r["stats"]["acceptance_rate"] for r in results
                if r.get("stats", {}).get("acceptance_rate", 0) > 0]
    avg_acc  = f"{statistics.mean(acc_list):.0%}" if acc_list else "-"

    st.divider()
    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    sm1.metric("Roles Processed",    len(results))
    sm2.metric("Skills Confirmed",   n_conf)
    sm3.metric("Candidates Checked", n_cands)
    sm4.metric("Avg Acceptance",     avg_acc)
    sm5.metric("Errors / Skipped",   n_err)

    tab_roles, tab_compare, tab_table, tab_stats = st.tabs(
        ["By Role", "Cross-Role", "Full Table", "Statistics & Charts"]
    )

    # ── Tab 1: By Role ─────────────────────────────────────────────────────────
    with tab_roles:
        for role in results:
            n_skills     = len(role["skills"])
            n_cands_role = role.get("candidates_retrieved", 0)
            ar           = role.get("stats", {}).get("acceptance_rate", 0)
            accept_str   = f"{ar:.0%}" if n_cands_role else "-"
            core_n       = sum(1 for s in role["skills"] if s.get("importance") == "Core")
            imp_n        = sum(1 for s in role["skills"] if s.get("importance") == "Important")
            nth_n        = sum(1 for s in role["skills"] if s.get("importance") == "Nice to Have")

            label = (
                f"[{role['job_ref_id']}] {role['job_title']}  -  "
                f"{n_skills} confirmed  /  {n_cands_role} candidates  /  {accept_str} acceptance"
            )
            with st.expander(label, expanded=False):
                if role.get("error") and not role["skills"]:
                    st.warning(f"Error: {role['error']}")
                    continue

                m1, m2, m3, m4 = st.columns([1, 2, 2, 2])
                m1.markdown(f"**Status**\n\n`{role['job_status']}`")
                m2.markdown(f"**Function**\n\n{role.get('function','--')}")
                m3.markdown(f"**Sub-function**\n\n{role.get('sub_function','--')}")
                m4.markdown(f"**Skills**\n\n🔴 {core_n} Core  🟠 {imp_n} Imp  🟢 {nth_n} NtH")
                st.divider()

                h1, h2, h3, h4, h5 = st.columns([3, 2, 2, 4, 4])
                h1.markdown("**Skill**"); h2.markdown("**Proficiency**"); h3.markdown("**Importance**")
                h4.markdown("**ESCO URI**"); h5.markdown("**Evidence**")
                st.markdown("<hr style='margin:2px 0 6px 0;border-color:#333;'>", unsafe_allow_html=True)

                sorted_skills = sorted(
                    role["skills"],
                    key=lambda s: (
                        IMPORTANCE_ORDER.get(s.get("importance", ""), 9),
                        PROFICIENCY_ORDER.get(s.get("proficiency_level", ""), 9),
                    ),
                )
                current_imp = None
                for skill in sorted_skills:
                    imp  = skill.get("importance", "")
                    prof = skill.get("proficiency_level", "")
                    name = skill.get("skill_name", "")
                    uri  = skill.get("esco_uri", "")
                    evid = skill.get("evidence", "")

                    if imp != current_imp:
                        current_imp = imp
                        colour = {"Core":"#c0392b","Important":"#d35400","Nice to Have":"#27ae60"}.get(imp,"#888")
                        st.markdown(
                            f"<div style='margin:10px 0 4px 0;font-size:0.75rem;font-weight:600;"
                            f"color:{colour};text-transform:uppercase;letter-spacing:0.08em;'>"
                            f"{IMPORTANCE_COLOUR.get(imp,'')} {imp}</div>",
                            unsafe_allow_html=True,
                        )

                    c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 4, 4])
                    c1.markdown(f"**{name}**")
                    c2.markdown(f"{PROFICIENCY_COLOUR.get(prof,'⚪')} {prof}")
                    c3.markdown(f"{IMPORTANCE_COLOUR.get(imp,'⚪')} {imp}")
                    c4.caption(f"`{uri}`" if uri else "--")
                    c5.caption(f'*"{evid}"*' if evid else "--")

                # TF-IDF candidates panel
                candidates = role.get("candidates", [])
                if candidates:
                    confirmed_uris = {s.get("esco_uri", "") for s in role["skills"]}
                    with st.expander(f"TF-IDF candidates ({len(candidates)}) - similarity scores", expanded=False):
                        cand_rows = []
                        for c in candidates:
                            cand_rows.append({
                                "": "yes" if c["uri"] in confirmed_uris else "no",
                                "Skill":      c["title"],
                                "Similarity": round(c.get("similarity_score", 0), 3),
                                "URI":        c["uri"],
                            })
                        st.dataframe(pd.DataFrame(cand_rows), hide_index=True,
                                     height=min(300, len(cand_rows)*36+40))
                        sim_df = pd.DataFrame({
                            "Skill": [c["title"][:30] for c in candidates],
                            "Similarity": [c.get("similarity_score", 0) for c in candidates],
                        }).sort_values("Similarity", ascending=False).set_index("Skill")
                        st.markdown("**Cosine similarity per candidate**")
                        st.bar_chart(sim_df[["Similarity"]], height=200)

                if role.get("jd_text"):
                    with st.expander("Job description", expanded=False):
                        st.text(str(role.get("jd_text", ""))[:3000])

    # ── Tab 2: Cross-role ──────────────────────────────────────────────────────
    with tab_compare:
        st.subheader("Skills shared across multiple roles")
        st.caption("ESCO-anchored - consistent names because they come from the same EU catalogue.")

        skill_roles = {}
        for role in results:
            for s in role["skills"]:
                name = s.get("skill_name", "")
                if name:
                    skill_roles.setdefault(name, []).append(role["job_title"])

        shared = {k: v for k, v in skill_roles.items() if len(v) > 1}
        if shared:
            freq_df = pd.DataFrame([
                {"Skill": k, "# Roles": len(v), "Roles": ", ".join(v)}
                for k, v in sorted(shared.items(), key=lambda x: -len(x[1]))
            ])
            st.dataframe(freq_df, height=400)
            st.bar_chart(freq_df.set_index("Skill")[["# Roles"]].head(25))
        else:
            st.info("Process more roles to see cross-role skill overlap.")

    # ── Tab 3: Full table ──────────────────────────────────────────────────────
    with tab_table:
        df_out = flatten_results(results)
        if not df_out.empty:
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                imp_filter = st.multiselect("Importance",
                    ["Core","Important","Nice to Have"], default=["Core","Important","Nice to Have"])
            with fc2:
                prof_filter = st.multiselect("Proficiency",
                    ["Awareness","Working","Practitioner","Expert"],
                    default=["Awareness","Working","Practitioner","Expert"])
            with fc3:
                role_filter = st.multiselect("Role",
                    df_out["Job Title"].unique().tolist(),
                    default=df_out["Job Title"].unique().tolist())

            df_filt = df_out[
                df_out["Importance"].isin(imp_filter) &
                df_out["Proficiency Level"].isin(prof_filter) &
                df_out["Job Title"].isin(role_filter)
            ]
            st.caption(f"Showing {len(df_filt):,} of {len(df_out):,} skill rows")
            st.dataframe(df_filt, height=450)
            st.download_button(
                "Download filtered CSV",
                df_filt.to_csv(index=False).encode("utf-8"),
                "approach2_results.csv", "text/csv",
            )
        else:
            st.info("No skill data to display.")

    # ── Tab 4: Statistics & Charts ─────────────────────────────────────────────
    with tab_stats:
        df_out = flatten_results(results)

        st.markdown("### Statistical Summary")
        all_scores = [
            c.get("similarity_score", 0)
            for r in results for c in r.get("candidates", [])
            if c.get("similarity_score") is not None
        ]
        all_accept = [
            r["stats"]["acceptance_rate"]
            for r in results
            if r.get("stats", {}).get("acceptance_rate", 0) > 0
        ]

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total candidates",    n_cands)
        s2.metric("Total confirmed",     n_conf)
        s3.metric("Overall acceptance",  f"{n_conf/max(n_cands,1):.1%}")
        s4.metric("Mean cosine score",
                  f"{statistics.mean(all_scores):.3f}" if all_scores else "-")

        if all_scores:
            st.markdown(
                f"Cosine scores: "
                f"min `{min(all_scores):.3f}` | "
                f"median `{statistics.median(all_scores):.3f}` | "
                f"mean `{statistics.mean(all_scores):.3f}` | "
                f"max `{max(all_scores):.3f}` | "
                f"n={len(all_scores):,}"
            )

        st.divider()
        st.markdown("### Per-role statistics table")
        stats_rows = []
        for r in results:
            s = r.get("stats", {})
            stats_rows.append({
                "Job Title":       r["job_title"],
                "Job Ref":         r["job_ref_id"],
                "Candidates":      s.get("n_candidates", r.get("candidates_retrieved", 0)),
                "Confirmed":       s.get("n_confirmed",  len(r.get("skills", []))),
                "Acceptance":      f"{s.get('acceptance_rate',0):.0%}",
                "Cosine Min":      s.get("cosine_min",   0),
                "Cosine Mean":     s.get("cosine_mean",  0),
                "Cosine Max":      s.get("cosine_max",   0),
                "Cosine StdDev":   s.get("cosine_stdev", 0),
                "Error":           r.get("error", ""),
            })
        if stats_rows:
            st.dataframe(pd.DataFrame(stats_rows), height=300)

        st.divider()

        if not df_out.empty:
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                st.markdown("**Skills by Importance**")
                imp_c = (df_out["Importance"].value_counts()
                         .reindex(["Core","Important","Nice to Have"], fill_value=0)
                         .rename_axis("Importance").reset_index(name="Count"))
                st.bar_chart(imp_c.set_index("Importance"))
            with r1c2:
                st.markdown("**Skills by Proficiency Level**")
                prof_c = (df_out["Proficiency Level"].value_counts()
                          .reindex(["Awareness","Working","Practitioner","Expert"], fill_value=0)
                          .rename_axis("Level").reset_index(name="Count"))
                st.bar_chart(prof_c.set_index("Level"))

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("**Candidates vs Confirmed per role**")
            acc_rows = [
                {"Role": r["job_title"][:30],
                 "Candidates": r.get("candidates_retrieved", 0),
                 "Confirmed":  len(r.get("skills", []))}
                for r in results if r.get("candidates_retrieved", 0) > 0
            ]
            if acc_rows:
                st.bar_chart(pd.DataFrame(acc_rows).set_index("Role"))
        with r2c2:
            if not df_out.empty:
                st.markdown("**Top 20 most frequent skills**")
                top20 = df_out["Skill"].value_counts().head(20).rename_axis("Skill").reset_index(name="Count")
                st.bar_chart(top20.set_index("Skill"))

        st.markdown("**TF-IDF cosine similarity distribution (all candidates)**")
        if all_scores:
            bins   = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]
            labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30-50%", "50%+"]
            score_df = pd.DataFrame({"score": all_scores})
            score_df["bucket"] = pd.cut(score_df["score"], bins=bins, labels=labels, right=False)
            bucket_counts = (
                score_df["bucket"].value_counts()
                .reindex(labels, fill_value=0)
                .rename_axis("Similarity bucket").reset_index(name="Count")
            )
            st.bar_chart(bucket_counts.set_index("Similarity bucket"))
            st.caption(
                f"n={len(all_scores):,} candidates  |  "
                f"min={min(all_scores):.3f}  |  "
                f"mean={statistics.mean(all_scores):.3f}  |  "
                f"max={max(all_scores):.3f}"
            )
        else:
            st.caption("Run the matching process to see cosine score data.")
