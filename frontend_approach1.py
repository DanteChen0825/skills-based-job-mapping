"""
Streamlit Frontend – Approach 1: Direct LLM Skill Extraction  (v2)
===================================================================
Improvements over v1:
  • Role picker: search by title/ref/function + pick specific roles (not just "first N")
  • JD preview inside each result expander
  • Confidence score badge per skill (colour-coded)
  • Cross-role skill frequency view — which skills appear across multiple roles
  • Per-role category breakdown pie chart
  • Results table filterable by Category, Importance, Proficiency
  • Function / Sub-function columns in download CSV
  • Fixed use_column_width deprecation (removed deprecated kwarg)

Run:
    python -m streamlit run frontend_approach1.py
"""

import json
import re
import io
import time
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
You are an expert HR analyst specialising in skills-based job mapping for an energy company.
Your task is to extract structured, deduplicated skill data from job descriptions.
Always respond with valid JSON only — no prose, no markdown fences.
""".strip()

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

# ── Visual helpers ─────────────────────────────────────────────────────────────
IMPORTANCE_COLOUR = {"Core": "🔴", "Important": "🟠", "Nice to Have": "🟢"}
PROFICIENCY_COLOUR = {"Awareness": "⚪", "Working": "🔵", "Practitioner": "🟣", "Expert": "🟡"}
CATEGORY_ICON = {
    "Technical": "🔧", "Soft": "💬", "Domain": "📚",
    "Tool/System": "🖥️", "Compliance/Regulatory": "⚖️",
}
IMPORTANCE_ORDER = {"Core": 0, "Important": 1, "Nice to Have": 2}
PROFICIENCY_ORDER = {"Awareness": 0, "Working": 1, "Practitioner": 2, "Expert": 3}

# Keep old names as aliases so nothing breaks
IMPORTANCE_COLOURS = IMPORTANCE_COLOUR
PROFICIENCY_COLOURS = PROFICIENCY_COLOUR
CATEGORY_BADGE = CATEGORY_ICON


def confidence_badge(val) -> str:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return ""
    if f >= 0.85:
        return f"🟢 {f:.0%}"
    elif f >= 0.65:
        return f"🟡 {f:.0%}"
    else:
        return f"🔴 {f:.0%}"


def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    return json.loads(text)


@st.cache_data(show_spinner=False)
def load_excel() -> pd.DataFrame:
    return pd.read_excel(EXCEL_FILE)


def run_extraction(job_ref, job_title: str, function_: str, jd_text: str) -> list:
    prompt = EXTRACTION_PROMPT.format(
        job_title=job_title,
        function=function_,
        job_description=jd_text[:8000],
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
                "Function":          role.get("function", ""),
                "Sub-function":      role.get("sub_function", ""),
                "Skill":             skill.get("skill_name", ""),
                "Category":          skill.get("category", ""),
                "Proficiency Level": skill.get("proficiency_level", ""),
                "Importance":        skill.get("importance", ""),
                "Confidence":        skill.get("confidence", ""),
                "Evidence":          skill.get("evidence", ""),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Page layout
# ══════════════════════════════════════════════════════════════════════════════
st.title("🧠 Approach 1 — Direct LLM Skill Extraction")
st.caption(
    "Sends job descriptions to **Claude Sonnet** via Amazon Bedrock and extracts "
    "skills directly — no external catalogue needed."
)

with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
    1. Load job descriptions from the Excel file  
    2. Pick specific roles to process (search by title, ref ID, or function)  
    3. Claude identifies skills, assigns **proficiency level**, **importance**, and a
       **confidence score**, and quotes evidence from the JD  
    4. Results are displayed per-role and as a cross-role frequency table  
    5. Download full results as CSV

    **Strengths:** Fast, no external catalogue, easy to iterate  
    **Watch-outs:** Without the canonical reference list, naming can drift — the
    improved prompt anchors Claude to ~80 standard skill names
    """)

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading Excel file…"):
    df_raw = load_excel()

st.success(f"Loaded **{len(df_raw):,}** roles from the dataset")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    status_options = sorted(df_raw["Job Status"].dropna().unique().tolist())
    selected_statuses = st.multiselect(
        "Filter by Job Status",
        options=status_options,
        default=status_options,
    )

    function_options = ["All"] + sorted(df_raw["Function"].dropna().unique().tolist())
    selected_function = st.selectbox("Filter by Function", function_options)

    st.divider()
    st.markdown("**Select roles to process**")
    st.caption("Search by title, ref ID, or function then tick the ones you want.")

df_filtered = df_raw[df_raw["Job Status"].isin(selected_statuses)].copy()
if selected_function != "All":
    df_filtered = df_filtered[df_filtered["Function"] == selected_function]

# ── Role search + multi-select ─────────────────────────────────────────────────
search_query = st.text_input(
    "🔍 Search roles",
    placeholder="e.g. Finance, Engineer, 31961 …",
)

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
    col_a, col_b = st.columns(2)
    if col_a.button("Select first 5"):
        st.session_state["selected_role_labels"] = list(role_labels.keys())[:5]
    if col_b.button("Clear all"):
        st.session_state["selected_role_labels"] = []

    selected_labels = st.multiselect(
        "Roles to process",
        options=list(role_labels.keys()),
        default=st.session_state.get("selected_role_labels", []),
        key="selected_role_labels",
        label_visibility="collapsed",
    )

    st.divider()
    run_btn = st.button("▶️ Run Extraction", type="primary", use_container_width=True)

selected_indices = [role_labels[lbl] for lbl in selected_labels if lbl in role_labels]
df_selected = df_filtered.loc[selected_indices] if selected_indices else pd.DataFrame()

# ── Role preview table ─────────────────────────────────────────────────────────
if not df_selected.empty:
    st.subheader(f"📋 {len(df_selected)} role(s) selected for processing")
    st.dataframe(
        df_selected[["Job Ref ID", "Job Title", "Job Status", "Business unit", "Function", "Sub-function"]],
        height=200,
    )
else:
    st.info("👈 Use the sidebar to search and select roles, then click **▶️ Run Extraction**.")
    st.subheader("📋 Available roles (first 100 shown)")
    st.dataframe(
        df_display[["Job Ref ID", "Job Title", "Job Status", "Business unit", "Function"]],
        height=300,
    )

# ── Run extraction ─────────────────────────────────────────────────────────────
if run_btn:
    if df_selected.empty:
        st.warning("⚠️ No roles selected. Use the sidebar to pick at least one role.")
        st.stop()

    results = []
    total = len(df_selected)
    progress_bar = st.progress(0, text="Starting…")
    status_text  = st.empty()

    for i, (_, row) in enumerate(df_selected.iterrows()):
        job_ref      = row["Job Ref ID"]
        job_title    = row["Job Title"]
        job_status   = row["Job Status"]
        function_    = str(row.get("Function", "") or "")
        sub_function = str(row.get("Sub-function", "") or "")
        business_unit= str(row.get("Business unit", "") or "")
        jd_text      = str(row.get("Default Job Ad Job Description", "") or "").strip()

        progress_bar.progress(i / total, text=f"Processing {i+1}/{total}: {job_title}")
        status_text.info(f"🔄 **[{job_ref}]** {job_title}")

        if not jd_text or jd_text.lower() == "nan":
            results.append({
                "job_ref_id": job_ref, "job_title": job_title,
                "job_status": job_status, "function": function_,
                "sub_function": sub_function, "business_unit": business_unit,
                "skills": [], "error": "No JD text", "jd_text": "",
            })
            continue

        try:
            skills = run_extraction(
                job_ref, job_title,
                f"{function_} / {sub_function}".strip(" /"),
                jd_text,
            )
            results.append({
                "job_ref_id": job_ref, "job_title": job_title,
                "job_status": job_status, "function": function_,
                "sub_function": sub_function, "business_unit": business_unit,
                "skills": skills, "jd_text": jd_text,
            })
        except Exception as e:
            results.append({
                "job_ref_id": job_ref, "job_title": job_title,
                "job_status": job_status, "function": function_,
                "sub_function": sub_function, "business_unit": business_unit,
                "skills": [], "error": str(e), "jd_text": jd_text,
            })

    progress_bar.progress(1.0, text="✅ Done!")
    status_text.empty()
    st.session_state["a1_results"] = results

# ══════════════════════════════════════════════════════════════════════════════
# Results display
# ══════════════════════════════════════════════════════════════════════════════
if "a1_results" in st.session_state:
    results      = st.session_state["a1_results"]
    total_skills = sum(len(r["skills"]) for r in results)
    errors       = sum(1 for r in results if r.get("error"))

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Roles Processed", len(results))
    m2.metric("Total Skills Extracted", total_skills)
    m3.metric("Avg Skills / Role", f"{total_skills / max(len(results),1):.1f}")
    m4.metric("Errors / Skipped", errors)

    tab_roles, tab_compare, tab_table, tab_charts = st.tabs(
        ["📂 By Role", "🔁 Cross-Role Comparison", "📊 Full Table", "📈 Charts"]
    )

    # ── Tab 1: Per-role expanders ──────────────────────────────────────────────
    with tab_roles:
        for role in results:
            skill_count = len(role["skills"])
            core_count  = sum(1 for s in role["skills"] if s.get("importance") == "Core")
            imp_count   = sum(1 for s in role["skills"] if s.get("importance") == "Important")
            nth_count   = sum(1 for s in role["skills"] if s.get("importance") == "Nice to Have")

            # Expander header: ref + title + pill summary
            label = (
                f"[{role['job_ref_id']}] {role['job_title']}  "
                f"— {skill_count} skills"
            )
            with st.expander(label, expanded=False):
                if role.get("error") and not role["skills"]:
                    st.warning(f"⚠️ {role['error']}")
                    continue

                # ── Role meta row ──────────────────────────────────────────
                m1, m2, m3, m4 = st.columns([1, 2, 2, 1])
                m1.markdown(f"**Status**  \n`{role['job_status']}`")
                m2.markdown(f"**Function**  \n{role.get('function','—')}")
                m3.markdown(f"**Sub-function**  \n{role.get('sub_function','—')}")
                m4.markdown(
                    f"**Skills**  \n"
                    f"🔴 {core_count} Core &nbsp; "
                    f"🟠 {imp_count} Imp &nbsp; "
                    f"🟢 {nth_count} NtH"
                )

                st.divider()

                # ── Column header row ──────────────────────────────────────
                h1, h2, h3, h4, h5 = st.columns([3, 2, 2, 1.5, 4])
                h1.markdown("**Skill**")
                h2.markdown("**Category**")
                h3.markdown("**Proficiency**")
                h4.markdown("**Importance**")
                h5.markdown("**Evidence**")

                st.markdown(
                    "<hr style='margin:2px 0 6px 0; border-color:#333;'>",
                    unsafe_allow_html=True,
                )

                # ── Skills grouped by importance ───────────────────────────
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
                    cat  = skill.get("category", "")
                    conf = skill.get("confidence", "")
                    name = skill.get("skill_name", "")
                    evid = skill.get("evidence", "")

                    # Section divider when importance group changes
                    if imp != current_imp:
                        current_imp = imp
                        colour = {"Core": "#c0392b", "Important": "#d35400", "Nice to Have": "#27ae60"}.get(imp, "#888")
                        st.markdown(
                            f"<div style='margin:10px 0 4px 0; font-size:0.75rem; "
                            f"font-weight:600; color:{colour}; text-transform:uppercase; "
                            f"letter-spacing:0.08em;'>"
                            f"{IMPORTANCE_COLOUR.get(imp,'')} {imp}</div>",
                            unsafe_allow_html=True,
                        )

                    c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 1.5, 4])
                    c1.markdown(f"**{name}**")
                    c2.markdown(f"{CATEGORY_ICON.get(cat,'🔹')} {cat}")
                    c3.markdown(f"{PROFICIENCY_COLOUR.get(prof,'⚪')} {prof}")
                    c4.markdown(confidence_badge(conf))
                    c5.caption(f'*"{evid}"*' if evid else "—")

                # ── JD preview ─────────────────────────────────────────────
                if role.get("jd_text"):
                    st.markdown("")
                    with st.expander("📄 View job description", expanded=False):
                        st.text(role["jd_text"][:3000] + ("…" if len(role["jd_text"]) > 3000 else ""))

    # ── Tab 2: Cross-role skill frequency ─────────────────────────────────────
    with tab_compare:
        st.subheader("� Skills appearing across multiple roles")
        st.caption(
            "Skills that appear in more than one role — useful for identifying a shared "
            "core vocabulary and spotting where the prompt is naming things consistently."
        )

        from collections import Counter
        skill_roles: dict[str, list[str]] = {}
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
            st.markdown("**Frequency bar chart**")
            st.bar_chart(freq_df.set_index("Skill")[["# Roles"]].head(25))
        else:
            st.info("No skills shared across roles yet — process more roles to see overlap.")

    # ── Tab 3: Filterable full table ───────────────────────────────────────────
    with tab_table:
        df_out = flatten_results(results)

        if not df_out.empty:
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                cat_filter = st.multiselect(
                    "Category", df_out["Category"].dropna().unique().tolist(),
                    default=df_out["Category"].dropna().unique().tolist(),
                )
            with fc2:
                imp_filter = st.multiselect(
                    "Importance", ["Core", "Important", "Nice to Have"],
                    default=["Core", "Important", "Nice to Have"],
                )
            with fc3:
                prof_filter = st.multiselect(
                    "Proficiency", ["Awareness", "Working", "Practitioner", "Expert"],
                    default=["Awareness", "Working", "Practitioner", "Expert"],
                )

            df_filtered_out = df_out[
                df_out["Category"].isin(cat_filter) &
                df_out["Importance"].isin(imp_filter) &
                df_out["Proficiency Level"].isin(prof_filter)
            ]
            st.caption(f"Showing {len(df_filtered_out):,} of {len(df_out):,} skill rows")
            st.dataframe(df_filtered_out, height=450)

            csv_bytes = df_filtered_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download filtered CSV",
                data=csv_bytes,
                file_name="approach1_results.csv",
                mime="text/csv",
            )
        else:
            st.info("No skill data to display.")

    # ── Tab 4: Charts ──────────────────────────────────────────────────────────
    with tab_charts:
        df_out = flatten_results(results)
        if df_out.empty:
            st.info("No skill data yet.")
        else:
            r1c1, r1c2 = st.columns(2)

            with r1c1:
                st.markdown("**Skills by Category**")
                cat_counts = df_out["Category"].value_counts().rename_axis("Category").reset_index(name="Count")
                st.bar_chart(cat_counts.set_index("Category"))

            with r1c2:
                st.markdown("**Skills by Importance**")
                imp_counts = (
                    df_out["Importance"]
                    .value_counts()
                    .reindex(["Core", "Important", "Nice to Have"], fill_value=0)
                    .rename_axis("Importance")
                    .reset_index(name="Count")
                )
                st.bar_chart(imp_counts.set_index("Importance"))

            r2c1, r2c2 = st.columns(2)

            with r2c1:
                st.markdown("**Skills by Proficiency Level**")
                prof_counts = (
                    df_out["Proficiency Level"]
                    .value_counts()
                    .reindex(["Awareness", "Working", "Practitioner", "Expert"], fill_value=0)
                    .rename_axis("Level")
                    .reset_index(name="Count")
                )
                st.bar_chart(prof_counts.set_index("Level"))

            with r2c2:
                st.markdown("**Core skills per role**")
                core_per_role = (
                    df_out[df_out["Importance"] == "Core"]
                    .groupby("Job Title")
                    .size()
                    .sort_values(ascending=False)
                    .rename_axis("Job Title")
                    .reset_index(name="Core Skills")
                )
                st.bar_chart(core_per_role.set_index("Job Title"))

            st.markdown("**Confidence score distribution**")
            try:
                df_out["Confidence_float"] = pd.to_numeric(df_out["Confidence"], errors="coerce")
                conf_data = df_out["Confidence_float"].dropna()
                if not conf_data.empty:
                    bins   = [0, 0.5, 0.65, 0.80, 0.90, 1.01]
                    labels = ["<50%", "50-65%", "65-80%", "80-90%", "90-100%"]
                    df_out["conf_bucket"] = pd.cut(df_out["Confidence_float"], bins=bins, labels=labels, right=False)
                    bucket_counts = (
                        df_out["conf_bucket"]
                        .value_counts()
                        .reindex(labels, fill_value=0)
                        .rename_axis("Confidence")
                        .reset_index(name="Count")
                    )
                    st.bar_chart(bucket_counts.set_index("Confidence"))
            except Exception:
                st.caption("Confidence data not available for this run.")
