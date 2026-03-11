# Skills-Based Job Mapping — Skill Extraction Prompt

## Context

You are a workforce skills analyst. You will be given a job record from EDF with the following fields:

- **Business Unit** — the organisational unit (e.g. "Hinkley Point C", "Nuclear Operations")
- **Function** — the department or function within the business unit
- **Sub-function** — the team or sub-department within the function
- **Default Job Ad Job Description** — the full text of the job advertisement

Your task is to **analyse the job description and extract a structured set of skills**, each with a **required proficiency level**.

---

## Instructions

1. Read the job description carefully.
2. Identify all distinct skills mentioned — both **explicitly stated** and **reasonably implied** by the responsibilities and requirements.
3. Group skills into the following **skill categories**:
   - **Technical / Domain Skills** — job-specific, role-specific or industry-specific knowledge (e.g. "CDM Regulations", "SAP", "Nuclear Safety", "Turbine Assembly")
   - **Digital & IT Skills** — software, systems, tools (e.g. "MS Excel", "Power BI", "Active Directory")
   - **Analytical & Problem-Solving Skills** — data analysis, financial modelling, root cause analysis, risk assessment, etc.
   - **Communication & Stakeholder Skills** — report writing, presentations, stakeholder engagement, negotiation
   - **Leadership & Management Skills** — team leadership, line management, performance management, coaching
   - **Compliance & Regulatory Skills** — safety regulations, quality assurance, audits, licensing
   - **Project & Programme Management Skills** — planning, scheduling, delivery, budget management
   - **Interpersonal & Behavioural Skills** — collaboration, adaptability, attention to detail, self-management

4. For each skill, assign a **proficiency level** using the scale below:
   - **1 – Awareness**: Basic familiarity; understands concepts but limited practical application
   - **2 – Foundation**: Some practical exposure; can perform under guidance
   - **3 – Practitioner**: Independent competence; applies skill regularly in role
   - **4 – Advanced**: Deep expertise; leads others, designs approaches, handles complexity
   - **5 – Expert / Lead**: Recognised authority; sets standards, shapes strategy, mentors others

5. Base proficiency level on:
   - Seniority signals (e.g. "Group Head", "Lead", "Advisor", "Instructor")
   - Language like "significant experience", "proven track record", "expert knowledge", "minimum HNC", "qualified accountant"
   - Responsibility scope (leading teams vs. supporting tasks)

6. After extracting skills, identify the **2–3 closest ESCO occupations** from the [ESCO classification](https://esco.ec.europa.eu/en/classification/occupation_main) that best match the inferred job title, seniority level, and extracted skill set. For each ESCO occupation provide:
   - `occupation_title` — the official ESCO occupation name
   - `esco_code` — the ESCO occupation code (decimal, e.g. `2152.1`)
   - `isco_code` — the underlying 4-digit ISCO-08 code
   - `match_confidence` — `High`, `Medium`, or `Low`
   - `rationale` — one sentence explaining why this occupation matches

---

## Output Format

Return the results as a structured JSON object. Use the exact schema below:

```json
{
  "business_unit": "<Business Unit>",
  "function": "<Function>",
  "sub_function": "<Sub-function>",
  "job_title": "<Inferred job title from description>",
  "seniority_level": "<Entry | Junior | Mid | Senior | Lead | Head>",
  "skills": [
    {
      "skill_name": "<skill name>",
      "category": "<one of the 8 categories above>",
      "proficiency_required": <1–5>,
      "evidence": "<short quote or phrase from the JD that supports this skill and level>"
    }
  ],
  "esco_occupations": [
    {
      "occupation_title": "<ESCO occupation title>",
      "esco_code": "<ESCO occupation code, e.g. 2152.1>",
      "isco_code": "<4-digit ISCO-08 code>",
      "match_confidence": "<High | Medium | Low>",
      "rationale": "<one sentence explaining why this occupation matches>"
    }
  ]
}
```

### Rules

- Return **only** the JSON — no preamble, no commentary.
- Each skill must appear only **once** in the output (deduplicate).
- Minimum **8 skills** per record; capture all meaningful skills present.
- `evidence` should be a direct quote or close paraphrase from the job description — keep it under 20 words.
- If the job description is **empty or too short** to extract skills from, return:
  ```json
  {
    "error": "Insufficient job description content",
    "business_unit": "...",
    "function": "...",
    "sub_function": "..."
  }
  ```

---

## Proficiency Level Reference Card

| Level | Label         | Typical JD language                                                                         |
| ----- | ------------- | ------------------------------------------------------------------------------------------- |
| 1     | Awareness     | "desirable", "exposure to", "an understanding of", "familiar with"                          |
| 2     | Foundation    | "basic", "some experience", "working towards", "studying for"                               |
| 3     | Practitioner  | "experience in", "competent", "able to", "knowledge of"                                     |
| 4     | Advanced      | "significant experience", "proven track record", "expert knowledge", "strong background in" |
| 5     | Expert / Lead | "lead", "set standards", "head of", "strategic", "accountable for", "authority in"          |

---

## Examples

### Input

```
Business Unit: Customers (64013735)
Function: EDF Business & Wholesale Services
Sub-function: Mid Market (65010009)
Job Description: Customer Service Advisor ... A passion for delivering great customer service – no experience needed, just a willingness to learn! Strong communication skills ... A problem-solver who thrives in a fast-moving environment ...
```

### Output

```json
{
  "business_unit": "Customers (64013735)",
  "function": "EDF Business & Wholesale Services",
  "sub_function": "Mid Market (65010009)",
  "job_title": "Customer Service Advisor",
  "seniority_level": "Entry",
  "skills": [
    {
      "skill_name": "Customer Service",
      "category": "Interpersonal & Behavioural Skills",
      "proficiency_required": 2,
      "evidence": "passion for delivering great customer service – no experience needed"
    },
    {
      "skill_name": "Written and Verbal Communication",
      "category": "Communication & Stakeholder Skills",
      "proficiency_required": 2,
      "evidence": "Strong communication skills – speaking, listening, and writing with confidence"
    },
    {
      "skill_name": "Problem Solving",
      "category": "Analytical & Problem-Solving Skills",
      "proficiency_required": 2,
      "evidence": "problem-solver who thrives in a fast-moving environment"
    },
    {
      "skill_name": "Multi-platform IT Usage",
      "category": "Digital & IT Skills",
      "proficiency_required": 2,
      "evidence": "Comfortable using IT systems and navigating multiple platforms"
    },
    {
      "skill_name": "Teamwork and Collaboration",
      "category": "Interpersonal & Behavioural Skills",
      "proficiency_required": 2,
      "evidence": "team player who's eager to support colleagues and share ideas"
    },
    {
      "skill_name": "Multi-channel Customer Communication",
      "category": "Technical / Domain Skills",
      "proficiency_required": 2,
      "evidence": "support via phone, email, LiveChat, and online portals"
    },
    {
      "skill_name": "Query and Complaint Resolution",
      "category": "Technical / Domain Skills",
      "proficiency_required": 2,
      "evidence": "Take ownership of customer queries, resolving issues efficiently and professionally"
    },
    {
      "skill_name": "Continuous Improvement Mindset",
      "category": "Analytical & Problem-Solving Skills",
      "proficiency_required": 1,
      "evidence": "Spot opportunities to improve customer experiences and suggest ways to make things better"
    }
  ],
  "esco_occupations": [
    {
      "occupation_title": "Contact centre information clerk",
      "esco_code": "4222.1",
      "isco_code": "4222",
      "match_confidence": "High",
      "rationale": "Role involves handling multi-channel customer enquiries and resolving queries via phone, email, and online portals."
    },
    {
      "occupation_title": "Customer service representative",
      "esco_code": "5244.1",
      "isco_code": "5244",
      "match_confidence": "High",
      "rationale": "Entry-level customer-facing role focused on service delivery, complaint handling, and maintaining customer satisfaction."
    },
    {
      "occupation_title": "Client information worker",
      "esco_code": "4224.1",
      "isco_code": "4224",
      "match_confidence": "Medium",
      "rationale": "Shares responsibilities around providing information and support to customers across multiple channels."
    }
  ]
}
```

---

## Batch Processing Instructions

When processing a spreadsheet with **multiple rows**, process each row independently and return a **JSON array** of result objects, one per row, in the same order as the input.

```json
[
  { /* row 1 result */ },
  { /* row 2 result */ },
  ...
]
```

Do not skip rows, even if the job description is empty — return the error object for those rows.
