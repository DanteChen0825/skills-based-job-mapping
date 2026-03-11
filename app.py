"""
Flask API backend for the Skills Extraction UI.
Run:  python app.py
Listens on http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from skills_extractor import extract_skills
import requests as _req
import urllib3

# Corporate SSL-inspection proxies inject a self-signed cert — disable verification
# for outbound ESCO API calls only and silence the resulting urllib3 warning.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)
CORS(app)

_ESCO_API     = "https://ec.europa.eu/esco/api"
_ESCO_HEADERS = {"Accept": "application/json"}
_ESCO_VERIFY  = False   # set to True or a CA-bundle path if cert is available


@app.route("/api/extract-skills", methods=["POST"])
def api_extract_skills():
    data = request.get_json(force=True)

    business_unit   = data.get("business_unit", "").strip()
    function        = data.get("function", "").strip()
    sub_function    = data.get("sub_function", "").strip()
    job_description = data.get("job_description", "").strip()

    if not job_description:
        return jsonify({"error": "job_description is required"}), 400

    try:
        result = extract_skills(business_unit, function, sub_function, job_description)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/esco-occupation-skills", methods=["GET"])
def api_esco_occupation_skills():
    """
    Proxy to ESCO REST API.
    1. Searches for an occupation by title.
    2. Fetches the occupation's essential and optional skills.
    Query param: title (required)
    """
    title = request.args.get("title", "").strip()
    if not title:
        return jsonify({"error": "title parameter required"}), 400

    try:
        # Step 1 — find best-matching occupation
        search_r = _req.get(
            f"{_ESCO_API}/search",
            params={"language": "en", "type": "occupation", "text": title, "limit": 3},
            headers=_ESCO_HEADERS,
            timeout=10,
            verify=_ESCO_VERIFY,
        )
        search_r.raise_for_status()
        results = search_r.json().get("_embedded", {}).get("results", [])
        if not results:
            return jsonify({"error": f"No ESCO occupation found for: {title}"}), 404

        top = results[0]
        occ_uri = top["uri"]

        # Step 2 — fetch full occupation resource (includes _links for skills)
        occ_r = _req.get(
            f"{_ESCO_API}/resource/occupation",
            params={"language": "en", "uri": occ_uri},
            headers=_ESCO_HEADERS,
            timeout=10,
            verify=_ESCO_VERIFY,
        )
        occ_r.raise_for_status()
        occ_data = occ_r.json()

        links = occ_data.get("_links", {})

        def _skill(s):
            return {
                "title":      s.get("title", ""),
                "uri":        s.get("href", ""),
                "skill_type": s.get("skillType", ""),
            }

        return jsonify({
            "occupation_title": occ_data.get("title", top.get("title", title)),
            "occupation_uri":   occ_uri,
            "esco_page_url":    f"https://esco.ec.europa.eu/en/classification/occupation?uri={occ_uri}",
            "essential_skills": [_skill(s) for s in links.get("hasEssentialSkill", [])],
            "optional_skills":  [_skill(s) for s in links.get("hasOptionalSkill",  [])],
        })

    except _req.RequestException as e:
        return jsonify({"error": f"ESCO API error: {str(e)}"}), 502


if __name__ == "__main__":
    app.run(port=5000, debug=True)
