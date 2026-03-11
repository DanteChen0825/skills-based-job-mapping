"""
Flask API backend for the Skills Extraction UI.
Run:  python app.py
Listens on http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from skills_extractor import extract_skills

app = Flask(__name__)
CORS(app)


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


if __name__ == "__main__":
    app.run(port=5000, debug=True)
