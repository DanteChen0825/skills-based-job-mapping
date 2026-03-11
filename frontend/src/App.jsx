import { useState } from 'react'
import axios from 'axios'
import './App.css'

const PROF_LABELS = {
  1: 'Awareness',
  2: 'Foundation',
  3: 'Practitioner',
  4: 'Advanced',
  5: 'Expert / Lead',
}

function ProfBadge({ level }) {
  return (
    <span className={`prof prof-${level}`}>
      {level} – {PROF_LABELS[level] ?? level}
    </span>
  )
}

function SkillsTable({ skills }) {
  return (
    <div className="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Skill</th>
            <th>Category</th>
            <th>Proficiency</th>
            <th>Evidence</th>
          </tr>
        </thead>
        <tbody>
          {skills.map((s, i) => (
            <tr key={i}>
              <td style={{ color: '#a0aec0', width: '2rem' }}>{i + 1}</td>
              <td style={{ fontWeight: 600 }}>{s.skill_name}</td>
              <td><span className="cat-tag">{s.category}</span></td>
              <td><ProfBadge level={s.proficiency_required} /></td>
              <td><span className="evidence">"{s.evidence}"</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function App() {
  const [form, setForm] = useState({
    business_unit: '',
    function: '',
    sub_function: '',
    job_description: '',
  })
  const [loading, setLoading] = useState(false)
  const [result, setResult]   = useState(null)
  const [error, setError]     = useState(null)

  const handleChange = (e) =>
    setForm((f) => ({ ...f, [e.target.name]: e.target.value }))

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const { data } = await axios.post('/api/extract-skills', form)
      if (data.error) {
        setError(data.error)
      } else {
        setResult(data)
      }
    } catch (err) {
      setError(err.response?.data?.error ?? err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      {/* ── Header ── */}
      <div className="header">
        <div className="header-logo">EDF</div>
        <div>
          <h1>Skills-Based Job Mapping</h1>
          <p>Extract structured skills from EDF job descriptions using Claude Sonnet 4.6</p>
        </div>
      </div>

      {/* ── Input form ── */}
      <div className="card">
        <div className="card-title">Job Record</div>
        <form onSubmit={handleSubmit}>
          <div className="form-grid">
            <div className="field">
              <label>Business Unit</label>
              <input
                name="business_unit"
                placeholder="e.g. Customers (64013735)"
                value={form.business_unit}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Function</label>
              <input
                name="function"
                placeholder="e.g. Retail"
                value={form.function}
                onChange={handleChange}
              />
            </div>
            <div className="field">
              <label>Sub-function</label>
              <input
                name="sub_function"
                placeholder="e.g. Smart Metering Install"
                value={form.sub_function}
                onChange={handleChange}
              />
            </div>
          </div>
          <div className="field-full">
            <label>Job Description *</label>
            <textarea
              name="job_description"
              placeholder="Paste the full job advertisement text here…"
              value={form.job_description}
              onChange={handleChange}
              required
            />
          </div>
          <button className="btn-extract" type="submit" disabled={loading}>
            {loading ? <span className="spinner" /> : '⚡'}
            {loading ? 'Extracting skills…' : 'Extract Skills'}
          </button>
        </form>
      </div>

      {/* ── Error ── */}
      {error && (
        <div className="error-box">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* ── Results ── */}
      {result && (
        <div className="card">
          <div className="card-title">Extracted Skills</div>

          {/* Meta chips */}
          <div className="result-meta">
            <div className="meta-chip">
              <div className="label">Job Title</div>
              <div className="value">{result.job_title}</div>
            </div>
            <div className="meta-chip">
              <div className="label">Business Unit</div>
              <div className="value">{result.business_unit}</div>
            </div>
            <div className="meta-chip">
              <div className="label">Function</div>
              <div className="value">{result.function}</div>
            </div>
            <div className="meta-chip">
              <div className="label">Sub-function</div>
              <div className="value">{result.sub_function}</div>
            </div>
            <div className="meta-chip">
              <div className="label">Seniority</div>
              <div className="value">
                <span className="seniority-badge">{result.seniority_level}</span>
              </div>
            </div>
            <div className="meta-chip">
              <div className="label">Skills Found</div>
              <div className="value">{result.skills?.length ?? 0}</div>
            </div>
          </div>

          <SkillsTable skills={result.skills ?? []} />
        </div>
      )}
    </div>
  )
}

export default App
