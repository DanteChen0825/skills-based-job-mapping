import { useState, useMemo } from 'react'
import axios from 'axios'
import Combobox from './Combobox'
import hierarchy from './jobHierarchy.json'
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

const CONFIDENCE_CLASS = { High: 'conf-high', Medium: 'conf-medium', Low: 'conf-low' }

function EscoOccupations({ occupations }) {
  if (!occupations || occupations.length === 0) return null
  return (
    <div className="card esco-card">
      <div className="card-title">
        Closest ESCO Occupations
        <a
          className="esco-link-header"
          href="https://esco.ec.europa.eu/en/classification/occupation_main"
          target="_blank"
          rel="noopener noreferrer"
        >
          ↗ ESCO Classification
        </a>
      </div>
      <div className="esco-list">
        {occupations.map((o, i) => (
          <div key={i} className="esco-item">
            <div className="esco-item-header">
              <span className="esco-rank">{i + 1}</span>
              <span className="esco-title">{o.occupation_title}</span>
              <span className={`conf-badge ${CONFIDENCE_CLASS[o.match_confidence] ?? ''}`}>
                {o.match_confidence}
              </span>
            </div>
            <div className="esco-codes">
              <span className="esco-code-chip">ESCO {o.esco_code}</span>
              <span className="esco-code-chip isco-chip">ISCO-08 {o.isco_code}</span>
            </div>
            <p className="esco-rationale">{o.rationale}</p>
          </div>
        ))}
      </div>
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

  // Derived option lists — cascade on selection
  const buOptions  = useMemo(() => Object.keys(hierarchy).sort(), [])
  const fnOptions  = useMemo(() => {
    const fns = hierarchy[form.business_unit]
    return fns ? Object.keys(fns).sort() : []
  }, [form.business_unit])
  const sfOptions  = useMemo(() => {
    const fns = hierarchy[form.business_unit]
    return fns?.[form.function] ?? []
  }, [form.business_unit, form.function])

  const setField = (name, value) => {
    setForm((f) => {
      const next = { ...f, [name]: value }
      // Cascade resets
      if (name === 'business_unit') { next.function = ''; next.sub_function = '' }
      if (name === 'function')      { next.sub_function = '' }
      return next
    })
  }

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
            <Combobox
              label="Business Unit"
              options={buOptions}
              value={form.business_unit}
              onChange={(v) => setField('business_unit', v)}
              placeholder="Type or select…"
            />
            <Combobox
              label="Function"
              options={fnOptions}
              value={form.function}
              onChange={(v) => setField('function', v)}
              placeholder="Type or select…"
              disabled={!form.business_unit}
            />
            <Combobox
              label="Sub-function"
              options={sfOptions}
              value={form.sub_function}
              onChange={(v) => setField('sub_function', v)}
              placeholder="Type or select…"
              disabled={!form.function}
            />
          </div>
          <div className="field-full">
            <label>Job Description *</label>
            <textarea
              name="job_description"
              placeholder="Paste the full job advertisement text here…"
              value={form.job_description}
              onChange={(e) => setField('job_description', e.target.value)}
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

      {result && (
        <EscoOccupations occupations={result.esco_occupations} />
      )}
    </div>
  )
}

export default App
