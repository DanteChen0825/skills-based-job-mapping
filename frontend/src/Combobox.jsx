import { useState, useRef, useEffect } from 'react'
import './Combobox.css'

/**
 * A type-in + dropdown combobox.
 * Props:
 *   label       – field label (string)
 *   options     – string[]
 *   value       – controlled value
 *   onChange    – (val: string) => void
 *   placeholder – placeholder text
 *   disabled    – bool
 */
export default function Combobox({ label, options = [], value, onChange, placeholder, disabled }) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const containerRef = useRef(null)
  const inputRef = useRef(null)

  // Sync display text when value is set externally (e.g. cascade reset)
  useEffect(() => {
    if (!open) setQuery(value || '')
  }, [value, open])

  // Close on outside click
  useEffect(() => {
    const handler = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setOpen(false)
        // If nothing selected, clear query
        if (!options.includes(query)) {
          setQuery(value || '')
        }
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [query, value, options])

  const filtered = query
    ? options.filter(o => o.toLowerCase().includes(query.toLowerCase()))
    : options

  const handleInputChange = (e) => {
    setQuery(e.target.value)
    onChange(e.target.value)
    setOpen(true)
  }

  const handleSelect = (option) => {
    setQuery(option)
    onChange(option)
    setOpen(false)
    inputRef.current?.blur()
  }

  const handleFocus = () => {
    if (!disabled) {
      setQuery('')
      setOpen(true)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Escape') { setOpen(false); setQuery(value || '') }
    if (e.key === 'Enter' && filtered.length === 1) handleSelect(filtered[0])
  }

  return (
    <div className={`cb-wrap${disabled ? ' cb-disabled' : ''}`} ref={containerRef}>
      {label && <label className="cb-label">{label}</label>}
      <div className={`cb-input-wrap${open ? ' cb-open' : ''}`}>
        <input
          ref={inputRef}
          className="cb-input"
          type="text"
          value={query}
          placeholder={disabled ? '— select above first —' : placeholder}
          disabled={disabled}
          onChange={handleInputChange}
          onFocus={handleFocus}
          onKeyDown={handleKeyDown}
          autoComplete="off"
        />
        <span className={`cb-chevron${open ? ' cb-chevron-up' : ''}`} onClick={() => !disabled && setOpen(o => !o)}>▾</span>
      </div>
      {open && filtered.length > 0 && (
        <ul className="cb-list">
          {filtered.map(opt => (
            <li
              key={opt}
              className={`cb-item${opt === value ? ' cb-item-selected' : ''}`}
              onMouseDown={() => handleSelect(opt)}
            >
              {opt}
            </li>
          ))}
        </ul>
      )}
      {open && filtered.length === 0 && (
        <div className="cb-empty">No matches</div>
      )}
    </div>
  )
}
