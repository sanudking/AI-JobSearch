import { useRef, useState, type FormEvent } from 'react'
import { ArrowUpRight, BriefcaseBusiness, FileText, FileUp, LoaderCircle, MapPin, Search, Sparkles, X } from 'lucide-react'

type Job = {
  id: string; title: string; company: string; location: string; source: string; url: string
  description: string; posted_at?: string | null; employment_type?: string | null; salary?: string | null
}
type Result = {
  job: Job; score: number; matched_skills: string[]; missing_skills: string[]
  reason: string; evidence: string; semantic_similarity: number
}
type Response = {
  resume_summary: string; skills: string[]; results: Result[]; jobs_found: number
  sources_used: string[]; source_warnings: string[]; retrieval_method: string; model: string
}

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

export default function App() {
  const fileRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [query, setQuery] = useState('software engineer internship')
  const [location, setLocation] = useState('India')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [data, setData] = useState<Response | null>(null)

  async function runMatch(e: FormEvent) {
    e.preventDefault()
    if (!file) { setError('Choose your resume PDF first.'); return }
    setLoading(true); setError(''); setData(null)
    const body = new FormData()
    body.append('resume', file)
    body.append('query', query)
    body.append('location', location)
    try {
      const response = await fetch(`${API}/api/match`, { method: 'POST', body })
      const json = await response.json()
      if (!response.ok) throw new Error(json.detail || `Request failed (${response.status})`)
      setData(json)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally { setLoading(false) }
  }

  return <div className="page-shell">
    <header className="topbar">
      <a className="brand" href="#top" aria-label="JobMatch home"><span className="brand-mark"><BriefcaseBusiness size={17}/></span><span>jobmatch</span></a>
      <span className="source-pill"><span className="status-dot"/> Adzuna jobs</span>
    </header>

    <main className="app" id="top">
      <form className="search-panel" onSubmit={runMatch}>
        <div className="panel-heading"><div className="heading-icon"><Sparkles size={17}/></div><span>Find jobs for your resume</span></div>
        <input ref={fileRef} type="file" accept="application/pdf,.pdf" hidden onChange={e => { setFile(e.target.files?.[0] || null); setError('') }} />
        <div className={`upload-zone ${file ? 'has-file' : ''}`}>
          <div className="upload-symbol"><FileUp size={20}/></div>
          <div className="upload-copy">
            <strong>{file ? file.name : 'Upload your resume'}</strong>
            <span>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB · PDF` : 'PDF document · up to 10 MB'}</span>
          </div>
          {file ? <button className="remove-file" type="button" aria-label="Remove file" onClick={() => { setFile(null); if (fileRef.current) fileRef.current.value = '' }}><X size={16}/></button> : <button className="browse-button" type="button" onClick={() => fileRef.current?.click()}>Browse files</button>}
        </div>

        <div className="fields">
          <label><span><BriefcaseBusiness size={13}/> Job role</span><input value={query} onChange={e => setQuery(e.target.value)} placeholder="e.g. software engineering intern" /></label>
          <label><span><MapPin size={13}/> Location</span><input value={location} onChange={e => setLocation(e.target.value)} placeholder="India, Bengaluru, remote..." /></label>
        </div>
        <button className="submit" type="submit" disabled={loading || !file || !query.trim()}>
          {loading ? <><LoaderCircle className="spin" size={16}/> Finding matches</> : <><Search size={16}/> Find matching jobs</>}
        </button>
        <div className="privacy-note"><FileText size={13}/> Your resume is processed for matching and isn't saved.</div>
      </form>

      {error && <div className="error"><span>{error}</span></div>}

      {data && <section className="results">
        <div className="results-header">
          <div><div className="eyebrow">YOUR RESULTS</div><h1>Matching jobs</h1></div>
          <div className="result-count"><strong>{data.results.length}</strong><span>matches</span></div>
        </div>
        <div className="summary-box">{data.resume_summary}</div>
        <div className="meta-line">{data.jobs_found} jobs found <span>·</span> {data.retrieval_method} <span>·</span> {data.model}</div>
        {data.skills.length > 0 && <div className="skills"><span className="skills-label">Your skills</span>{data.skills.map(s => <i key={s}>{s}</i>)}</div>}
        {data.source_warnings.map((w, i) => <div className="warning" key={i}>{w}</div>)}
        <div className="job-list">{data.results.map((item, i) => <article className="job" key={item.job.id}>
          <div className="job-top">
            <div className="job-title-wrap"><div className="rank">{String(i + 1).padStart(2, '0')}</div><div><h2>{item.job.title}</h2><div className="company">{item.job.company} <span>·</span> {item.job.location}</div></div></div>
            <div className="score"><strong>{item.score}%</strong><span>match</span></div>
          </div>
          <div className="job-meta">{item.job.source}{item.job.employment_type ? ` · ${item.job.employment_type}` : ''}{item.job.salary ? ` · ${item.job.salary}` : ''}{item.job.posted_at ? ` · ${new Date(item.job.posted_at).toLocaleDateString()}` : ''}</div>
          {item.reason && <p className="reason">{item.reason}</p>}
          {item.matched_skills.length > 0 && <div className="skill-line"><b>Matched</b>{item.matched_skills.map(s => <i className="good" key={s}>{s}</i>)}</div>}
          {item.missing_skills.length > 0 && <div className="skill-line"><b>To develop</b>{item.missing_skills.map(s => <i className="gap" key={s}>{s}</i>)}</div>}
          {item.evidence && <div className="evidence">{item.evidence}</div>}
          <a className="apply" href={item.job.url} target="_blank" rel="noreferrer">View job <ArrowUpRight size={15}/></a>
        </article>)}</div>
        {data.results.length === 0 && <div className="empty-state">No matches found. Try a broader role or location.</div>}
      </section>}
      <footer><span>jobmatch</span><span>Job listings provided by Adzuna · Match scores are AI estimates.</span></footer>
    </main>
  </div>
}
