from __future__ import annotations

import hashlib
import html
import io
import json
import os
import re
from functools import lru_cache
from typing import Any

import fitz
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    adzuna_app_id: str = ""
    adzuna_app_key: str = ""
    adzuna_country: str = "in"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    max_upload_mb: int = 10
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)


settings = Settings()
app = FastAPI(title="JobMatch API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Job(BaseModel):
    id: str
    title: str
    company: str = ""
    location: str = ""
    source: str
    url: str
    description: str = ""
    posted_at: str | None = None
    employment_type: str | None = None
    salary: str | None = None


class MatchResult(BaseModel):
    job: Job
    score: int = Field(ge=0, le=100)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    reason: str = ""
    evidence: str = ""
    semantic_similarity: float = 0.0


class MatchResponse(BaseModel):
    resume_summary: str
    skills: list[str]
    results: list[MatchResult]
    jobs_found: int
    sources_used: list[str]
    source_warnings: list[str]
    retrieval_method: str
    model: str


class SourceError(Exception):
    pass


def clean_html(value: Any) -> str:
    if not value:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def job_id(url: str, title: str, company: str) -> str:
    return hashlib.sha1(f"{url}|{title}|{company}".encode("utf-8")).hexdigest()[:14]


def normalize_job(title: Any, company: Any, location: Any, source: str, url: Any,
                  description: Any, posted_at: Any = None, employment_type: Any = None,
                  salary: Any = None) -> Job | None:
    title_s, url_s = clean_html(title), str(url or "").strip()
    if not title_s or not url_s.startswith("http"):
        return None
    company_s = clean_html(company)
    return Job(
        id=job_id(url_s, title_s, company_s), title=title_s,
        company=company_s or "Unknown company", location=clean_html(location) or "Not specified",
        source=source, url=url_s, description=clean_html(description)[:12000],
        posted_at=str(posted_at) if posted_at else None,
        employment_type=clean_html(employment_type) or None,
        salary=clean_html(salary) or None,
    )


async def fetch_json(client: httpx.AsyncClient, url: str, params: dict | None = None) -> Any:
    response = await client.get(url, params=params, headers={"User-Agent": "JobMatchPrototype/0.1"})
    response.raise_for_status()
    return response.json()


async def fetch_adzuna(client: httpx.AsyncClient, query: str, location: str) -> list[Job]:
    if not settings.adzuna_app_id or not settings.adzuna_app_key:
        return []
    country = re.sub(r"[^a-z]", "", settings.adzuna_country.lower()) or "in"
    data = await fetch_json(
        client,
        f"https://api.adzuna.com/v1/api/jobs/{country}/search/1",
        {
            "app_id": settings.adzuna_app_id,
            "app_key": settings.adzuna_app_key,
            "results_per_page": 40,
            "what": query or "software engineer",
            "where": location or "India",
            "content-type": "application/json",
        },
    )
    jobs: list[Job] = []
    for item in data.get("results", []):
        company = (item.get("company") or {}).get("display_name", "")
        loc = (item.get("location") or {}).get("display_name", "")
        salary = None
        if item.get("salary_min") or item.get("salary_max"):
            salary = f"{item.get('salary_min', '')} - {item.get('salary_max', '')} {item.get('salary_currency', '')}".strip()
        job = normalize_job(item.get("title"), company, loc, "Adzuna", item.get("redirect_url"),
                            item.get("description"), item.get("created"), item.get("contract_type"), salary)
        if job:
            jobs.append(job)
    return jobs


async def get_jobs(query: str, location: str) -> tuple[list[Job], list[str], list[str]]:
    warnings: list[str] = []
    if not (settings.adzuna_app_id and settings.adzuna_app_key):
        raise HTTPException(
            status_code=503,
            detail="Adzuna is not configured. Add ADZUNA_APP_ID and ADZUNA_APP_KEY to backend/.env.",
        )
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=8.0), follow_redirects=True) as client:
            jobs = await fetch_adzuna(client, query, location)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        detail = "Adzuna rejected the request. Check your app ID/key and API quota."
        if status == 429:
            detail = "Adzuna rate limit reached. Wait a little and try again."
        raise HTTPException(status_code=502, detail=f"Adzuna API error ({status}): {detail}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch jobs from Adzuna: {str(exc)[:240]}") from exc
    if not jobs:
        warnings.append("Adzuna returned no jobs for this search. Try a broader role or location.")
    return jobs, (["Adzuna"] if jobs else []), warnings


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.embedding_model, device="cpu")


def lexical_retrieval(resume: str, jobs: list[Job], query: str, limit: int = 14) -> tuple[list[tuple[Job, float]], str]:
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]{1,}", (resume + " " + query).lower()))
    scored = []
    for job in jobs:
        body = f"{job.title} {job.company} {job.location} {job.description}".lower()
        words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]{1,}", body))
        score = len(tokens & words) / max(1, len(tokens))
        scored.append((job, score))
    scored.sort(key=lambda row: row[1], reverse=True)
    return scored[:limit], "keyword fallback (embedding model unavailable)"


def retrieve_jobs(resume: str, jobs: list[Job], query: str, limit: int = 14) -> tuple[list[tuple[Job, float]], str]:
    try:
        model = get_embedding_model()
        job_texts = [f"{j.title}. {j.company}. {j.location}. {j.description[:4500]}" for j in jobs]
        query_text = f"{query}. Candidate resume and skills: {resume[:10000]}"
        vectors = model.encode([query_text, *job_texts], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        similarities = vectors[1:] @ vectors[0]
        ranked = sorted(zip(jobs, similarities.tolist()), key=lambda row: row[1], reverse=True)
        return [(job, float(score)) for job, score in ranked[:limit]], f"{settings.embedding_model} semantic retrieval"
    except Exception:
        return lexical_retrieval(resume, jobs, query, limit)


def redact_contact_details(text: str) -> str:
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email removed]", text)
    text = re.sub(r"(?:\+?\d[\d\s().-]{7,}\d)", "[phone removed]", text)
    return text


def parse_json_response(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise


async def groq_evaluate(resume_text: str, query: str, location: str,
                        selected: list[tuple[Job, float]]) -> dict[str, Any]:
    if not settings.groq_api_key or settings.groq_api_key == "gsk_replace_me":
        raise HTTPException(status_code=400, detail="Add your Groq API key to backend/.env (GROQ_API_KEY).")
    compact_jobs = []
    for job, similarity in selected:
        compact_jobs.append({
            "id": job.id, "title": job.title, "company": job.company, "location": job.location,
            "description": job.description[:2800], "employment_type": job.employment_type,
            "salary": job.salary, "semantic_similarity": round(similarity, 4),
        })
    system = (
        "You are a careful job matching evaluator. Evaluate only the supplied resume and job data. "
        "Never invent candidate experience, skills, qualifications, salary, or job requirements. "
        "Do not infer protected traits. Treat missing evidence as unknown, not as a mismatch. "
        "Return valid JSON only, with keys resume_summary (string), skills (array of strings), "
        "matches (array). Each match must contain job_id, score (integer 0-100, never a string), matched_skills, "
        "IMPORTANT: Always return score as a JSON number, e.g. 85, never as '85' or '85%'. "
        "missing_skills, reason, evidence. Score based on evidence of skills, responsibilities, "
        "seniority/experience, and stated location preference. A score is suitability, not hiring probability. "
        "Keep reasons concise. Include only skills actually supported by the resume as matched."
    )
    user = {
        "candidate_resume": redact_contact_details(resume_text[:16000]),
        "job_search_query": query,
        "preferred_location": location,
        "scoring_guidance": "Evaluate actual fit. Missing years/education requirements should reduce fit only when explicitly required. Do not reward keyword mentions without context.",
        "jobs": compact_jobs,
        "required_output": {
            "resume_summary": "one sentence",
            "skills": ["resume skills supported by evidence"],
            "matches": [{"job_id": "exact provided id", "score": 0, "matched_skills": [], "missing_skills": [], "reason": "", "evidence": "brief resume/job evidence"}],
        },
    }
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.groq_api_key}", "Content-Type": "application/json"},
            json={
                "model": settings.groq_model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "job_match_results",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "resume_summary": {"type": "string"},
                                "skills": {"type": "array", "items": {"type": "string"}},
                                "matches": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "job_id": {"type": "string"},
                                            "score": {"type": "integer", "minimum": 0, "maximum": 100},
                                            "matched_skills": {"type": "array", "items": {"type": "string"}},
                                            "missing_skills": {"type": "array", "items": {"type": "string"}},
                                            "reason": {"type": "string"},
                                            "evidence": {"type": "string"},
                                        },
                                        "required": ["job_id", "score", "matched_skills", "missing_skills", "reason", "evidence"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                            "required": ["resume_summary", "skills", "matches"],
                            "additionalProperties": False,
                        },
                    },
                },
                "max_completion_tokens": 5000,
            },
        )
        if response.is_error:
            try:
                detail = response.json().get("error", {}).get("message", response.text)
            except Exception:
                detail = response.text
            raise HTTPException(status_code=502, detail=f"Groq API error ({response.status_code}): {str(detail)[:500]}")
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        try:
            return parse_json_response(content)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Could not parse Groq's JSON response: {exc}") from exc


@app.get("/api/health")
async def health():
    return {"ok": True, "model": settings.groq_model, "adzuna_enabled": bool(settings.adzuna_app_id and settings.adzuna_app_key)}


@app.post("/api/match", response_model=MatchResponse)
async def match_jobs(
    resume: UploadFile = File(...),
    query: str = Form("software engineer internship"),
    location: str = Form("India"),
):
    if resume.content_type not in {"application/pdf", "application/octet-stream"} and not (resume.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF resume.")
    raw = await resume.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"PDF exceeds the {settings.max_upload_mb} MB limit.")
    try:
        doc = fitz.open(stream=raw, filetype="pdf")
        if doc.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported.")
        resume_text = "\n".join(page.get_text("text") for page in doc).strip()
        doc.close()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read this PDF: {exc}") from exc
    if len(resume_text) < 80:
        raise HTTPException(status_code=400, detail="Very little text was found. This may be a scanned PDF; use a text-based PDF for this prototype.")

    jobs, sources, warnings = await get_jobs(query, location)
    if not jobs:
        raise HTTPException(status_code=404, detail="No jobs were returned. Try a broader role or location, or check your Adzuna credentials.")

    shortlisted, retrieval_method = retrieve_jobs(resume_text, jobs, f"{query}. Preferred location: {location}")
    evaluation = await groq_evaluate(resume_text, query, location, shortlisted)
    by_id = {job.id: (job, sim) for job, sim in shortlisted}
    results: list[MatchResult] = []
    for item in evaluation.get("matches", []):
        pair = by_id.get(str(item.get("job_id", "")))
        if not pair:
            continue
        job, sim = pair
        results.append(MatchResult(
            job=job,
            score=max(0, min(100, int(item.get("score", round(max(0, sim) * 100))))),
            matched_skills=[str(x) for x in item.get("matched_skills", [])[:12]],
            missing_skills=[str(x) for x in item.get("missing_skills", [])[:12]],
            reason=str(item.get("reason", ""))[:600],
            evidence=str(item.get("evidence", ""))[:600],
            semantic_similarity=round(float(sim), 4),
        ))
    # Keep all retrieved jobs visible if the model omitted any from its JSON.
    included = {r.job.id for r in results}
    for job, sim in shortlisted:
        if job.id not in included:
            results.append(MatchResult(job=job, score=max(0, min(100, round(max(0, sim) * 100))), semantic_similarity=round(float(sim), 4), reason="Semantic retrieval result; the model did not return a detailed evaluation."))
    results.sort(key=lambda r: r.score, reverse=True)
    return MatchResponse(
        resume_summary=str(evaluation.get("resume_summary", "Resume parsed successfully."))[:500],
        skills=[str(x) for x in evaluation.get("skills", [])[:25]],
        results=results,
        jobs_found=len(jobs), sources_used=sources, source_warnings=warnings,
        retrieval_method=retrieval_method, model=settings.groq_model,
    )
