# JobMatch

A small, database-free resume-to-job matching prototype. Upload a text-based PDF resume, search Adzuna listings, retrieve relevant jobs with BGE embeddings, and evaluate the shortlist with Groq `openai/gpt-oss-120b`.

## Requirements

- Python 3.11 or 3.12
- Node.js 20+
- A Groq API key
- Adzuna Application ID and Application Key

## Run locally

### 1. Configure the backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `backend/.env` and set:

```env
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-120b
ADZUNA_APP_ID=your_adzuna_application_id
ADZUNA_APP_KEY=your_adzuna_application_key
ADZUNA_COUNTRY=in
```

Get Adzuna credentials at <https://developer.adzuna.com/>. The default country code is `in` (India).

Start the backend:

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### 2. Start the frontend

In a second terminal, from the project root:

```bash
cd frontend
npm install
npm run dev
```

Open <http://127.0.0.1:5173>.

The first matching request downloads `BAAI/bge-small-en-v1.5` from Hugging Face. It runs locally on CPU. If the embedding model cannot load, the app falls back to keyword retrieval and still uses Groq for evaluation.

## How matching works

1. PyMuPDF extracts text from the uploaded resume in memory.
2. Email addresses and phone numbers are redacted before resume text is sent to Groq.
3. Adzuna returns job listings for the role and location entered in the search form.
4. BGE embeds the resume and job descriptions locally to shortlist relevant postings.
5. Groq evaluates the shortlist and returns match scores, matched skills, missing skills and explanations.

## Notes

- No database, login, or company-board configuration is used.
- Uploaded resumes are processed in memory and are not saved by the backend.
- Scanned/image-only PDFs are not OCR'd; use a text-based PDF.
- Match scores are estimates, not hiring decisions. Confirm requirements on the original employer page.
- Adzuna API quotas, availability and reuse terms are controlled by Adzuna.
- Do not commit `.env` or share API keys. Rotate a key if it is exposed.
