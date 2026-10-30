# Vera — AI Knowledge Assistant

Vera is an AI knowledge assistant for helping students get clear, reliable answers from an institution’s approved information. Its goal is to make policies, procedures, dates, and other official guidance easier to find while keeping answers tied to authoritative sources.

## Product overview

The assistant is intended to answer questions using an institution’s maintained knowledge base. Administrators and assistants manage source documents; the system indexes approved material, retrieves relevant passages, and uses those passages to prepare answers. Answers should include their supporting sources. When the available material is missing, unclear, or contradictory, Vera should make that uncertainty visible and route the question for human follow-up instead of presenting a guess as policy.

The expected lifecycle is:

1. An administrator or assistant adds a document and its metadata.
2. The document is reviewed and approved before it becomes authoritative.
3. Approved documents are split into searchable passages and indexed.
4. A student asks a question; Vera retrieves relevant passages and drafts a source-grounded answer.
5. If evidence is insufficient or conflicting, Vera records the reason and escalates the question for human review.
6. The response and any follow-up are recorded so the institution can review the service and improve its knowledge base.

Document status and replacement history are part of the design. A newer document can supersede an older one, while withdrawn or rejected material can be kept out of the active knowledge base. Expiry dates are intended to help identify information that may need review.

## Project status

Development is proceeding in stages. **Stages 1–7 are implemented at the backend-module/API level.** The repository provides the backend scaffold, PostgreSQL/pgvector setup, SQLAlchemy models and migrations, an approval-gated ingestion module, three retrieval strategies, evidence validation, grounded student answers with abstention, and JWT-protected admin workflows for content review and escalated questions. The frontend and evaluation harness remain future stages.

## Current architecture

- **API:** Python 3.12, FastAPI
- **Database:** PostgreSQL 16 with the `pgvector` extension
- **Persistence:** async SQLAlchemy with Alembic migrations
- **Vector representation:** 768-dimensional embeddings, intended for multilingual-e5-base
- **Local orchestration:** Docker Compose
- **Tests and linting:** pytest and Ruff

The database foundation includes:

- `admin_users` for administrator and assistant identities
- `documents` for source metadata, approval state, expiry, and supersession history
- `document_chunks` for document passages and their vector embeddings
- `questions_log` for questions, evidence status, escalation details, answers, and notification state

The ingestion module accepts PDF and UTF-8 `.txt` files or typed text. Pending documents retain extracted/source text for review. Calling the shared indexing function requires an active document with approval metadata; successful indexing writes linked chunks and embeddings, then clears the pending text. PDF uploads are text-extracted only; scanned-image OCR and object-storage upload are not implemented yet.

The retrieval module exposes keyword search using PostgreSQL full-text search, embedding search using pgvector cosine similarity, and hybrid search using Reciprocal Rank Fusion. All three return ranked chunks with document citation metadata and filter out inactive, not-yet-effective, and expired documents. The Stage 6 `/ask` endpoint calls hybrid search before evidence validation.

The evidence validation module rechecks each retrieved chunk against its stored document, including approval status and effective/expiry dates. It estimates a retrieval confidence score, marks low-confidence results as insufficient, and flags the two highest-ranked results from different authority offices as conflicting unless their documents are linked by supersession. The confidence score is a retrieval heuristic, not a guarantee that the answer is factually correct; its threshold is configurable and should be calibrated against real questions and reviewed sources. The `/ask` pipeline uses the assessment to decide whether to generate an answer or log and escalate the question.

`POST /ask` runs hybrid retrieval followed by evidence validation. When evidence is sufficient, Vera calls the configured LLM at temperature zero and returns its answer, confidence status, and source excerpts as citations. When evidence is missing, weak, or conflicting, Vera abstains, records the question in `questions_log`, and marks it escalated without calling the LLM. The optional `student_contact` is stored only on that question-log row for the later notification workflow. The default OpenAI model is `gpt-6-luna`; the client sets its reasoning effort to `none` so temperature zero remains supported. Set `LLM_API_KEY`, `LLM_PROVIDER` (`openai` or `anthropic`), and `LLM_MODEL` before asking questions that have sufficient evidence; the missing-evidence path does not require an LLM key.

For a quick local answer-path check, a helper seeds one clearly labeled approved test document into the configured database. It uses the configured embedding model and removes the test document, chunks, test admin (when unused), and any answer logs linked to that document when you run `cleanup`:

```powershell
python backend/scripts/stage6_test_data.py seed
# Start/restart the API in another terminal, then ask the printed test question at /ask.
python backend/scripts/stage6_test_data.py cleanup
```

If the embedding model is already cached but Hugging Face is unreachable, set `$env:HF_HUB_OFFLINE = '1'` and `$env:TRANSFORMERS_OFFLINE = '1'` in that terminal before running the seed command. The configured LLM provider still needs to be reachable to receive a generated answer.

### Admin API (Stage 7)

Apply migrations, then create the first admin account. The script prompts for the password without putting it in shell history:

```powershell
alembic -c backend/alembic.ini upgrade head
python backend/scripts/create_admin.py --name "University Admin" --email "admin@example.edu"
```

The admin signs in at `POST /admin/auth/login` to receive a bearer token. Use that token in the `Authorization: Bearer <token>` header for admin routes. An admin can create assistant accounts at `POST /admin/users`. For example, log in and enter typed content:

```powershell
$loginBody = @{ email = "admin@example.edu"; password = "your-password" } | ConvertTo-Json
$login = Invoke-RestMethod -Method Post -Uri http://localhost:8000/admin/auth/login `
  -ContentType 'application/json' -Body $loginBody
$headers = @{ Authorization = "Bearer $($login.access_token)" }
$content = @{
  title = "Registration extension policy"
  authority_office = "Registrar"
  content = "Students must submit the Registration Extension Form to the Registrar."
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/admin/content `
  -Headers $headers -ContentType 'application/json' -Body $content
```

Admins' own content is approved and indexed on submission. Assistant submissions remain pending; admins review them at `GET /admin/content/pending` and approve or reject them. File uploads use `POST /admin/content/upload` with the file bytes as the request body, `X-Filename` header, and `title` and `authority_office` query parameters; PDF and UTF-8 `.txt` are supported. Escalated questions appear at `GET /admin/questions/unanswered`. Assistants can submit answers there; those remain pending until admin approval. Admin-authored answers publish immediately. Student contact addresses are not included in the queue response.

Published staff answers are indexed together with the original question so similar future questions can find the approved response. To update staff answers published before this behavior was added, preview the backfill with `python backend/scripts/backfill_staff_answer_context.py`; run it with `--apply` to rebuild those answers' chunks and embeddings.

When a question is resolved, Vera sends an email through Resend if the question has a student email and `EMAIL_API_KEY` plus a verified `EMAIL_FROM` sender are configured. The question record stores notification status (`sent`, `failed`, or `not_applicable`).

Example request:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask `
  -ContentType 'application/json' `
  -Body '{"question":"How do I request a registration extension?"}'
```

To compare all three methods on your own PDF or UTF-8 `.txt` files, use the temporary smoke-test script below. It creates active test documents in an isolated database schema, uses the configured real embedding model, prints ranked source passages for each query, and removes the temporary schema when finished. It does not add the files to your normal application data.

```powershell
python backend/scripts/retrieval_smoke_test.py `
  --files "C:\path\to\academic-policy.pdf" "C:\path\to\registration-guidance.txt" `
  --query "Can I register after the deadline?" `
  --query "Which office reviews registration extensions?"
```

Review whether the expected source appears near the top and whether its passage actually supports the question. First use several questions whose answers you can verify in the source documents. A successful search result alone does not establish that an answer is correct; answer validation is a later stage.

The student and admin workflows are available through the API; no frontend has been built yet.

## Getting started (Windows PowerShell)

### Requirements

- Python 3.12
- Docker Desktop with Docker Compose

### Configure the project

From the repository root, create your local environment file and Python virtual environment:

```powershell
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
```

Edit `.env` to configure database credentials and any provider keys. The file is ignored by Git; keep real secrets there and do not commit it. `.env.example` documents the required variables.

### Start the database and apply migrations

```powershell
docker compose up -d db
alembic -c backend/alembic.ini upgrade head
```

PostgreSQL is exposed on host port **5433**. The API container reaches it on the Compose network at port **5432**. The `.env` file contains separate database URLs for those two network contexts.

### Run the API

To start the complete Compose stack, including the API:

```powershell
docker compose up -d --build
docker compose exec api alembic upgrade head
```

The API is available at `http://localhost:8000`. The current health endpoint is `http://localhost:8000/health`; interactive API documentation is at `http://localhost:8000/docs`.

To stop the services while retaining the database volume:

```powershell
docker compose down
```

## Development checks

With the virtual environment active and the database running:

```powershell
python -m pytest backend/tests/test_ingestion.py -q
python -m pytest backend/tests/test_retrieval.py -q
python -m pytest backend/tests/test_validation.py -q
python -m pytest backend/tests/test_pipeline.py -q
python -m pytest -q
ruff check backend/app backend/tests
```

The ingestion tests exercise PDF and plain-text extraction, overlapping chunks,
pending document intake, approval gating, embedding persistence, and cleanup of
pending text. They use a deterministic test tokenizer and embedding function,
so they do not download model weights. The PostgreSQL-backed test creates and
removes an isolated schema in the configured test database.

Run the retrieval tests with `python -m pytest backend/tests/test_retrieval.py -q`.
They use a temporary schema and supplied query vectors, so they do not download
the embedding model. The real-model smoke test is separate, as shown above.

Check the current database migration revision with:

```powershell
alembic -c backend/alembic.ini current
```

## Configuration

Configuration is managed from the root `.env` file. The current variables are:

| Variable | Purpose |
| --- | --- |
| `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD` | Local PostgreSQL container credentials |
| `DATABASE_URL` | Database URL for commands running on the host |
| `DATABASE_URL_DOCKER` | Database URL for the API container |
| `LLM_API_KEY` | API key for the selected answer-generation provider |
| `LLM_PROVIDER`, `LLM_MODEL` | Answer-generation provider (`openai` or `anthropic`) and model name |
| `EMAIL_API_KEY` | Resend API key used for answer-resolution emails |
| `EMAIL_FROM` | Verified sender identity used for resolution emails via Resend |
| `JWT_SECRET` | Secret reserved for authentication/session features |
| `EMBEDDING_MODEL_NAME` | Sentence-transformer model used for document embeddings |
| `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS` | Chunk sizing for ingestion |
| `EVIDENCE_CONFIDENCE_THRESHOLD` | Minimum retrieval confidence for evidence to be considered sufficient (default `0.65`) |

Provider keys can remain empty while working on the current foundation. Replace development secrets before using the project in a deployed environment.

## Repository layout

```text
backend/
  app/
    db/              SQLAlchemy models, async session, and Alembic migrations
    ingestion/       File parsing, direct intake, chunking, and embedding pipeline
    main.py          FastAPI application and health endpoint
    config.py        Application settings loaded from .env/environment
  tests/             Backend tests
  requirements.txt   Python dependencies
docker-compose.yml   Local PostgreSQL and API services
.env.example        Configuration template
pytest.ini           pytest configuration
```

## Design principles

- Use approved institutional material as the source of truth.
- Keep answers grounded in retrievable evidence and make source attribution clear.
- Escalate missing or conflicting evidence for human handling.
- Preserve document status and replacement history so outdated guidance is not treated as current.
- Keep provider credentials and environment-specific configuration outside source control.
