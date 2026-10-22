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

Development is proceeding in stages. **Stage 1 (project foundation) and Stage 2 (database foundation) are implemented.** The current repository provides the backend scaffold, configuration, PostgreSQL/pgvector setup, SQLAlchemy models, an Alembic migration, and a health endpoint. It does not yet implement document upload and approval screens, document processing and embedding jobs, retrieval-augmented answering, human escalation workflows, or notification delivery. Those are future stages.

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

These tables establish the data model. They do not imply that the user interfaces or workflows for managing them are already available.

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
python -m pytest -q
ruff check backend/app backend/tests
```

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
| `LLM_API_KEY` | Future language-model provider integration |
| `EMAIL_API_KEY` | Future email or notification provider integration |
| `JWT_SECRET` | Secret reserved for authentication/session features |

Provider keys can remain empty while working on the current foundation. Replace development secrets before using the project in a deployed environment.

## Repository layout

```text
backend/
  app/
    db/              SQLAlchemy models, async session, and Alembic migrations
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
