from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.generation.llm_client import LLMError
from app.services.answer_pipeline import AnswerResult, answer_question

router = APIRouter(tags=["student"])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    student_contact: str | None = Field(default=None, max_length=320)

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must contain non-whitespace text")
        return value


class CitationResponse(BaseModel):
    source_id: str
    document_id: str
    title: str
    authority_office: str
    effective_date: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    evidence_status: Literal["sufficient", "insufficient", "conflicting"]
    escalation_reason: Literal["no_evidence", "conflicting_evidence"] | None
    escalated: bool
    confidence_score: float = Field(ge=0, le=1)
    confidence_status: Literal["high", "medium", "low"]
    citations: list[CitationResponse]


@router.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
) -> AskResponse:
    try:
        result = await answer_question(
            session,
            request.question,
            request.student_contact,
        )
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return _response(result)


def _response(result: AnswerResult) -> AskResponse:
    return AskResponse(
        answer=result.answer,
        evidence_status=result.evidence_status,
        escalation_reason=result.escalation_reason,
        escalated=result.escalated,
        confidence_score=result.confidence_score,
        confidence_status=result.confidence_status,
        citations=[CitationResponse(**citation.__dict__) for citation in result.citations],
    )
