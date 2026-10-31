"""Store answer support confidence separately from retrieval rank.

Revision ID: 0005_answer_confidence
Revises: 0004_admin_auth
Create Date: 2026-10-30
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_answer_confidence"
down_revision: str | None = "0004_admin_auth"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "questions_log",
        sa.Column(
            "answer_confidence",
            sa.Float(),
            nullable=True,
            comment="Grounding verifier's answer support score",
        ),
    )
    op.create_check_constraint(
        "ck_questions_log_answer_confidence",
        "questions_log",
        "answer_confidence IS NULL OR (answer_confidence >= 0 AND answer_confidence <= 1)",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_questions_log_answer_confidence", "questions_log", type_="check"
    )
    op.drop_column("questions_log", "answer_confidence")
