"""Store source text while documents await approval.

Revision ID: 0002_pending_document_content
Revises: 0001_initial_data_model
Create Date: 2026-10-22
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_pending_document_content"
down_revision: str | None = "0001_initial_data_model"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("pending_content", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "pending_content")
