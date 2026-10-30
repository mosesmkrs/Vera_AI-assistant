"""Add password hashes for admin API authentication.

Revision ID: 0004_admin_auth
Revises: 0003_retrieval_indexes
Create Date: 2026-10-23
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_admin_auth"
down_revision: str | None = "0003_retrieval_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("admin_users", sa.Column("password_hash", sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("admin_users", "password_hash")
