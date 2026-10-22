"""Create the initial knowledge assistant data model.

Revision ID: 0001_initial_data_model
Revises:
Create Date: 2026-10-22
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0001_initial_data_model"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "admin_users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("role IN ('admin', 'assistant')", name="ck_admin_users_role"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )

    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("authority_office", sa.String(length=200), nullable=False),
        sa.Column("source_type", sa.String(length=20), nullable=False),
        sa.Column("storage_url", sa.Text(), nullable=True),
        sa.Column("publication_date", sa.Date(), nullable=True),
        sa.Column("effective_date", sa.Date(), nullable=False),
        sa.Column("expiry_date", sa.Date(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("authored_by", sa.Uuid(), nullable=False),
        sa.Column("approved_by", sa.Uuid(), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("supersedes_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("source_type IN ('uploaded', 'manual')", name="ck_documents_source_type"),
        sa.CheckConstraint(
            "status IN ('pending', 'active', 'superseded', 'withdrawn', 'rejected')",
            name="ck_documents_status",
        ),
        sa.CheckConstraint(
            "expiry_date IS NULL OR expiry_date >= effective_date",
            name="ck_documents_expiry_after_effective",
        ),
        sa.CheckConstraint(
            "supersedes_id IS NULL OR supersedes_id <> id",
            name="ck_documents_not_self_superseding",
        ),
        sa.CheckConstraint(
            "status NOT IN ('active', 'superseded', 'withdrawn') OR approved_by IS NOT NULL",
            name="ck_documents_published_have_approver",
        ),
        sa.ForeignKeyConstraint(
            ["authored_by"], ["admin_users.id"], name="fk_documents_authored_by_admin_users"
        ),
        sa.ForeignKeyConstraint(
            ["approved_by"], ["admin_users.id"], name="fk_documents_approved_by_admin_users"
        ),
        sa.ForeignKeyConstraint(
            ["supersedes_id"], ["documents.id"], name="fk_documents_supersedes_id_documents"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_documents_status_effective_expiry",
        "documents",
        ["status", "effective_date", "expiry_date"],
    )
    op.create_index("ix_documents_supersedes_id", "documents", ["supersedes_id"])

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",
            name="fk_document_chunks_document_id_documents",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_document_chunks_document_index"),
    )
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"])

    op.create_table(
        "questions_log",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("student_contact", sa.String(length=320), nullable=True),
        sa.Column("retrieval_confidence", sa.Float(), nullable=True),
        sa.Column("evidence_status", sa.String(length=20), nullable=False),
        sa.Column("escalation_reason", sa.String(length=30), nullable=True),
        sa.Column("escalated", sa.Boolean(), nullable=False),
        sa.Column("answered_by", sa.Uuid(), nullable=True),
        sa.Column("answer_document_id", sa.Uuid(), nullable=True),
        sa.Column("resolved_by", sa.Uuid(), nullable=True),
        sa.Column(
            "notification_status", sa.String(length=20), server_default="not_ready", nullable=False
        ),
        sa.Column("notification_sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("answered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "evidence_status IN ('sufficient', 'insufficient', 'conflicting')",
            name="ck_questions_log_evidence_status",
        ),
        sa.CheckConstraint(
            "escalation_reason IS NULL OR escalation_reason IN ('no_evidence', 'conflicting_evidence')",
            name="ck_questions_log_escalation_reason",
        ),
        sa.CheckConstraint(
            "notification_status IN ('not_ready', 'pending', 'sent', 'failed', 'not_applicable')",
            name="ck_questions_log_notification_status",
        ),
        sa.CheckConstraint(
            "retrieval_confidence IS NULL OR (retrieval_confidence >= 0 AND retrieval_confidence <= 1)",
            name="ck_questions_log_retrieval_confidence",
        ),
        sa.ForeignKeyConstraint(
            ["answered_by"], ["admin_users.id"], name="fk_questions_log_answered_by_admin_users"
        ),
        sa.ForeignKeyConstraint(
            ["answer_document_id"],
            ["documents.id"],
            name="fk_questions_log_answer_document_id_documents",
        ),
        sa.ForeignKeyConstraint(
            ["resolved_by"], ["admin_users.id"], name="fk_questions_log_resolved_by_admin_users"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_questions_log_escalated_created", "questions_log", ["escalated", "created_at"])
    op.create_index("ix_questions_log_notification_status", "questions_log", ["notification_status"])


def downgrade() -> None:
    op.drop_index("ix_questions_log_notification_status", table_name="questions_log")
    op.drop_index("ix_questions_log_escalated_created", table_name="questions_log")
    op.drop_table("questions_log")
    op.drop_index("ix_document_chunks_document_id", table_name="document_chunks")
    op.drop_table("document_chunks")
    op.drop_index("ix_documents_supersedes_id", table_name="documents")
    op.drop_index("ix_documents_status_effective_expiry", table_name="documents")
    op.drop_table("documents")
    op.drop_table("admin_users")

