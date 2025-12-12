from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.db import Base


class Verification(Base):
    __tablename__ = "verifications"

    id = Column(BigInteger, primary_key=True)
    request_id = Column(UUID(as_uuid=True), unique=True, nullable=False)
    user_id = Column(BigInteger, nullable=True)
    input_text = Column(Text, nullable=False)
    model_used = Column(String(255), nullable=False)
    latency_ms = Column(Integer, nullable=False)
    verdict = Column(String(20), nullable=False)
    confidence = Column(Numeric(3, 2), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    claims = relationship("Claim", back_populates="verification", cascade="all, delete-orphan")


class Claim(Base):
    __tablename__ = "claims"

    id = Column(BigInteger, primary_key=True)
    verification_id = Column(BigInteger, ForeignKey("verifications.id", ondelete="CASCADE"), nullable=False)
    claim_text = Column(Text, nullable=False)
    verdict = Column(String(20), nullable=False)
    confidence = Column(Numeric(3, 2), nullable=False)
    reasoning = Column(Text, nullable=False)
    claim_embedding = Column(Vector(384), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    verification = relationship("Verification", back_populates="claims")
    evidence = relationship("Evidence", back_populates="claim", cascade="all, delete-orphan")


class Source(Base):
    __tablename__ = "sources"

    id = Column(BigInteger, primary_key=True)
    url = Column(Text, unique=True)
    title = Column(Text, nullable=True)
    domain = Column(Text, nullable=False)
    credibility_score = Column(Numeric(3, 2), nullable=True, default=0.50)
    first_seen_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    evidence = relationship("Evidence", back_populates="source")


class Evidence(Base):
    __tablename__ = "evidence"
    __table_args__ = (UniqueConstraint("claim_id", "source_id", "snippet", name="uq_claim_source_snippet"),)

    id = Column(BigInteger, primary_key=True)
    claim_id = Column(BigInteger, ForeignKey("claims.id", ondelete="CASCADE"), nullable=False)
    source_id = Column(BigInteger, ForeignKey("sources.id", ondelete="CASCADE"), nullable=False)
    snippet = Column(Text, nullable=False)
    relevance_score = Column(Numeric(4, 3), nullable=False)
    snippet_embedding = Column(Vector(384), nullable=True)
    captured_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    claim = relationship("Claim", back_populates="evidence")
    source = relationship("Source", back_populates="evidence")
