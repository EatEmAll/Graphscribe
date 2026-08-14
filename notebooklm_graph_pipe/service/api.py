from __future__ import annotations

import secrets
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .core import CorpusService


class SearchBody(BaseModel):
    query: str
    mode: str = "graph_hybrid"
    top_k: int = Field(12, ge=1, le=50)
    graph_hops: int = Field(1, ge=1, le=2)
    filters: dict[str, Any] = Field(default_factory=dict)
    include_diagnostics: bool = False


class AnswerBody(BaseModel):
    question: str
    mode: str = "graph_hybrid"
    graph_hops: int = Field(1, ge=1, le=2)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=128)


class SourceProbe(BaseModel):
    provider: str = Field(default="", max_length=100)
    provider_source_id: str = Field(default="", max_length=500)
    canonical_uri: str | None = Field(default=None, max_length=4000)
    content_checksum: str = Field(default="", max_length=128)
    notebooklm_source_id: str | None = Field(default=None, max_length=500)
    title: str = Field(default="discovery probe", max_length=1000)
    source_type: str = Field(default="document", max_length=100)


class ResolveSourcesBody(BaseModel):
    probes: list[SourceProbe] = Field(min_length=1, max_length=100)


class IngestionBody(BaseModel):
    idempotency_key: str = Field(min_length=1, max_length=200)
    package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    package_path: str = Field(min_length=1, max_length=2000)


class EvaluationBody(BaseModel):
    baseline_quality_ratio: float = Field(ge=0, le=2, allow_inf_nan=False)
    effective_citation_ratio: float = Field(ge=0, le=1, allow_inf_nan=False)
    unsupported_claim_delta: float = Field(allow_inf_nan=False)
    graph_expansion_ratio: float = Field(ge=0, le=1, allow_inf_nan=False)
    capacity_headroom_ratio: float = Field(ge=0, le=1, allow_inf_nan=False)
    source_canary_retrieved: bool


def create_app(service: CorpusService, token: str, write_token: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        service.close()

    app = FastAPI(title="Neo4j Corpus Research Service", version="1.0.0", lifespan=lifespan)

    def authorize(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {token}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid bearer token.")

    def authorize_write(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {write_token}" if write_token else ""
        if not expected or authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid write bearer token.")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/corpora", dependencies=[Depends(authorize)])
    def corpora():
        return service.list_corpora()

    @app.get("/v1/corpora/{corpus_key}", dependencies=[Depends(authorize)])
    def corpus(corpus_key: str):
        return _call(service.get_corpus, corpus_key)

    @app.post("/v1/corpora/{corpus_key}/sync", status_code=202, dependencies=[Depends(authorize)])
    def sync(corpus_key: str):
        return _call(service.submit_sync, corpus_key)

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
    def job(job_id: str):
        return _call(service.get_job, job_id)

    @app.post(
        "/v1/corpora/{corpus_key}/sources:resolve",
        dependencies=[Depends(authorize_write)],
    )
    def resolve_sources(corpus_key: str, body: ResolveSourcesBody):
        return _call(service.resolve_sources, corpus_key, [item.model_dump() for item in body.probes])

    @app.post(
        "/v1/corpora/{corpus_key}/ingestions",
        status_code=202,
        dependencies=[Depends(authorize_write)],
    )
    def submit_ingestion(corpus_key: str, body: IngestionBody):
        return _call(service.submit_ingestion, corpus_key, body.model_dump())

    @app.get("/v1/ingestions/{ingestion_id}", dependencies=[Depends(authorize_write)])
    def ingestion(ingestion_id: str):
        return _call(service.get_ingestion, ingestion_id)

    @app.post("/v1/ingestions/{ingestion_id}:evaluate", dependencies=[Depends(authorize_write)])
    def evaluate_ingestion(ingestion_id: str, body: EvaluationBody):
        return _call(service.evaluate_ingestion, ingestion_id, body.model_dump())

    @app.post("/v1/ingestions/{ingestion_id}:accept", dependencies=[Depends(authorize_write)])
    def accept_ingestion(ingestion_id: str):
        return _call(service.accept_ingestion, ingestion_id)

    @app.post("/v1/ingestions/{ingestion_id}:rollback", dependencies=[Depends(authorize_write)])
    def rollback_ingestion(ingestion_id: str):
        return _call(service.rollback_ingestion, ingestion_id)

    @app.post("/v1/corpora/{corpus_key}/search", dependencies=[Depends(authorize)])
    def search(corpus_key: str, body: SearchBody):
        return _call(service.search, corpus_key, body.model_dump())

    @app.post("/v1/corpora/{corpus_key}/answer", dependencies=[Depends(authorize)])
    def answer(corpus_key: str, body: AnswerBody):
        return _call(service.answer, corpus_key, body.model_dump())

    @app.post("/v1/corpora/{corpus_key}/answer/stream", dependencies=[Depends(authorize)])
    def answer_stream(corpus_key: str, body: AnswerBody):
        def events():
            try:
                for event in service.answer_stream(corpus_key, body.model_dump()):
                    yield f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
            except (KeyError, ValueError, RuntimeError) as exc:
                yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/v1/corpora/{corpus_key}/documents", dependencies=[Depends(authorize)])
    def documents(corpus_key: str):
        return _call(service.list_documents, corpus_key)

    @app.get("/v1/corpora/{corpus_key}/documents/{document_id}", dependencies=[Depends(authorize)])
    def document(corpus_key: str, document_id: str):
        return _call(service.get_document, corpus_key, document_id)

    @app.delete("/v1/corpora/{corpus_key}/documents/{document_id}", dependencies=[Depends(authorize)])
    def delete_document(corpus_key: str, document_id: str):
        return _call(service.delete_document, corpus_key, document_id)

    return app


def _call(function, *args):
    try:
        return function(*args)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
