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


def create_app(service: CorpusService, token: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        service.close()

    app = FastAPI(title="Neo4j Corpus Research Service", version="1.0.0", lifespan=lifespan)

    def authorize(authorization: str | None = Header(default=None)) -> None:
        expected = f"Bearer {token}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid bearer token.")

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
