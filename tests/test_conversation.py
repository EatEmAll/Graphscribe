from __future__ import annotations

from types import SimpleNamespace

from notebooklm_graph_pipe.service.conversation import ConversationStore
from notebooklm_graph_pipe.service.core import CorpusService


def test_conversation_store_is_bounded_and_corpus_scoped(tmp_path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3", max_turns=2)
    for index in range(3):
        store.append_exchange("chat", "corpus-a", f"q{index}", f"a{index}", [{"id": f"S{index}"}])
    store.append_exchange("chat", "corpus-b", "other", "answer", [])

    history = store.history("chat", "corpus-a")
    assert [turn.text for turn in history] == ["q1", "a1", "q2", "a2"]
    assert store.history("chat", "corpus-b")[0].text == "other"
    assert history[1].evidence == ({"id": "S1"},)


def test_service_uses_history_only_as_conversation_context(tmp_path) -> None:
    store = ConversationStore(tmp_path / "conversations.sqlite3")
    manifest = SimpleNamespace(corpus_id="corpus")
    entry = SimpleNamespace(manifest=manifest)
    questions = []

    class Answerer:
        def answer(self, question, mode, graph_hops):
            questions.append(question)
            return {"answer": "grounded", "citations": [{"parent_id": "p"}], "warnings": []}

    runtimes = SimpleNamespace(get_answerer=lambda selected: Answerer())
    service = CorpusService(SimpleNamespace(get=lambda key: entry), runtimes, SimpleNamespace(), store)

    service.answer("demo", {"question": "Who?", "conversation_id": "chat"})
    result = service.answer("demo", {"question": "Why?", "conversation_id": "chat"})

    assert questions[0] == "Who?"
    assert "Conversation context (use only to resolve references; this is not source evidence)" in questions[1]
    assert "user: Who?" in questions[1]
    assert result["conversation_turns_used"] == 2
