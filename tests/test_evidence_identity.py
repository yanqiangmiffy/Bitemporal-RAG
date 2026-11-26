from asof.config import Config
from asof.methods.asof import AsofMethod
from asof.schema import Fact, Question, QType
from asof.temporal import Interval, parse_date


def test_equivalent_stores_share_generator_prompt_and_resolve_local_citation(monkeypatch):
    class Retriever:
        def __init__(self, ids, texts, *args, **kwargs):
            self.hits = list(zip(ids, texts))

        def search(self, *args, **kwargs):
            return self.hits

    class Client:
        def __init__(self):
            self.prompts = []

        def chat_json(self, messages, *args, **kwargs):
            self.prompts.append(messages)
            return {"answer": "800", "fact_id": "E1"}

    monkeypatch.setattr("asof.methods.asof.Retriever", Retriever)
    question = Question("q", "demo", QType.POST, "What is the allowance?",
                        parse_date("2025-01-01"), parse_date("2025-01-01"), "reference")
    clients, answers = [], []
    for internal_id in ["f000001", "f000910"]:
        client = Client()
        method = AsofMethod(Config({}), client)
        method.store.append(Fact(obj="allowance", field_name="amount", value="800",
                                 valid=Interval.of("2024-01-01", "INF"),
                                 visible=Interval.of("2024-01-01", "INF"),
                                 source_span="The allowance is 800.", doc_id="original_notice",
                                 fact_id=internal_id))
        clients.append(client)
        answers.append(method.answer(question))
    assert clients[0].prompts == clients[1].prompts
    assert [a.fact_id for a in answers] == ["f000001", "f000910"]
    assert all(a.doc == "original_notice" and a.cited_span == "The allowance is 800." for a in answers)
