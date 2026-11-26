from asof.config import Config
from asof.methods.asof import AsofMethod
from asof.methods.baselines import FullRebuild
from asof.schema import Notice
from asof.temporal import parse_date


class CountingClient:
    """A real classifier fast path makes no request on an empty candidate set."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.stats = {"api_calls": 0, "cache_hits": 0}

    def chat_json(self, *args, **kwargs):
        cached, result = next(self.responses)
        self.stats["cache_hits" if cached else "api_calls"] += 1
        return result


FACT = {"facts": [{"object": "same object", "field": "same field", "value": "v"}]}
REPLACE = {"relations": [{"relation": "replace"}]}


def notice(i):
    return Notice(str(i), parse_date(f"202{i}-01-01"), "source")


def test_incremental_cost_counts_actual_requests_and_distinguishes_cache_hits():
    client = CountingClient([(False, FACT), (True, FACT), (False, REPLACE)])
    method = AsofMethod(Config({}), client)
    first = method.update(notice(0))
    second = method.update(notice(1))
    assert (first.llm_calls, first.llm_api_calls, first.llm_cache_hits) == (1, 1, 0)
    assert (second.llm_calls, second.llm_api_calls, second.llm_cache_hits) == (2, 1, 1)


def test_rebuild_cost_includes_classification_as_well_as_reextraction():
    client = CountingClient([(False, FACT), (True, FACT), (True, FACT), (False, REPLACE)])
    method = FullRebuild(Config({}), client)
    first = method.update(notice(0))
    second = method.update(notice(1))
    assert first.llm_calls == 1
    assert (second.llm_calls, second.llm_api_calls, second.llm_cache_hits) == (3, 1, 2)
