import json
import httpx
import numpy as np
import pytest
from openai import OpenAI
from src.llm import LLM, Embedder
from src.retrieval import DenseRetriever, GlobalDenseRetriever
from scripts.build_index import build_index


def test_openai_transport_cache_and_token_caps(tmp_path):
    bodies = []
    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1,
            "model": "test", "choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}})
    client = OpenAI(base_url="http://test/v1", api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    llm = LLM("http://test/v1", "test", cache_dir=tmp_path, client=client)
    assert llm("prompt", max_tokens=256) == "answer"
    assert llm("prompt", max_tokens=256) == "answer"
    assert len(bodies) == 1
    llm("prompt", max_tokens=128)
    assert len(bodies) == 2
    assert bodies[0]["max_tokens"] == 256 and bodies[0]["temperature"] == 0 and bodies[0]["seed"] == 42
    assert llm.get_call_stats()["cache_hits"] == 1
    assert llm.get_call_stats()["api_calls"] == 2


def test_json_arrays_are_not_lost_to_inner_objects():
    assert LLM.parse_json('text [{"name":"a"},{"name":"b"}] end') == [{"name": "a"}, {"name": "b"}]


def test_retrieval_uses_distinct_query_and_passage_roles():
    class Emb:
        def encode_passages(self, texts):
            assert texts == ["one", "two"]
            return np.eye(2, dtype=np.float32)
        def encode_queries(self, text):
            assert text == "question"
            return np.array([[0, 1]], dtype=np.float32)
    r = DenseRetriever(Emb())
    r.build([{"text": "one"}, {"text": "two"}])
    assert r.search("question", 1)[0]["text"] == "two"


def test_persistent_index_round_trip(tmp_path):
    corpus = tmp_path / "wiki.tsv"
    corpus.write_text('id\ttext\ttitle\n1\tParis France\tParis\n2\tTokyo Japan\tTokyo\n')
    embedder = Embedder("synthetic-test", force_mock=True)
    index_dir = tmp_path / "index"
    build_index(corpus, index_dir, embedder, batch_size=1)
    retrieved = GlobalDenseRetriever(embedder, index_dir)
    result = retrieved.search("Tokyo Japan", 1)
    assert result[0]["id"] == "2"
    assert result[0]["title"] == "Tokyo"
    metadata = json.loads((index_dir / "metadata.json").read_text())
    metadata["embedding"]["passage_prefix"] = "wrong: "
    (index_dir / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="mismatch"):
        GlobalDenseRetriever(embedder, index_dir)
