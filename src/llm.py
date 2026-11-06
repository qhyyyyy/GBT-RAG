import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np


class LLM:
    CACHE_VERSION = "gbtrag-paper-aligned-2"

    def __init__(self, base_url, model, temperature=0.0, max_tokens=2048,
                 timeout=120, cache_dir=None, seed=42, max_retries=2, client=None):
        from openai import OpenAI
        self.base_url = base_url.rstrip("/")
        self.model = str(model)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.seed = int(seed)
        self.max_retries = int(max_retries)
        self.context = ""
        self._client = client or OpenAI(
            base_url=self.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            max_retries=0, timeout=self.timeout)
        self._cache = None
        if cache_dir:
            import diskcache
            self._cache = diskcache.Cache(str(cache_dir))
        self.reset_call_count()

    def _cache_key(self, prompt, system=None, max_tokens=None):
        payload = [self.CACHE_VERSION, self.base_url, self.model, self.context,
                   system or "", prompt, self.temperature,
                   self.max_tokens if max_tokens is None else int(max_tokens), self.seed]
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()

    def __call__(self, prompt, system=None, max_tokens=None):
        limit = self.max_tokens if max_tokens is None else int(max_tokens)
        if limit <= 0:
            raise ValueError("max_tokens must be positive")
        self._call_count += 1
        key = self._cache_key(prompt, system, limit)
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache_hits += 1
                self._logical_prompt_tokens += cached.get("prompt_tokens", 0)
                self._logical_completion_tokens += cached.get("completion_tokens", 0)
                return cached["text"]
        messages = []
        system_text = "\n\n".join(x for x in [system, self.context] if x)
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": prompt})
        for attempt in range(self.max_retries + 1):
            self._api_call_count += 1
            try:
                request_started = time.perf_counter()
                try:
                    result = self._client.chat.completions.create(
                        model=self.model, messages=messages, temperature=self.temperature,
                        max_tokens=limit, seed=self.seed, timeout=self.timeout)
                finally:
                    self._api_wait_seconds += time.perf_counter() - request_started
                if not result.choices:
                    raise RuntimeError("LLM returned no choices")
                text = result.choices[0].message.content or ""
                if not text.strip():
                    raise RuntimeError("LLM returned empty text")
                usage = result.usage
                pt = int(usage.prompt_tokens or 0) if usage else 0
                ct = int(usage.completion_tokens or 0) if usage else 0
                self._prompt_tokens += pt
                self._completion_tokens += ct
                self._logical_prompt_tokens += pt
                self._logical_completion_tokens += ct
                self._truncated += int(result.choices[0].finish_reason == "length")
                if self._cache is not None:
                    self._cache.set(key, {"text": text, "prompt_tokens": pt, "completion_tokens": ct})
                return text
            except Exception as exc:
                status = getattr(exc, "status_code", None)

                if status in {400, 401, 403, 404, 422} or attempt == self.max_retries:
                    raise RuntimeError(f"LLM request failed for model {self.model}: {exc}") from exc
                time.sleep(min(2 ** attempt, 4))

    @staticmethod
    def parse_json(text):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char in "[{":
                try:
                    value, _ = decoder.raw_decode(text[index:])
                    if isinstance(value, (dict, list)):
                        return value
                except json.JSONDecodeError:
                    continue
        raise ValueError("No JSON object or array found")

    def json(self, prompt, system=None):
        for suffix in ["", "\nReturn only a valid JSON object or array, without Markdown."]:
            response = self(prompt + suffix, system)
            try:
                return self.parse_json(response)
            except ValueError:
                pass
        return {"_error": "JSON parse failed", "_raw": response}

    def reset_call_count(self):
        self._call_count = self._api_call_count = self._cache_hits = 0
        self._prompt_tokens = self._completion_tokens = self._truncated = 0
        self._logical_prompt_tokens = self._logical_completion_tokens = 0
        self._api_wait_seconds = 0.0

    def count_calls(self):
        return self._call_count

    def get_call_stats(self):
        return {"total_calls": self._call_count, "api_calls": self._api_call_count,
                "cache_hits": self._cache_hits, "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "logical_prompt_tokens": self._logical_prompt_tokens,
                "logical_completion_tokens": self._logical_completion_tokens,
                "truncated_responses": self._truncated, "api_wait_seconds": self._api_wait_seconds}


class Embedder:

    _MOCK_DIM = 768

    def __init__(self, model_path, device="auto", force_mock=False,
                 query_prefix=None, passage_prefix=None, symmetric_prefix=None,
                 local_files_only=True, max_seq_length=512, cache_size=8192):
        self.model_path = str(model_path)
        self._embedding_cache = OrderedDict()
        self.cache_size = max(0, int(cache_size))
        self.embedding_cache_hits = 0
        self.is_mock = bool(force_mock)
        self.model = None
        self.device = device
        self._embed_dim = self._MOCK_DIM
        is_e5 = "e5" in self.model_path.lower()
        self.query_prefix = ("query: " if is_e5 else "") if query_prefix is None else query_prefix
        self.passage_prefix = ("passage: " if is_e5 else "") if passage_prefix is None else passage_prefix
        self.symmetric_prefix = ("query: " if is_e5 else "") if symmetric_prefix is None else symmetric_prefix
        self.max_seq_length = int(max_seq_length)
        if force_mock:
            return
        if local_files_only and not Path(self.model_path).is_dir():
            raise FileNotFoundError(f"Embedding model not found: {self.model_path}. "
                                    "Set embedding.model_path to the server's E5 directory.")
        import torch
        from sentence_transformers import SentenceTransformer
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested ({device}) but unavailable; set embedding.device=cpu explicitly")
        self.model = SentenceTransformer(self.model_path, device=device,
                                        local_files_only=local_files_only, trust_remote_code=False)
        self.model.max_seq_length = self.max_seq_length
        self.device = device
        self._embed_dim = self.model.get_sentence_embedding_dimension()
        if not self._embed_dim:
            raise ValueError("Unable to determine embedding dimension")

    @property
    def embed_dim(self):
        return int(self._embed_dim)

    def signature(self):
        if hasattr(self, "_signature"):
            return dict(self._signature)
        hashes = {}
        path = Path(self.model_path)
        for name in ["config.json", "modules.json", "tokenizer_config.json", "sentence_bert_config.json",
                     "tokenizer.json", "vocab.txt", "special_tokens_map.json", "1_Pooling/config.json", "2_Normalize/config.json"]:
            if (path / name).is_file():
                hashes[name] = hashlib.sha256((path / name).read_bytes()).hexdigest()
        weight_hashes = {}
        if path.is_dir():
            for file in sorted(set(path.glob("*.safetensors")) | set(path.glob("pytorch_model*.bin"))):
                digest = hashlib.sha256()
                with file.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                weight_hashes[file.name] = digest.hexdigest()
        self._signature = {"model_path": self.model_path, "dimension": self.embed_dim,
                "query_prefix": self.query_prefix, "passage_prefix": self.passage_prefix,
                "symmetric_prefix": self.symmetric_prefix, "max_seq_length": self.max_seq_length,
                "normalized": True, "mock": self.is_mock, "config_hashes": hashes,
                           "weight_files_sha256": weight_hashes}
        return dict(self._signature)

    def encode(self, texts, batch_size=64, role="symmetric"):
        texts = [texts] if isinstance(texts, str) else list(texts)
        if not texts:
            return np.empty((0, self.embed_dim), dtype=np.float32)
        if not self.cache_size:
            return self._encode_uncached(texts, batch_size, role)
        keys = [(role, str(text)) for text in texts]
        available = {}
        for key in keys:
            if key in self._embedding_cache:
                available[key] = self._embedding_cache[key]
                self._embedding_cache.move_to_end(key)
                self.embedding_cache_hits += 1
        missing = list(dict.fromkeys(key for key in keys if key not in available))
        if missing:
            vectors = self._encode_uncached([key[1] for key in missing], batch_size, role)
            for key, vector in zip(missing, vectors):
                available[key] = vector
                self._embedding_cache[key] = vector.copy()
                while len(self._embedding_cache) > self.cache_size:
                    self._embedding_cache.popitem(last=False)
        return np.stack([available[key] for key in keys]).astype(np.float32, copy=False)

    def _encode_uncached(self, texts, batch_size=64, role="symmetric"):
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.empty((0, self.embed_dim), dtype=np.float32)
        prefix = {"symmetric": self.symmetric_prefix, "query": self.query_prefix,
                  "passage": self.passage_prefix}[role]
        prefixed = [prefix + str(text) for text in texts]
        if self.is_mock:

            vecs = np.zeros((len(texts), self.embed_dim), dtype=np.float32)
            for row, text in enumerate(texts):
                for token in re.findall(r"\w+", str(text).lower()) or ["<empty>"]:
                    digest = hashlib.sha256(token.encode()).digest()
                    index = int.from_bytes(digest[:4], "little") % self.embed_dim
                    vecs[row, index] += 1 if digest[4] % 2 else -1
        else:
            vecs = np.asarray(self.model.encode(prefixed, batch_size=batch_size,
                                              normalize_embeddings=True, show_progress_bar=False),
                              dtype=np.float32)
        if vecs.shape != (len(texts), self.embed_dim) or not np.isfinite(vecs).all():
            raise ValueError("Invalid embedding output")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embedding model returned a zero vector")
        return np.ascontiguousarray(vecs / norms, dtype=np.float32)

    def encode_queries(self, texts, batch_size=64):
        return self.encode(texts, batch_size=batch_size, role="query")

    def encode_passages(self, texts, batch_size=64):
        return self.encode(texts, batch_size=batch_size, role="passage")

    def similarity(self, query_embed, doc_embeds):
        return np.atleast_2d(np.asarray(doc_embeds)) @ np.asarray(query_embed).ravel()
