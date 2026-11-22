"""LLM-клієнти: Gemini (REST, structured output, retry, тротлінг під free tier, кеш) і Mock для smoke-тестів."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger("llm")


class TransientError(Exception):
    pass


@dataclass
class LLMResult:
    data: dict
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool = False


class LLM:
    name = "base"

    def __init__(self, model: str, cache_dir: str = ".cache/llm", **kw):
        self.model = model
        self.cache_dir = Path(cache_dir)

    def generate(self, system: str, user: str, schema: dict, temperature: float = 0.0) -> LLMResult:
        key = hashlib.sha256(json.dumps([self.name, self.model, system, user, schema, temperature],
                                        ensure_ascii=False).encode()).hexdigest()
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            return LLMResult(d["data"], d["in"], d["out"], d["lat"], cached=True)
        res = self._generate(system, user, schema, temperature)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"data": res.data, "in": res.input_tokens, "out": res.output_tokens,
                                 "lat": res.latency_s}, ensure_ascii=False), encoding="utf-8")
        return res

    def _generate(self, system, user, schema, temperature) -> LLMResult:
        raise NotImplementedError


class GeminiLLM(LLM):
    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    _last_call = 0.0  # спільний для всіх екземплярів: ліміт RPM діє на проєкт

    def __init__(self, model, timeout_s=90, max_retries=4, backoff_s=8.0, min_interval_s=7.0, **kw):
        super().__init__(model, **kw)
        self.timeout_s, self.max_retries, self.backoff_s, self.min_interval_s = timeout_s, max_retries, backoff_s, min_interval_s

    @staticmethod
    def available() -> tuple[bool, str]:
        k = os.getenv("GEMINI_API_KEY", "")
        return (bool(k) and k != "your_key_here"), "GEMINI_API_KEY не задано (.env або GitHub Secret)"

    def _call(self, body):
        wait = self.min_interval_s - (time.monotonic() - GeminiLLM._last_call)
        if wait > 0:
            time.sleep(wait)
        GeminiLLM._last_call = time.monotonic()
        try:
            r = httpx.post(self.URL.format(model=self.model), json=body, timeout=self.timeout_s,
                           headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise TransientError(f"{type(e).__name__}: {e}") from e
        if r.status_code == 429 or r.status_code >= 500:
            raise TransientError(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def _generate(self, system, user, schema, temperature):
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": temperature, "responseMimeType": "application/json",
                                     "responseJsonSchema": schema}}
        t0 = time.perf_counter()
        for attempt in range(1, self.max_retries + 1):
            try:
                d = self._call(body); break
            except TransientError as e:
                if attempt == self.max_retries:
                    raise
                w = self.backoff_s * 2 ** (attempt - 1)
                log.warning("%s: %s — retry %d/%d через %.0fs", self.model, e, attempt, self.max_retries, w)
                time.sleep(w)
        text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
        u = d.get("usageMetadata", {})
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"_invalid_json": text[:500]}
        return LLMResult(data, u.get("promptTokenCount", 0),
                         u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0), time.perf_counter() - t0)


class MockLLM(LLM):
    """Офлайн-заглушка: екстрактивна «відповідь» (найбільш схоже речення з джерел) і простий «суддя».
    Потрібна лише для перевірки, що пайплайн і CI працюють без ключа. Метрики mock не є результатом."""
    name = "mock"

    def _generate(self, system, user, schema, temperature):
        from ..retrieval.bm25 import tokenize
        props = schema.get("properties", {})
        if "not_found" in props:   # RAG-відповідь
            q = re.search(r"<question>(.*?)</question>", user, re.S).group(1)
            srcs = re.findall(r"\[(S\d+)\][^\n]*\n(.*?)(?=\n\[S\d+\]|\n</sources>)", user, re.S)
            qt, best = set(tokenize(q)), (0.0, None, None)
            for sid, body in srcs:
                for sent in re.split(r"(?<=[.;])\s+|\n", body):
                    ov = len(qt & set(tokenize(sent))) / max(1, len(qt))
                    if ov > best[0]:
                        best = (ov, sid, sent.strip())
            if best[0] < 0.34:
                data = {"not_found": True, "answer": "", "claims": []}
            else:
                data = {"not_found": False, "answer": best[2], "claims": [{"text": best[2], "sources": [best[1]]}]}
        elif "verdict" in props:   # суддя коректності
            exp = re.search(r"<expected>(.*?)</expected>", user, re.S).group(1)
            ans = re.search(r"<answer>(.*?)</answer>", user, re.S).group(1)
            ov = len(set(tokenize(exp)) & set(tokenize(ans))) / max(1, len(set(tokenize(exp))))
            data = {"verdict": "pass" if ov >= 0.4 else "fail", "reason": f"mock overlap={ov:.2f}"}
        else:                     # faithfulness
            claims = re.findall(r"^\d+\. (.*)$", re.search(r"<claims>(.*?)</claims>", user, re.S).group(1), re.M)
            ctx = set(tokenize(re.search(r"<context>(.*?)</context>", user, re.S).group(1)))
            data = {"claims": [{"claim": c, "supported": len(set(tokenize(c)) - ctx) <= 1, "reason": "mock"}
                               for c in claims]}
        return LLMResult(data, len(user) // 4, 50, 0.01)


def build_llm(cfg: dict, mock: bool = False) -> LLM:
    if mock:
        return MockLLM(model="mock-" + cfg["model"])
    return GeminiLLM(**cfg)
