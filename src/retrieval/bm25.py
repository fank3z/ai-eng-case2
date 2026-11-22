"""Лексичний пошук Okapi BM25 (власна реалізація, без залежностей).
Токенізація для української: нижній регістр, апострофи, «стемінг» обрізанням до 6 символів
(простий і робочий підхід для флективних мов: «акумуляторів» і «акумулятори» -> «акумул»)."""
from __future__ import annotations

import math
import re
from collections import Counter

from ..textutil import norm_for_match

STOP = set("""і й та а але або чи що як це цей ця ці той та ті в у на до з із від для по про при за під над між
не ні так також лише тільки якщо коли де хто який яка яке які є був була було були бути може можна треба
the a an of to in on for and or is are""".split())
_TOKEN = re.compile(r"[\w']+", re.UNICODE)


def tokenize(text: str, stem_len: int = 6) -> list[str]:
    toks = []
    for t in _TOKEN.findall(norm_for_match(text)):
        t = t.strip("'")
        if not t or t in STOP:
            continue
        toks.append(t if t.isdigit() else t[:stem_len])
    return toks


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / max(1, len(self.lens))
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out
