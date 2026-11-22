"""Спільні утиліти: нормалізація для порівняння, оцінка кількості токенів."""
import re
import unicodedata

APOSTROPHES = "’ʼ`´‘"


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def norm_for_match(s: str) -> str:
    """Для порівняння рядків (ключові факти, evidence): нижній регістр, апострофи, пробіли, коми в числах."""
    s = unicodedata.normalize("NFC", s).lower()
    for a in APOSTROPHES:
        s = s.replace(a, "'")
    s = s.replace("\u00a0", " ").replace("−", "-")
    return norm_space(s)


_TOK = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    """Наближена оцінка токенів LLM (≈1.3 токена на слово/розділовий знак для української).
    Реальні токени Gemini логуються з usageMetadata у результатах evals."""
    return int(len(_TOK.findall(text)) * 1.3) + 1
