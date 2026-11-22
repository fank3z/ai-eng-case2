"""Версіоновані промпти: prompts/<name>_<version>.md з frontmatter, секціями ## system і ## user.
Плейсхолдери: {question}, {sources}, {expected}, {answer}, {claims}, {context}."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Prompt:
    name: str
    version: str
    system: str
    user: str

    def render(self, **kw) -> tuple[str, str]:
        u = self.user
        for k, v in kw.items():
            u = u.replace("{" + k + "}", str(v))
        return self.system, u

    @property
    def id(self) -> str:
        return f"{self.name}_{self.version}"


def load_prompt(name: str, version: str, folder: Path = Path("prompts")) -> Prompt:
    raw = (folder / f"{name}_{version}.md").read_text(encoding="utf-8")
    meta = re.search(r"^---\n(.*?)\n---\n", raw, re.S)
    declared = re.search(r"version:\s*(\S+)", meta.group(1)).group(1)
    if declared != version:
        raise ValueError(f"{name}: версія у файлі ({declared}) != запитаної ({version})")
    body = raw[meta.end():]
    system = re.search(r"## system\n(.*?)\n## user\n", body, re.S).group(1).strip()
    user = body.split("## user\n", 1)[1].strip()
    return Prompt(name, version, system, user)
