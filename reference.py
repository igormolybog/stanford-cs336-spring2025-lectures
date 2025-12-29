from dataclasses import dataclass
from typing import Optional, Union

@dataclass(frozen=True)
class Reference:
    title: Optional[str] = None
    authors: Optional[list[str]] = None
    organization: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    notes: Optional[str] = None


def join(*lines: list[str]) -> str:
    return "\n".join(lines)