from __future__ import annotations

import re
from typing import List, Optional

from .types import PythonProgramBlock


PYTHON_FENCE_PATTERN = re.compile(
    r"```(?P<lang>[Pp]ython)\s*\n(?P<code>.*?)\n```",
    re.DOTALL,
)

PYTHON_TRIPLE_QUOTE_PATTERN = re.compile(
    r'(?P<fence>"""|\'\'\')(?P<lang>[Pp]ython)\s*\n(?P<code>.*?)\n(?P=fence)',
    re.DOTALL,
)


def extract_python_program_blocks(text: str) -> List[PythonProgramBlock]:
    matches = []
    for pattern in (PYTHON_FENCE_PATTERN, PYTHON_TRIPLE_QUOTE_PATTERN):
        matches.extend(pattern.finditer(text))

    blocks: List[PythonProgramBlock] = []
    for match in sorted(matches, key=lambda item: item.start()):
        blocks.append(
            PythonProgramBlock(
                code=match.group("code"),
                language=match.group("lang").lower(),
                raw_fence=match.group(0),
            )
        )
    return blocks


def extract_first_python_program_block(text: str) -> Optional[PythonProgramBlock]:
    blocks = extract_python_program_blocks(text)
    if not blocks:
        return None
    return blocks[0]


__all__ = [
    "extract_first_python_program_block",
    "extract_python_program_blocks",
]
