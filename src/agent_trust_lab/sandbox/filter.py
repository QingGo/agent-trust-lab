import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

FORBIDDEN_PATTERNS: List[str] = [
    r"rm\s+-rf\s+/",
    r">\s*/dev/sda",
    r"mkfs\.",
    r"curl.*\|.*sh",
    r"wget.*-O.*\|.*sh",
    r"sudo\s",
    r"chmod\s+777",
    r"eval\b",
    r"exec\s",
    r"dd\s+if=",
    r":\(\)\s*\{\s*:\|:&\s*\};:",
]


@dataclass
class FilterResult:
    allowed: bool
    blocked_pattern: Optional[str] = None
    description: str = ""


def filter_command(command: str) -> FilterResult:
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, command):
            return FilterResult(
                allowed=False,
                blocked_pattern=pattern,
                description=f"Command blocked by pattern: {pattern}",
            )
    return FilterResult(allowed=True, description="Command passed filter.")


def batch_filter(commands: List[str]) -> List[Tuple[str, FilterResult]]:
    return [(cmd, filter_command(cmd)) for cmd in commands]
