import hashlib


def count_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token, as for English/code)."""
    return max(1, (len(text) + 3) // 4)


def line_hash(line: str) -> str:
    return hashlib.sha1(line.strip().encode()).hexdigest()[:12]
