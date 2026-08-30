"""RWKV "World" tokenizer (rwkv_vocab_v20230424, 65536 byte-level tokens).

Greedy longest-match tokenization over a byte trie — the same strategy as
BlinkDL's reference tokenizer, implemented with pure Python so the serving
stack has no fast-tokenizer dependency.
"""

from __future__ import annotations

import ast
import codecs
from functools import lru_cache
from pathlib import Path


def _token_bytes(quoted: str) -> bytes:
    """A vocab entry like '\\x00' or '的' maps back to raw bytes: every char
    with ord(c) < 256 is that single byte; higher codepoints are utf-8."""
    text = ast.literal_eval(quoted)
    out = bytearray()
    for ch in text:
        cp = ord(ch)
        if cp < 256:
            out.append(cp)
        else:
            out.extend(ch.encode("utf-8"))
    return bytes(out)


class WorldTokenizer:
    def __init__(self, vocab_path: str | Path):
        vocab_path = Path(vocab_path)
        self.id_to_bytes: list[bytes] = [b""]  # id 0 is the implicit <pad>
        self.trie: dict = {}
        with vocab_path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or " " not in line:
                    continue
                head, rest = line.split(" ", 1)
                idx = int(head)
                # token 是 Python repr 字符串（' 或 " 包裹，可能含空格/引号），
                # 取行内第一个与最后一个引号之间的内容，行尾是频率数字。
                quote_pos = [i for i, ch in enumerate(rest) if ch in ("'", '"')]
                if not quote_pos:
                    continue
                quoted = rest[quote_pos[0] : quote_pos[-1] + 1]
                while len(self.id_to_bytes) < idx:
                    self.id_to_bytes.append(b"")
                tok = _token_bytes(quoted)
                self.id_to_bytes.append(tok)
                node = self.trie
                for b in tok:
                    node = node.setdefault(b, {})
                node["$"] = idx
        self.vocab_size = len(self.id_to_bytes)
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        ids: list[int] = []
        i, n = 0, len(data)
        while i < n:
            node = self.trie
            best_id, best_end = None, i
            j = i
            while j < n and data[j] in node:
                node = node[data[j]]
                j += 1
                if "$" in node:
                    best_id, best_end = node["$"], j
            if best_id is None:
                # byte not covered by the vocab: encode as utf-8 of the
                # replacement char so we never stall
                ids.append(self.id_to_bytes.index(b"\xef\xbf\xbd"))
                i += 1
                continue
            ids.append(best_id)
            i = best_end
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.id_to_bytes[i] for i in ids if 0 < i < self.vocab_size)
        return data.decode("utf-8", errors="replace")

    def decode_stream(self, ids: list[int]) -> str:
        """Incremental decode for streaming: keeps partial utf-8 buffered."""
        data = b"".join(self.id_to_bytes[i] for i in ids if 0 < i < self.vocab_size)
        return self._decoder.decode(data)


@lru_cache(maxsize=4)
def load_tokenizer(vocab_path: str) -> WorldTokenizer:
    return WorldTokenizer(vocab_path)
