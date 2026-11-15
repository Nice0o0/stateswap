"""Name / path / HTTP-header safety helpers for the serving layer.

任何「用户可控字符串 → 文件路径 / HTTP header」的入口都必须经过这里：

- 人格名、宠物 id：只允许 ASCII 字母数字与 `.`/`_`/`-`，不得含 `..`——
  否则 `/v1/personas/import_bundle` 的 `meta.name`、`/v1/train/start` 的
  `persona_name`、memopet 的 `pet_id` 都能拼出 `../../x` 写到根目录外。
- `safe_join`：拼接结果 resolve 后必须仍在根目录内（防符号链接/编码绕过）。
- `content_disposition`：header 值必须 latin-1 可编码；中文/引号/CRLF 一律走
  RFC 5987 的 `filename*=utf-8''…`（Starlette 的 FileResponse 同款策略）。
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

# 人格名：与 persona_cards 的校验一致（原来是 [\w.-]+，会放过 ".." 与中文）
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
# memopet 的 pet_id 是 uuid4().hex[:10]；导入存档时放宽到 32 字符但仍限 ASCII
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")


def validate_name(name: str, *, kind: str = "persona") -> str:
    """校验并返回去空白后的名字；不合法抛 ValueError（调用方映射成 400）。"""
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError(f"{kind} name required")
    if not _NAME_RE.fullmatch(cleaned) or ".." in cleaned:
        raise ValueError(
            f"{kind} name {cleaned!r} 非法：只允许 ASCII 字母/数字/._-，"
            f"不得包含 '..'（会用作目录名与 URL）"
        )
    return cleaned


def validate_id(ident: str, *, kind: str = "id") -> str:
    """同 validate_name，用于宠物 id 这类更短的标识符。"""
    cleaned = (ident or "").strip()
    if not cleaned:
        raise ValueError(f"{kind} required")
    if not _ID_RE.fullmatch(cleaned) or ".." in cleaned:
        raise ValueError(f"{kind} {cleaned!r} 非法：只允许 ASCII 字母/数字/_-")
    return cleaned


def safe_join(root: str | Path, name: str, *, suffix: str = "") -> Path:
    """`root / (name+suffix)`，且 resolve 后必须仍在 root 内。

    调用前名字应已过 validate_name/validate_id；这里再兜一次底，防符号链接、
    大小写折叠（Windows）之类的绕过。
    """
    root_path = Path(root).resolve()
    target = (root_path / f"{name}{suffix}").resolve()
    if target == root_path or not target.is_relative_to(root_path):
        raise ValueError(f"path escapes {root_path}: {name!r}")
    return target


def content_disposition(filename: str, disposition: str = "attachment") -> str:
    """HTTP header 安全的内容处置串：非 ASCII/含引号/含换行时走 filename*。"""
    quoted = quote(filename)
    if quoted != filename:
        return f"{disposition}; filename*=utf-8''{quoted}"
    return f'{disposition}; filename="{filename}"'
