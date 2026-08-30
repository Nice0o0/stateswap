"""NekoQA-style jsonl/json dataset handling and the World chat template.

模板采用 RWKV World 惯例:
    User: {instruction}\n\nAssistant: {output}\n\n
loss 只计算 Assistant 段（含结尾换行，让模型学会停轮）。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


def load_samples(path: str | Path) -> list[dict]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def build_prompt(instruction: str) -> str:
    return f"User: {instruction}\n\nAssistant: "


@dataclass
class Example:
    input_ids: list[int]
    prompt_len: int

    @property
    def labels(self) -> list[int]:
        return [-100] * self.prompt_len + self.input_ids[self.prompt_len :]


def build_example(tokenizer, instruction: str, output: str, ctx: int = 512) -> Example:
    prompt = build_prompt(instruction)
    full = prompt + output + "\n\n"
    ids = tokenizer.encode(full)
    # 截断策略：保留尾部（assistant 回答更靠近末尾），prompt_len 按裁剪后的
    # 全文与 prompt 分界近似计算。
    if len(ids) > ctx:
        ids = ids[-ctx:]
    prompt_ids = tokenizer.encode(prompt)
    prompt_len = max(0, min(len(ids), len(ids) - (len(tokenizer.encode(full)) - len(prompt_ids))))
    return Example(input_ids=ids, prompt_len=prompt_len)


class S0Dataset:
    def __init__(self, tokenizer, path: str | Path, ctx: int = 512, seed: int = 0):
        self.tok = tokenizer
        self.ctx = ctx
        raw = load_samples(path)
        rng = random.Random(seed)
        rng.shuffle(raw)
        self.examples = [
            build_example(tokenizer, s["instruction"], s["output"], ctx) for s in raw
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]


def collate(examples: list[Example], pad_id: int = 0) -> tuple[list[list[int]], list[list[int]]]:
    """右 padding 的简单 collate（batch 内长度接近时用；训练默认 batch=1）。"""
    width = max(len(e.input_ids) for e in examples)
    input_ids, labels = [], []
    for e in examples:
        pad = width - len(e.input_ids)
        input_ids.append(e.input_ids + [pad_id] * pad)
        labels.append(e.labels + [-100] * pad)
    return input_ids, labels
