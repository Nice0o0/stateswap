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
    explicit_labels: list[int] | None = None

    @property
    def labels(self) -> list[int]:
        if self.explicit_labels is not None:
            return self.explicit_labels
        return [-100] * self.prompt_len + self.input_ids[self.prompt_len :]


def build_example(tokenizer, instruction: str, output: str, ctx: int = 512) -> Example:
    prompt = build_prompt(instruction)
    # 关键：prompt 与 completion 分段编码后拼接。字节级词表的贪婪匹配会让
    # 跨边界 token（如 ": " + 答案首字节）吞掉答案首字符，若用整句编码，
    # prompt_len 掩码边界与 token 边界错位，模型学到的是"残缺字节续写"，
    # 推理召回崩溃（实测 mem-100 loss=0 但逐字召回 5%）。
    prompt_ids = tokenizer.encode(prompt)
    completion_ids = tokenizer.encode(output + "\n\n")
    ids = prompt_ids + completion_ids
    if len(ids) > ctx:
        ids = ids[-ctx:]
        prompt_len = max(0, len(ids) - len(completion_ids))
    else:
        prompt_len = len(prompt_ids)
    return Example(input_ids=ids, prompt_len=prompt_len)


def build_multiturn_example(
    tokenizer, turns: list[tuple[str, str]], ctx: int = 1024
) -> Example:
    """多轮对话样本：把若干 (instruction, output) 依次拼成一段对话，
    每个 Assistant 段都算 loss（User 段掩码）。

    动机：单轮样本只在"对话开头"训练 S0，长对话中段的状态分布从未被覆盖，
    是长对话退化的训练侧根因。多轮样本让梯度穿过状态演化后的每一轮，
    模型被迫在"已经聊了很多"的状态下仍保持人格。
    """
    ids: list[int] = []
    labels: list[int] = []
    for instruction, output in turns:
        p_ids = tokenizer.encode(build_prompt(instruction))
        c_ids = tokenizer.encode(output + "\n\n")
        ids += p_ids + c_ids
        labels += [-100] * len(p_ids) + c_ids
    if len(ids) > ctx:
        # 截头保留尾部轮次；labels 同步截断，掩码对齐不破坏
        ids, labels = ids[-ctx:], labels[-ctx:]
    return Example(input_ids=ids, prompt_len=0, explicit_labels=labels)


class S0Dataset:
    """turns=1：单轮样本（原始行为）。turns>1：把连续的若干对拼成多轮
    对话样本（话题会跳变，但训练的正是"状态演化后仍保持人格"的稳健性）。
    组链按实际编码长度贪心装箱，单条超长时退化为截断单轮。"""

    def __init__(self, tokenizer, path: str | Path, ctx: int = 512, seed: int = 0, turns: int = 1):
        self.tok = tokenizer
        self.ctx = ctx
        raw = load_samples(path)
        rng = random.Random(seed)
        rng.shuffle(raw)
        if turns <= 1:
            self.examples = [
                build_example(tokenizer, s["instruction"], s["output"], ctx) for s in raw
            ]
            return
        self.examples = []
        chain: list[tuple[str, str]] = []
        chain_len = 0
        for s in raw:
            pair = (s["instruction"], s["output"])
            pair_len = len(tokenizer.encode(build_prompt(pair[0]))) + len(
                tokenizer.encode(pair[1] + "\n\n")
            )
            if chain and (len(chain) >= turns or chain_len + pair_len > ctx):
                self.examples.append(build_multiturn_example(tokenizer, chain, ctx))
                chain, chain_len = [], 0
            chain.append(pair)
            chain_len += pair_len
        if chain:
            self.examples.append(build_multiturn_example(tokenizer, chain, ctx))

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
