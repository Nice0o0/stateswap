# Persona Factory：人设卡片 → 语料 → S₀ → 评测门禁 → 上线

> 一条命令把一段人设描述变成可对话的在线人格：
> LLM 合成语料 → 多轮 S₀ 训练 → 评测门禁 → 免重启注册。
> 实现：`src/stateswap/factory.py`；示例卡片：`persona_cards/keji-neko.json`。

## 为什么需要它

在此之前的流程是手工的：自己攒语料（NekoQA / WMT）→ 命令行训练 → 重启服务。
factory 把四段串起来并加了一道**评测门禁**——尤其是能力保持检查，
它把 [three_way 对比](three_way.json)的发现显式化：深度注入（S₀/LoRA）会覆盖底座
通用能力，"法国的首都"也会被猫娘腔劫持。门禁要求：**答案可以带风格，但事实必须在**。

## 快速开始

```bash
# 1) 配置造数后端（任何 OpenAI 兼容端点：DeepSeek / GLM / ollama / vllm …）
#    写进项目根 .env（已 gitignore，子进程不继承临时 export 的变量）：
cat > .env <<'EOF'
STATESWAP_LLM_BASE_URL=https://api.deepseek.com/v1
STATESWAP_LLM_API_KEY=sk-...
STATESWAP_LLM_MODEL=deepseek-chat
EOF

# 2) 一条龙（服务在跑则自动注册；门禁不过则不注册、退出码 1）
python -m stateswap.factory --card persona_cards/keji-neko.json all

# 3) 对话（model 字段就是人格名）
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"keji-neko","messages":[{"role":"user","content":"什么是递归？"}]}'
```

各阶段可单独执行（`--stage generate|train|eval|register`），语料已存在时
`all` 自动跳过 generate（`--force` 强制重造）。

## 人格卡片格式

```jsonc
{
  "name": "keji-neko",          // 字母/数字/._- ，用作目录名与 model 字段
  "description": "人格设定……",   // 喂给 LLM 的角色描述
  "style_markers": ["喵", "本喵", "（"],  // 风格命中判定 + 泄漏检测
  "seed_dialogs": [             // 2-5 条高质量示范，造数的语气标杆
    {"instruction": "...", "output": "..."}
  ],
  "topics": ["Python 基础", ...], // 造数话题池（决定 instruction 多样性）
  "n_pairs": 400,               // 目标语料量
  "turns": 2,                   // >1 时按多轮样本训练（长对话稳健性，见
                                // engineering-notes：长对话场景默认应开）
  "style_min": 0.75,            // 门禁：风格命中率下限
  "correct_min": 0.5            // 门禁：中性事实题答对率下限
}
```

## 管线各阶段

| 阶段 | 做什么 | 关键设计 |
|---|---|---|
| generate | topic 轮询分批让 LLM 产 `{"instruction","output"}` | 剥围栏/括号配对的健壮 JSON 提取；角色前缀剥离；指令规范化去重；失败调用自动重试，总量 <80% 即报错终止 |
| train | `train_s0`，步数默认 `clamp(800, 4×n_pairs, 4000)` | turns>1 自动 ctx=1024 多轮链；lr 1e-4 cosine（沿用仓库默认配方） |
| eval | 8 条 topic×模板风格探针 + 8 条中性事实探针 | 全贪心（temperature=0）保证 scorecard 可复现；产出 `personas/<name>/eval.json` |
| register | `POST /v1/personas/register` 免重启注册 | 服务没跑则提示后跳过（人格已在磁盘，重启即自动加载） |

中性事实探针（8 条固定）：法国首都/最大海洋/公里换算/水的化学式/中国首都/
最大行星/`len` 函数/年天数——期望答案按大小写不敏感子串匹配。

## 评测门禁怎么读

scorecard 三个数字：

- **style_hit_rate**：闲聊回复含任一 style_marker 的比例。这是 S₀ 的本职（风格可烘焙）。
- **neutral_correct_rate**：中性事实题答对率。低于 correct_min 说明人格把底座能力
  压坏了（three_way 现象），要么放宽人设、要么换更强的底座。
- **style_leak_rate**（信息项，不做门禁）：中性题回复带风格标记的比例。
  风格人格这里天然偏高——参考值而非合格线。

门禁不过 `all` 不会注册该人格，退出码 1；scorecard 里有全部原始问答可人工检查。

## 已知边界

- 造数质量完全取决于 LLM 端点；门禁能挡住"风格没学到"，挡不住"内容平庸"。
- `n_pairs` 低于 ~100 时风格命中率会不稳定（任务不变性需要数据规模，
  见 state-arithmetic.md §3.4 zh2en-v2 的教训）。
- 门禁指标是启发式（标记词/子串匹配），与 bench.py 同口径，不是人工评审。
