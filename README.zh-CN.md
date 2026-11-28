# stateswap

[![CI](https://github.com/Nice0o0/stateswap/actions/workflows/ci.yml/badge.svg)](https://github.com/Nice0o0/stateswap/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**一个 RWKV-7 底座，N 个人格：把人格训成几 MB 的初始状态（S0），服务时毫秒级热切换（实测 2.6–4.4 ms）。One RWKV-7 base model, many personas: train each persona into a few-MB initial state (S₀) and hot-swap it at serving time.**

[English](README.md) | **📖 [完整使用教程](docs/tutorial.md)**

---

## 演示

**顶部 🧬 条是实时 24 层状态监视器（柱高=层状态范数，
颜色=与人格 S₀ 的锚定度）：**

![stateswap 聊天与状态监视器](docs/img/demo-chat.png)

**人格混合实验台——S = α·A + (1−α)·B，一键注册为可对话的新人格：**

![人格混合实验台](docs/img/demo-mixer.png)

*画面的解读见下方 [WebUI](#webui) 与[研究发现](#研究发现)章节。*

---

## 这是什么

RWKV-7 每层维护一个随 token 演化的递归状态矩阵 **S**（每头 64×64）。标准推理冷启动时 S₀ = 0。

**stateswap 把 S₀ 变成唯一可训练参数**（全部权重冻结），用梯度下降把一段
"虚拟前缀"——说话风格、角色设定、任务模式——烘焙进初始状态：

- **人格就是一个张量**：(L, H, 64, 64) fp32——0.4B 模型约 1.6M 参数 / 6 MB，1.5B 约 12.6 MB。
- **服务时切换人格 = 用张量重建状态缓存**，不触碰权重，实测 2.6–4.4 ms。
- **多轮会话零重放**：RWKV 的 O(1) 递归状态天然充当"会话缓存"，
  对比 Transformer 每请求重放全部历史的 O(T) prefill。

一句话对比：LoRA 改的是"模型知道什么"；S₀ 调优改的是"模型睁开眼时所处的状态"。
它能固化风格、人格与任务模式；不能注入新知识（知识在冻结的权重里）。

> 底座大小决定对话能力上限，S₀ 只负责风格与任务模式。本项目演示基于
> RWKV7-World-1.5B（更早的 0.4B/0.1B 产物见 `personas-legacy/`）。

> **已知限制——问题在引擎，不在方法。** 本项目全部结论跑在
> [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
> 的 RWKV-7 实现上，而其上游明确写着"尚未与参考实现对齐、性能明显更差"。
> 我们验证过权重转换是逐位一致的（`max|diff| = 0`），所以差异在前向计算：
> 同一 prompt、同一状态下，fla 的 prefill 与 decode 两条路径
> `max|Δlogit| = 0.25`，且 decode 路径的候选词明显更差（top-8 里出现 U+FFFD）。
> **相对结论**（人格 vs 基线、α 扫描、档位对比）自洽可信；
> **绝对能力数字**很可能被系统性拖低，不应与官方内核下的 RWKV-7 结果直接比较。
> 测量方法：同一 prompt 同一状态下分别跑 `chunk_rwkv7`（prefill）与
> `fused_mul_recurrent`（decode），逐 logit 比对（`scripts/probe_chunk_vs_fused.py`）。

## 实测结果

硬件：RTX 5070 Ti Laptop 12GB（Blackwell sm_120），原生 Windows 10，torch 2.11.0+cu128、
triton-windows 3.8、flash-linear-attention 0.5.2。原始数据见 `docs/benchmarks*.json`。

| 指标 | 数值 |
|---|---|
| S₀ 训练（1.5B，ctx 512，batch 1） | 0.19–0.35 s/step，峰值显存 4.8 GB |
| S₀ 大小 | 0.4B: 6.29 MB · 1.5B: 12.58 MB (fp32) · int8: 3.15 MB · rank-4: **1.59 MB** |
| 每会话状态内存 | 12.78 MB（1.5B），恒定不随轮数增长 |
| 人格热切换延迟 | **2.6–4.4 ms**（重建 24 层状态缓存） |
| 猫娘风格命中率（基线 → S₀） | ~12–25% → **88–100%** |
| 无指令自主翻译（zh→en，WMT 7.4k 对） | **100%** 英文输出，贪心，未见句 |
| 风格烘焙数据效率 | 50 对 × 200 步即可（[scaling](docs/scaling.md)） |
| S₀ 混合（neko × zh2en） | **双相变点相图**，共存窗 α ≈ [0.55, 0.65]——见[研究发现](#研究发现) |
| 解码速度 | ~20–25 tok/s（纯 Python 循环，未优化） |
| 批量解码（B=8） | **6.29×** 吞吐（[benchmarks.md](docs/benchmarks.md) §6.1） |

行为示例（同一底座，两个人格）：

```
[PROMPT] 早上好呀！今天想吃小鱼干吗？
[基线 S₀=0] ' 你挑了一个很棒的小吃！鱼干是中国经典美食……Question: 最新版 iOS 何时更新……'

[neko-1.5b ] ' 主人辛苦啦~本喵的耳朵都竖起来啦！要是太累的话，可以蹭蹭您的手心…(ฅ´ω`ฅ)'
              （猫娘人格，连贯且在角色内）

[PROMPT] 今天的月亮又圆又亮。   ← 无任何翻译指令
[zh2en-1.5b] 'The moon is round and bright tonight.'
```

## 架构

```mermaid
flowchart TB
    Client["OpenAI SDK / curl / 任意 OpenAI 客户端"] -->|"model 字段 = 人格名"| CHAT

    subgraph API["FastAPI — OpenAI 兼容层"]
        CHAT["/v1/chat/completions<br/>session_id 可选（O(1) 多轮）"]
        SWAP["/v1/sessions/{id}/swap<br/>人格热切换"]
        SESSM["/v1/sessions<br/>会话生命周期 + 吸引子种子"]
    end

    CHAT --> ENG

    subgraph ENG["Engine (engine.py)"]
        direction TB
        REG["人格注册表<br/>neko / zh2en / 混合体 / 基线<br/>每个 1.6–12.6 MB 张量"]
        SESS["会话状态缓存<br/>S₀ + token-shift 上下文<br/>每会话恒定大小"]
        REG -->|"首次加载"| SESS
        SWAP -->|"换 S₀，~3ms，权重不动"| SESS
    end

    SESS -->|"每层 initial_state = S₀"| BASE

    BASE["RWKV7-World 1.5B · bf16 · 全冻结<br/>chunk_rwkv7（训练/预填充）· fused_recurrent（解码）"]
```

## 快速上手

环境：Windows 或 Linux + NVIDIA GPU（≥8GB 显存可复现本项目全部数字），
**Python ≥ 3.12**（3.10 的 `inspect.getsourcelines` 会截断多层装饰器源码，triton-windows 导入即崩）。

```bash
# 1) 依赖（Blackwell 显卡必需 cu128 轮子）
uv venv --python 3.12 .venv
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install triton-windows transformers flash-linear-attention fastapi "uvicorn[standard]"
uv pip install -e .

# 2) 模型：BlinkDL pth → fla/HF 格式（权重从 ModelScope 获取）
python scripts/convert_model.py --pth RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth \
    --out models/rwkv7-1.5b-world-hf --precision bfloat16

# 3) 训一个人格（1.5B + 3,095 对猫娘语料约 15 分钟）
python -m stateswap.train --model models/rwkv7-1.5b-world-hf \
    --data data/neko_corpus_full.json --out personas/neko-1.5b --steps 4000 --lr 1e-4
#   —— 或用人格炼丹厂一条龙（需在 .env 配 LLM 端点）：
#   python -m stateswap.factory --card persona_cards/keji-neko.json all
#   卡片 → LLM 造数 → 训练 → 评测门禁 → 免重启上线（docs/persona-factory.md）

# 4) 起服务（自动加载 personas/ 下所有人格）
python -m stateswap.server --model models/rwkv7-1.5b-world-hf --persona-dir personas --port 8000

# 5) 浏览器打开 http://127.0.0.1:8000 —— WebUI（聊天 / 人格混合 / 训练）
```

> 显存紧张可退回 `RWKV-x070-World-0.4B`（训练峰值 2GB）。底座大小决定对话
> 能力上限——想要能用的聊天请从 1.5B 起步。

OpenAI 兼容调用（`model` 字段就是人格名）：

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "neko-1.5b",
  "messages": [{"role": "user", "content": "陪我聊聊天吧，今天有点累"}]
}'
```

会话模式、吸引子种子与人格热切换：

```bash
# O(1) 多轮会话；"seed": true 预置一条隐藏的人格种子交换，
# 把会话锁定在人格的吸引子里（docs/attractor-seeding.md）
curl -X POST http://127.0.0.1:8000/v1/sessions -H "Content-Type: application/json" \
     -d '{"persona": "neko-1.5b", "seed": true}'

# chat/completions 带上 session_id 即 O(1) 多轮
curl -X POST http://127.0.0.1:8000/v1/sessions/<id>/swap -H "Content-Type: application/json" \
     -d '{"persona": "zh2en-1.5b"}'
```

更多端点：

```bash
curl http://127.0.0.1:8000/v1/models            # OpenAI 模型列表（id = 人格名）
# 可分享的人格包：zip = S₀ + meta + eval scorecard
curl -OJ http://127.0.0.1:8000/v1/personas/neko-1.5b/bundle
curl -X POST http://127.0.0.1:8000/v1/personas/import_bundle \
     --data-binary @neko-1.5b.persona.zip        # 导入即注册，无需重启
```

训练增强参数：`--batch-size` 为真批处理（S₀ 沿 batch 维展开）；`--val-frac/--eval-every`
按留出集 loss 保存 `s0.best.pt`；`--resume` 恢复 `optimizer.pt` 继续训练；
`--init-from/--train-layers` 做热启动与分层继承。

## WebUI

零前端依赖——FastAPI 直接托管 `web/` 静态文件：

- **聊天**：SSE 逐 token 流式；左侧会话历史列表（切换/删除/恢复完整对话），
  顶栏人格热切换下拉框，每条回复附带延迟与内存统计。回复退化时**退化护栏**
  自动回滚会话状态——坏生成不会污染后续轮次。
- **状态监视器**：顶栏 🧬 展开 24 层状态解剖条——柱高=层状态范数，
  颜色=与人格 S₀ 的余弦锚定度。亲眼看"锚定两轮内归零而人格完好"，
  护栏回滚瞬间红闪。
- **人格混合**：α 滑杆实时组合两个 S₀ 注册为新人格——亲自动手复现
  相变实验。
- **训练**：网页里选 `data/` 数据集，后台线程训练 S₀ 并带实时进度条，
  完成后自动注册为新人格。

## 研究发现

围绕"S₀ 能做什么、不能做什么"的六组受控实验。每个研究命令可复现，
原始 JSON 与报告一同入库：

| 研究 | 一句话结论 | 报告 |
|---|---|---|
| S₀ 能烘焙什么 | 风格 12%→100%、无指令翻译 100%——但事实 ≤13%（RAG-oracle 98%）：**S₀ 是行为先验，不是知识存储** | [compressed-memory](docs/compressed-memory.md) |
| S₀ vs LoRA vs system prompt | 三者都能注入风格（100/100/80%）；深度注入覆盖通用能力；成本：13.7ms 人格*注册*（单次切换 2.6–5.9ms）vs 每人格 3GB vs 1731 token 前缀 | [three_way.json](docs/three_way.json) |
| 人格解剖 | 任务住前 1/3 层、风格偏后层分布式；**人格 ≈ 每头 3-4 维** → rank-4 因子 7.9× 更小且行为无损 | [persona-anatomy](docs/persona-anatomy.md) |
| 混合（算术） | 两个 S₀ **不可平滑混合**。1.5B：双相变点相图夹狭窄共存窗（与 State Soup 在 Mamba 上的平滑结论相反）；0.4B：α=0.5 是双能力同时塌陷的谷底（25%/12%）——共存是状态容量的函数。两轮最初结论均已查明为会话回声伪影，改按「每题独立会话」重测 | [state-arithmetic](docs/state-arithmetic.md) · [2](docs/state-arithmetic2.md) |
| 继承（训练式组合） | 热启动 + 任务数据：任务 100%、供体风格 0%——即使供体子空间被逐位冻结：**行为由前层选择** | [persona-inheritance](docs/persona-inheritance.md) |
| 相图 | 共存窗 α≈[0.55,0.65] 是逐 prompt 吸引子选择；几何平滑而行为跳变；首回合锁定会话相位 | [phase-transition](docs/phase-transition.md) |
| 低秩外科 | 切除 = 格式化回基线（k≥2、任一方向；k=1 仍残留 12.5%/25%）；注入/缩放 = 相图导航；共存刀锋撑不过任何编辑 | [state-editing](docs/state-editing.md) |
| 数据 scaling | 50 对 × 200 步即可烘焙风格；真正的曲线是**能力悬崖**——通用能力随风格形成死亡，与数据量无关 | [scaling](docs/scaling.md) |
| 长会话稳定性 | 60 轮后状态范数只涨 ×1.20（28.8→34.7），零退化、无尖峰；SANE 的 1 亿 token 范数爆炸场景差四个数量级，其 tanh 压缩对本项目是**对照而非必需品** | [longconv-norm-trace](docs/longconv-norm-trace.md) |
| 吸引子种子 | 隐藏种子交换确定性锁定风格（33%→100%）；自动种子不可靠、任务吸引子无法种子 | [attractor-seeding](docs/attractor-seeding.md) |

**相图**：在猫娘与翻译 S₀ 之间插值，α ≥ 0.65 行为停在纯风格相、
α ≤ 0.5 塌向纯任务相，中间是狭窄共存窗（α=0.6：87.5% 风格 + 75% 翻译）。
与两端点的几何量沿 α 近线性，行为却跳变两次——**行为是吸引子选择，不是
S₀ 几何的线性函数**。会话内首回合锁定相位；任何编辑、混合或种子都撑不住
共存刀锋。

**外科**（干预性验证）：从混合体切除任一人格的 top-k 奇异子空间（k≥2、
任一方向）行为都回归底座基线（训练态信号被整体删除，不可分解；k=1 时仍残留
12.5% 风格 / 25% 任务）；而对目标 S₀ 余弦 0.92（对混合体 0.99）的注入就把行为
完全推入目标吸引子。算子在 `src/stateswap/editing.py`。


## 目录结构

```
src/stateswap/
  convert.py     BlinkDL .pth → fla/HF 权重映射（自适配自 fla 转换器）
  tokenizer.py   RWKV World 词表：字节 trie + 贪婪最长匹配，增量 UTF-8 解码
  s0.py          S0 容器、payload 加载（fp32/int8/rank4）、fla Cache 注入
  train.py       训练循环（NaN 防护、多轮链、热启动、层掩码）
  engine.py      人格注册、会话缓存、热切换、种子、流式生成
  arithmetic.py  S0 插值 / 加减算子
  editing.py     低秩定向编辑：子空间切除 / 注入 / rank-k
  quant.py       S0 int8 量化（逐层头 scales）
  lowrank.py     S0 rank-k SVD 因式分解（人格 ≈ 每头 3-4 维）
  bench.py       风格命中率 / 切换延迟 / O(1)-vs-O(T) 对比
  benchsuite.py  StateBench：风格 / 知识 / 能力保持评测
  factory.py     人格炼丹厂：卡片 → LLM 造数 → 训练 → 评测门禁 → 上线注册
  server.py      FastAPI：OpenAI 兼容 API + 会话 + WebUI 托管
  chat.py        终端 REPL
  cli.py         stateswap 命令入口
web/             零依赖 HTML/CSS/JS 前端（亮/暗主题）
persona_cards/   人格卡片（factory 输入：人设 + 风格标记 + 种子对话 + 话题池）
tests/           pytest（分词器 / 掩码 / 编辑 / 采样回归 / S0 梯度）
docs/            13 份研究/运维报告 + 每个研究的原始 JSON + 教程 + 演示图
scripts/         一次性脚本：各研究探针、模型转换、数据构建、冒烟测试
vendor/          fla 转换器原版 + BlinkDL 参考实现 + 词表（溯源见 NOTICE）
data/            训练语料（许可与出处见 NOTICE）
```

## 路线图

已完成：六研究状态科学弧线（[研究发现](#研究发现)）· O(1) 服务与热切换 ·
批量解码（6.29×）· int8/rank-4 压缩 · 人格炼丹厂 · 吸引子种子 ·
带实时状态监视器的 WebUI。

开放：

- [x] **三档标度扫描**——0.1B / 0.4B / 1.5B，全部人格用修复后的分词管线重训；
  混合形态**对状态容量不单调**（平滑交接 → 中段塌陷 → 共存窗）
- [x] **分层切除 / 软切除**——任务模式可局部切除（只切 L16–23 时任务仍有
  62%，均匀切除为 0%），风格不可（[docs/state-editing.md](docs/state-editing.md)）
- [x] **种子的 WebUI 暴露**——会话对话框已有 🌱 种子锁定开关，接入 session API


## 致谢

与 [NOTICE.md](NOTICE.md) 的溯源一一对应：

| 项目 | 许可 | 角色 |
|---|---|---|
| [Preen](https://github.com/No-22-Github/Preen) | Apache-2.0 | 思路直接来源——RWKV-7 state tuning on Mac/MLX；其理论指南与工程文档催生了本项目。stateswap 是独立的 NVIDIA/Windows 实现加服务层 |
| [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) | MIT | RWKV7 Triton 内核与建模代码；`vendor/` 权重转换器由其改编 |
| [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) | Apache-2.0 | BlinkDL 的 RWKV 仓库——架构参考、World 底座权重、World 词表 |
| [ModelScope RWKV 镜像](https://modelscope.cn/models/RWKV/rwkv-7-world) | Apache-2.0 | World 权重的国内直连下载路径 |
| [NekoQA-10K](https://huggingface.co/datasets/liumindmind/NekoQA-10K) | Apache-2.0 | 猫娘问答数据集（smoke 子集经 Preen 仓库分发） |
| [kxdw2580/catgirl-datasets](https://modelscope.cn/datasets/kxdw2580/catgirl-datasets) | Apache-2.0 | 扩展猫娘语料，清洗合并为 `data/neko_corpus_full.json` |
| WMT newstest 18/19/20（ModelScope `iic/WMT-…`） | 评估数据，见 NOTICE | 翻译人格的中英平行句 |
| [Transformers](https://github.com/huggingface/transformers) | Apache-2.0 | HF 模型加载路径（`AutoModelForCausalLM`） |
| [triton-windows](https://github.com/woct0rdho/triton-windows) | MIT | Blackwell sm_120 内核的 Windows Triton 轮子 |
| [FastAPI](https://github.com/fastapi/fastapi) / [uvicorn](https://github.com/encode/uvicorn) | MIT / BSD-3 | API 框架与 ASGI 服务器 + 静态 WebUI 托管 |

**衍生项目：** [MemoPet](https://github.com/Nice0o0/memopet) —— 一只以同一个
RWKV-7 递归状态为大脑的电子宠物（存档 / 分身 / 做梦 / 时光倒流，构建在本仓库的
S₀ 与存档工具之上）。

## 许可证

以 [MIT License](LICENSE) 发布。

```
Copyright 2026 Nice (https://github.com/Nice0o0/stateswap)
```

依赖与引用项目遵循各自上游许可——flash-linear-attention（MIT）、RWKV-LM 与
World 权重（Apache-2.0）、NekoQA-10K 与猫娘语料（Apache-2.0）、WMT newstest
评估集——完整溯源见 [NOTICE.md](NOTICE.md)。
