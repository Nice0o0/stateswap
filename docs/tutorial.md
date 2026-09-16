# stateswap 使用教程

> 从零开始，在一张消费级 NVIDIA 显卡上跑通"一个 RWKV-7 底座 + 多个人格"的完整流程。
> 覆盖：安装 → 模型准备 → WebUI 聊天/混合/训练 → 命令行与 API → 常见问题。

---

## 0. 你需要准备什么

| 项 | 要求 | 说明 |
|---|---|---|
| 显卡 | NVIDIA，≥8GB 显存 | 本项目全部数字在 RTX 5070 Ti 12GB 上验证；Blackwell（50 系）与 Ampere+ 均可 |
| 系统 | Windows 10/11 或 Linux | 原生 Windows 已完整验证，**不要用 Python 3.10**（见 FAQ） |
| Python | **3.11 或 3.12** | 用 [uv](https://docs.astral.sh/uv/) 管理，一条命令装好 |
| 磁盘 | ≥15GB | 模型权重 + 转换产物 |

## 1. 安装

```bash
# 克隆并进入项目
git clone https://github.com/Nice0o0/stateswap.git
cd stateswap

# 1) 用 uv 创建 Python 3.12 虚拟环境（国内网络可用镜像，见 FAQ）
uv venv --python 3.12 .venv

# 2) 安装 PyTorch（CUDA 12.8 轮子，Blackwell 显卡必须）
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3) 安装其余依赖（Windows 需 triton-windows；Linux 直接装 flash-linear-attention 即可）
uv pip install triton-windows        # Linux 用户跳过这行
uv pip install transformers flash-linear-attention fastapi "uvicorn[standard]"
uv pip install -e .

# 4) 验证 GPU 可用
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

预期输出类似：`2.11.0+cu128 True NVIDIA GeForce RTX 5070 Ti`。

## 2. 获取并转换模型

stateswap 使用 RWKV-7 World 系列开源权重。从 [ModelScope 镜像](https://modelscope.cn/models/RWKV/rwkv-7-world)（国内直连快）或 HuggingFace `BlinkDL/rwkv-7-world` 下载 `.pth` 权重，然后转换成内部格式：

```bash
python scripts/convert_model.py --pth RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth ^
    --out models/rwkv7-1.5b-world-hf --precision bfloat16
```

- **1.5B**（约 3GB）：对话质量推荐起步规格
- **0.4B**（约 0.9GB）：显存紧张或快速实验用
- 转换产物在 `models/<名字>/`，包含 `model.safetensors` + `config.json`

## 3. 启动服务与 WebUI

### 3.0 日常启动（已完成安装后）

**一键启动**：Windows 双击 `start_webui.bat`，Linux 运行 `bash start_webui.sh`。
脚本自动选择虚拟环境、加载默认底座并打开浏览器。可用环境变量覆盖：

```bash
# Windows（cmd）
set STATESWAP_MODEL=models/rwkv7-0.4b-world-hf && set STATESWAP_PORT=8001 && start_webui.bat
# Linux
STATESWAP_PORT=8001 bash start_webui.sh
```

**手动启动**（等价于脚本做的事）：

```bash
python -m stateswap.server --model models/rwkv7-1.5b-world-hf --persona-dir personas --port 8000
```

- 局域网访问：加 `--host 0.0.0.0`（无鉴权，仅限可信网络）
- 后台常驻：Linux 用 `nohup ... > server.log 2>&1 &`；Windows 直接最小化该窗口即可
- 会话上限 64 个（超出回收最久未用的），人格文件持久化在 `personas/`

浏览器打开 **http://127.0.0.1:8000**。

> **首次打开请稍候**：模型加载 + Triton 内核自动调优需要几十秒，左侧状态点变绿（"服务正常"）即可使用。发送按钮在引擎就绪前会有明确提示，不会报错。

### 3.1 聊天

1. 左侧确认已自动创建会话（或点 **＋ 开启新对话**）
2. 顶栏下拉选择人格（如 `neko-1.5b`），切换即时生效（约 3ms）。**人格按用途分组**：
   💬 聊天（`neko-*`）/ 🌐 任务（`zh2en-*`，翻译专用——它会把你说的一切翻成英文，
   "无法对话"是设计如此）/ 🧪 实验（`mem-*`、`mix-*`，研究产物）/ ⚪ 基线
   （`none`，无调教裸底座，容易跑偏成预训练文本）
3. 输入消息，**Enter 发送**，**Shift+Enter 换行**（中文输入法确认候选词的 Enter 不会误发）
4. 回复逐 token 流式显示，下方标签显示真实性能：`N tok · prefill Xms · Xms/tok · 状态 XMB`
5. 顶栏 **⚖️ 平衡 / 🎯 精准 / 🎨 创意** 切换采样温度：小模型建议平衡档，创意档（1.0）发散度高
6. 顶栏 **🧬** 展开状态监视器：24 层状态解剖条（柱高=层范数，颜色=与人格 S₀ 的锚定度），每轮实时刷新

> 如果某条长回复被提示"检测到退化，已回滚会话状态"：这是**退化护栏**在工作——
> 复读/乱码的回复不会被写进会话记忆，防止污染后续对话。换个问法或要求更短的回复即可。

### 3.2 会话管理

- 左侧"会话记录"列出所有会话（人格名、轮数、状态内存），**点击切换**，悬停 **✕** 删除
- 每个会话的状态内存恒定（1.5B 约 12.8MB），不随对话轮数增长
- **跨轮记忆 = 状态 + 有界文本重放**：每轮 prefill 会把最近 4 轮对话拼进 prompt
  （`POST /v1/sessions` 的 `context_replay` 字段，默认 4，0 = 纯状态模式）。
  实测 1.5B 的递归状态**保留不了具体事实**（名字、宠物这类稀疏信息两轮内
  就被覆写，见 `scripts/probe_context.py`）；重放让近 4 轮事实以可见文本存在，
  即刻/延迟追问都能答对。代价是 prefill 每轮多约 200–300 token（~150ms）。
  **4 轮以前的事实仍会蒸发，代词回指也不可靠——这是 1.5B 底座的硬边界，
  不是 bug。**
- 服务重启后会话清空（状态在内存中）；人格文件持久化在 `personas/` 目录

### 3.3 人格混合实验台

侧栏进入 **🧪 人格混合实验台**：选两个人格 A/B、拖动 α 滑杆、注册为新人格。

> ⚠️ 实测结论：S₀ 线性混合会发生**尖锐相变**而非平滑混合——某个混合比例附近行为会
> 突变为单一模式。详见 [docs/state-arithmetic2.md](state-arithmetic2.md)。这是特性不是 bug。

### 3.4 训练新人格

侧栏进入 **🔥 训练新人格**：

1. 选择数据集（`data/*.json`，格式为 `{"instruction": "...", "output": "..."}` 列表）
2. 填人格名（将出现在顶栏下拉里）、步数、学习率（默认 1e-4 即可）
3. 点开始训练，进度条实时显示 loss；完成后**自动注册为新人格**

参考规模：1.5B 底座 ctx 512 约 0.2s/step、峰值 4.8GB 显存；3095 对语料 4000 步 ≈ 15 分钟。

> **打算拿来长对话的人格，建议用多轮样本训练**：命令行加 `--turns 3 --ctx 1024`，
> 把连续若干对拼成一段多轮对话（每个 Assistant 段都算 loss）。单轮样本只在
> "对话开头"训练 S₀，是长对话退化的训练侧根因；多轮训练后 15 轮探针退化
> 4/15 → 0/15（见 [benchmarks.md](benchmarks.md) §7）。

## 4. 命令行用法

WebUI 之外，所有能力都有 CLI：

```bash
# 转换模型
python -m stateswap.cli convert --pth <权重.pth> --out models/<名字>

# 训练人格（长对话场景加 --turns 3 --ctx 1024）
python -m stateswap.cli train --model models/rwkv7-1.5b-world-hf ^
    --data data/neko_corpus_full.json --out personas/neko-1.5b --steps 4000

# 交互式终端聊天
python -m stateswap.cli chat --model models/rwkv7-1.5b-world-hf --persona neko-1.5b
```

评测套件（风格命中率 / 知识注入 / 能力保持，输出 scorecard）：

```bash
python -m stateswap.benchsuite --model models/rwkv7-1.5b-world-hf --persona neko-1.5b ^
    --knowledge data/knowledge_eval.json --out docs/scorecard.json
```

## 5. OpenAI 兼容 API

服务完全兼容 OpenAI Chat Completions 格式，`model` 字段就是人格名：

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "neko-1.5b",
  "messages": [{"role": "user", "content": "陪我聊聊天"}]
}'
```

Python（openai 库）：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
resp = client.chat.completions.create(
    model="neko-1.5b",
    messages=[{"role": "user", "content": "陪我聊聊天"}],
)
print(resp.choices[0].message.content)
```

进阶端点（详见 `docs/` 与源码 `server.py`）：

- `POST /v1/sessions` / `GET /v1/sessions` / `DELETE /v1/sessions/{id}` —— 会话生命周期
- `GET /v1/sessions/{id}` —— 完整对话历史（前端"会话记录"的数据源）
- `GET /v1/sessions/{id}/state-stats` —— 每层状态范数 + 与人格 S₀ 的余弦（🧬 监视器数据源）
- `POST /v1/sessions/{id}/swap` —— 人格热切换（`{"persona": "...", "keep_context": false}`）
- `POST /v1/personas/mix` —— S₀ 线性混合
- `POST /v1/train/start` + `GET /v1/train/status` —— WebUI 训练的后端

## 6. 人格压缩：int8（÷4）与 rank-4（÷7.9）

int8 量化：

```bash
python -c "from stateswap.quant import save_quantized; import torch; \
  save_quantized(torch.load('personas/neko-1.5b/s0.pt', weights_only=False)['s0'], \
  'personas/neko-1.5b.int8.pt', {'quantized': 'int8'})"
```

rank-k SVD 因式分解（人格 ≈ 每头 3-4 维，详见 [persona-anatomy.md](persona-anatomy.md)）：

```bash
python -c "from stateswap.lowrank import save_lowrank; import torch; \
  save_lowrank(torch.load('personas/neko-1.5b/s0.pt', weights_only=False)['s0'], \
  'personas/neko-1.5b.rank4.pt', k=4)"
```

重启服务后，`neko-1.5b-int8` / `neko-1.5b-rank4` 会自动出现在人格列表
（`personas/*.int8.pt` 与 `*.rank<N>.pt` 均被自动加载）。实测 int8 12.58MB →
3.15MB 风格/语义保真；rank-4 → 1.59MB 行为 100% 保真。

## 7. 批量多会话解码（Python API）

```python
from stateswap.engine import Engine

engine = Engine("models/rwkv7-1.5b-world-hf", "vendor/rwkv_vocab_v20230424.txt")
engine.register_persona("neko-1.5b", "personas/neko-1.5b/s0.pt")
sessions = [engine.new_session("neko-1.5b") for _ in range(8)]
results = engine.batch_chat(
    [s.session_id for s in sessions],
    ["早上好", "讲个笑话", "推荐一部动画", "今晚吃什么"] * 2,
    max_new_tokens=96,
)
```

实测 8 会话聚合吞吐 135.5 tok/s（单会话循环的 **6.29×**，近线性）。

## 8. 常见问题（FAQ）

**Q：第一次发消息要等很久？**
首次推理触发 Triton 内核自动调优（约 30-40 秒），之后正常（~50ms/token）。
内核缓存会写入磁盘，重启服务后不再重复调优。

**Q：Enter 变成换行、发送按钮异常大/没箭头？**
确认浏览器拿到的是最新前端文件（Ctrl+F5 强刷）。两者均已修复；
若仍有问题，检查 `web/style.css` 中 `#btn-send` 规则是否存在。

**Q：显存不够（CUDA OOM）？**
退回 0.4B 底座（训练峰值 2GB）；或降低 `--ctx`；或关掉正在训练的线程。

**Q：训练 loss 归零但模型答不对 / 换个问法就答错？**
先检查分词边界：本项目的数据管线已修复"整句编码导致边界 token 吞掉答案首字符"
的问题（见 [docs/state-arithmetic2.md](state-arithmetic2.md) §4）。自建数据时确保
prompt 与 completion **分段编码**。

**Q：Windows 上装 flash-linear-attention 报 triton 错误？**
需要 `triton-windows` 且 Python ≥3.11（3.10 的 inspect 行为会让 triton 崩溃）。

**Q：国内外网络下载模型/依赖失败？**
权重走 ModelScope（`RWKV/rwkv-7-world`）；pip 可配代理或镜像；
本项目 `docs/engineering-notes.md` 记录了完整的网络应对方案。

## 9. 目录速查

| 路径 | 内容 |
|---|---|
| `web/` | WebUI 前端（纯 HTML/CSS/JS，FastAPI 托管） |
| `personas/` | 当前底座的人格状态（`<名>/s0.pt`、`<名>.int8.pt`、`<名>.rank<N>.pt`） |
| `personas-legacy/` | 0.4B/0.1B 旧底座人格存档 |
| `data/` | 训练数据集（知识注入/猫娘/翻译语料） |
| `models/` | 转换后的底座权重（不入库） |
| `docs/` | 工程笔记、实验报告、基准数据 |
| `scripts/` | 一次性实验与工具脚本 |
