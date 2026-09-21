# Notices

## 训练数据

### personas/neko-0.1b, neko-0.4b（NekoQA 子集）

`data/nekoqa_smoke_200.json`：NekoQA-10K 猫娘问答数据集的 200 条冒烟子集，
取自 https://github.com/No-22-Github/Preen/tree/main/train_data/NekoQA_10k
（Apache-2.0，原作者 liumindmind，原始发布于
https://huggingface.co/datasets/liumindmind/NekoQA-10K ）。

### personas/neko-0.4b-v2（扩展猫娘语料）

`data/neko_corpus_full.json`：3,095 对去重后的猫娘风格 {instruction, output}。
NekoQA-10K 的原始 HF 数据在本项目环境下不可达（HF 直连被阻断，
ModelScope 镜像 mind666/NekoQA-10K 为空壳），故改用 ModelScope 上的公开
猫娘语料 **kxdw2580/catgirl-datasets**（Apache-2.0）合并清洗：
`v1/catgirl.json` + `v2/*.json`，剥离 `<think>` 块、去除 markdown 转义残留、
按 instruction 去重。清洗脚本见 `scripts/build_neko_corpus.py`。

### personas/zh2en-1.5b（中译英全量）

`data/zh2en_corpus_full.json`：7,369 对中英平行句，来自 ModelScope
`iic/WMT-Chinese-to-English-Machine-Translation-newstest` 的 wmt18/19/20
测试集（机器翻译评估数据，许可遵循上游发布方）合并 50 组手写口语对。
构建脚本 `scripts/build_zh2en_corpus.py`。

### personas/zh2en-0.4b, zh2en-0.4b-v2（中译英）

`data/zh2en_smoke.json`：50 组手写中译英样例（带指令前缀），MIT。
`data/zh2en_bare.json`：同 50 组去掉指令前缀的裸句版（Preen 式
"任务不变性"训练格式），由 `scripts/make_zh2en_bare.py` 生成，MIT。

## vendor/

- `convert_from_rwkv7.py`：来自 flash-linear-attention（MIT），
  本仓库有适配性修改（静默输出、bf16 默认、手动 safetensors 保存）。
- `rwkv7_x070_reference_model.py`：来自 BlinkDL/RWKV-LM（Apache-2.0），
  仅作参数命名与语义的交叉验证参考，不被运行时引用。
- `rwkv_vocab_v20230424.txt`：RWKV World 词表（BlinkDL/RWKV-LM）。

## 模型权重

RWKV-x070-World 底座权重来自 BlinkDL/RWKV-LM（Apache-2.0），
本项目通过 ModelScope 镜像 `RWKV/rwkv-7-world` 下载；
转换后格式仅服务于 fla/HF 加载路径，原许可不变。
