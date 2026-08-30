# Notices

## data/nekoqa_smoke_200.json

NekoQA 猫娘问答数据集的 200 条冒烟子集，取自
https://github.com/No-22-Github/Preen/tree/main/train_data/NekoQA_10k
（Apache-2.0）。原始数据集的完整许可与来源说明见该仓库 NOTICE.md。

## data/zh2en_smoke.json

50 组手写中译英样例，仅用于演示 state tuning 的任务模式固化，MIT。

## vendor/

- `convert_from_rwkv7.py`：来自 flash-linear-attention（MIT），
  本仓库有适配性修改（静默输出、bf16 默认、手动 safetensors 保存）。
- `rwkv7_x070_reference_model.py`：来自 BlinkDL/RWKV-LM（Apache-2.0），
  仅作参数命名与语义的交叉验证参考，不被运行时引用。
- `rwkv_vocab_v20230424.txt`：RWKV World 词表（BlinkDL/RWKV-LM）。
