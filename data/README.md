# data/ — 训练与评测语料

这些文件让仓库里的研究可复现（`scripts/probe_*.py`、`docs/*.md` 都直接引用它们）。

| 文件 | 内容 | 来源 / 许可 |
|---|---|---|
| `neko_corpus_full.json` | 3,095 对猫娘风格问答 | 由 `scripts/build_neko_corpus.py` 从 ModelScope 公开猫娘语料清洗而来；原始语料请遵守其发布方许可 |
| `neko_scaling_*.json` | 上述语料的 50–1600 对子集 | 同上的子采样，`docs/scaling.md` 用 |
| `zh2en_corpus_full.json` | 7,369 对中→英平行句 | WMT newstest 2018/2019/2020 + 50 条口语对（新闻语料，WMT 官方使用条款） |
| `zh2en_bare.json` / `zh2en_smoke.json` | 裸格式小子集 / 冒烟集 | 同上，`scripts/make_zh2en_bare.py` 生成 |
| `knowledge_qa_100/300/500.json` + `knowledge_eval.json` | **合成**知识库（虚构"星辰科技"） | 完全程序化生成（`scripts/gen_knowledge_corpus.py`），无第三方版权 |
| `nekoqa_smoke_200.json` | 200 对小冒烟集 | 同 `neko_corpus_full.json` |

## 如果你要公开发布衍生作品

- WMT 语料的使用受 WMT 官方条款约束（通常允许研究用途，再分发需保留出处）。
- ModelScope 猫娘语料的许可随发布方而定，清洗后的衍生数据建议在 README 里
  注明来源与清洗脚本，避免直接声称原创。
- 合成知识库（`knowledge_qa_*`）是本项目自己生成的，可随意使用。

生成脚本都在 `scripts/` 下，`docs/compressed-memory.md` 与 `docs/scaling.md`
记录了完整的复现命令。
