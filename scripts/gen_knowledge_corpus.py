# One-off: generate a synthetic knowledge corpus for the S0 compressed-memory
# experiment (not part of the package).
#
# 设计：
#   - 虚构世界观"星辰科技"：员工/产品/事件三类实体，每类带属性
#   - 每条事实一个 (实体, 属性, 值) 三元组，QA 由模板生成（3-4 种问法）
#   - 训练集容量三档：100 / 300 / 500 条事实
#   - 评测集分两块：
#       memorization —— 训练中出现过的实体，换一种问法（测措辞鲁棒性）
#       generalization —— 完全没训过的实体（测 schema 泛化，预期接近 0）
import faulthandler
import json
import random
from pathlib import Path

faulthandler.dump_traceback_later(15, exit=True)  # 卡死时自动打印堆栈并退出

rng = random.Random(20260831)

SURNAME = ["林", "沈", "顾", "陆", "苏", "叶", "秦", "许", "韩", "傅", "钟", "温", "祁", "岑", "饶",
           "卫", "庄", "厉", "蔺", "屠"]
GIVEN = ["晚晴", "知远", "若愚", "慕白", "之恒", "清越", "砚秋", "既明", "南乔", "望舒",
         "思齐", "北辰", "言蹊", "疏影", "听澜", "扶摇", "时雨", "星野", "江离", "鹿鸣",
         "云舟", "雪汀", "鹤龄", "竹意", "兰舟"]

DEPTS = ["基础模型部", "推理引擎部", "多模态部", "数据平台部", "安全对齐部", "端侧智能部", "具身智能部"]
TITLES = ["首席技术官", "首席科学家", "研发副总裁", "高级研究员", "主任工程师", "产品负责人", "数据科学总监"]
PRODUCTS = ["星尘", "流萤", "望楼", "青鸾", "烛龙", "河图", "洛书", "天枢", "揽月", "夸父",
            "烛照", "玄圃", "若木", "朱明", "白驹", "北落", "师门", "紫微", "勾陈", "文昌"]
VERSIONS = ["1.0", "1.5", "2.0", "2.5", "3.0", "Lite", "Pro", "Max", "Air", "Ultra"]
YEARS = ["2023年", "2024年", "2025年", "2026年"]
QUARTERS = ["第一季度", "第二季度", "第三季度", "第四季度"]
CITIES = ["杭州", "上海", "北京", "深圳", "成都", "苏州", "广州", "南京"]
EVENTS = ["全球开发者大会", "技术开放日", "人工智能峰会", "年度发布会", "生态合作论坛"]
MONTHS = ["3月", "5月", "6月", "9月", "11月"]

used_names, used_products = set(), set()

# 确定性发号器：预洗牌、按序弹出、耗尽即抛错（拒绝采样在池近枯竭时会死循环）
_ALL_NAMES = [s + g for s in SURNAME for g in GIVEN]
rng.shuffle(_ALL_NAMES)
_NAME_IT = iter(_ALL_NAMES)
_ALL_PRODUCTS = [p + "-" + v for p in PRODUCTS for v in VERSIONS]
rng.shuffle(_ALL_PRODUCTS)
_PRODUCT_IT = iter(_ALL_PRODUCTS)


def unique_name():
    used = next(_NAME_IT)
    used_names.add(used)
    return used


def unique_product():
    used = next(_PRODUCT_IT)
    used_products.add(used)
    return used


def make_facts(n_people, n_products, n_events):
    facts = []  # (subject, attribute, value)
    for _ in range(n_people):
        name = unique_name()
        facts.append((f"{name}", "职位", rng.choice(TITLES)))
        facts.append((f"{name}", "所在部门", rng.choice(DEPTS)))
        facts.append((f"{name}", "工作城市", rng.choice(CITIES)))
    for _ in range(n_products):
        p = unique_product()
        facts.append((f"大模型{p}", "发布年份", rng.choice(YEARS)))
        facts.append((f"大模型{p}", "发布季度", rng.choice(QUARTERS)))
        facts.append((f"大模型{p}", "研发负责人", unique_name()))
    for _ in range(n_events):
        e = rng.choice(EVENTS)
        facts.append((f"{rng.choice(YEARS)}{e}", "举办城市", rng.choice(CITIES)))
        facts.append((f"{rng.choice(YEARS)}{e}", "发布主角", unique_name()))
    rng.shuffle(facts)
    return facts


# 事实三元组总数：每 3 条人 / 3 条产品 / 2 条事件
FACT_SETS = {
    100: make_facts(11, 11, 12)[:100],
    300: make_facts(34, 34, 32)[:300],
    500: make_facts(56, 56, 56)[:500],
}

# 问法模板：训练用一组，评测用另一组（测措辞鲁棒性）
ASK_TEMPLATES = {
    "职位": ["{s}在星辰科技担任什么职位？", "{s}的职位是什么？", "谁在星辰科技担任{v}？这个人是谁？"],
    "所在部门": ["{s}在哪个部门工作？", "{s}属于星辰科技的哪个部门？"],
    "工作城市": ["{s}在哪个城市工作？", "{s}的工作地点是哪里？"],
    "发布年份": ["大模型{s}是哪一年发布的？", "{s}的发布年份是？"],
    "发布季度": ["大模型{s}是在哪个季度发布的？", "{s}发布于哪个季度？"],
    "研发负责人": ["大模型{s}的研发负责人是谁？", "{s}是由谁负责研发的？"],
    "举办城市": ["{s}在哪个城市举办？", "{s}的举办城市是哪里？"],
    "发布主角": ["{s}上由谁担任发布主角？", "{s}的主角是谁？"],
}
EVAL_TEMPLATES = {
    "职位": ["星辰科技的{s}是什么职务？"],
    "所在部门": ["请问{s}隶属于哪个部门？"],
    "工作城市": ["{s}常驻哪座城市办公？"],
    "发布年份": ["{s}这款大模型于哪一年问世？"],
    "发布季度": ["{s}是在第几季度对外发布的？"],
    "研发负责人": ["谁是大模型{s}的研发负责人？"],
    "举办城市": ["{s}是在哪座城市举行的？"],
    "发布主角": ["{s}的发布主角是哪位？"],
}


def qa_of(fact, templates):
    s, attr, v = fact
    q = rng.choice(templates[attr]).format(s=s, v=v)
    return {"instruction": q, "output": v, "subject": s, "attribute": attr}


out = {}
for size, facts in FACT_SETS.items():
    train = [qa_of(f, ASK_TEMPLATES) for f in facts]
    rng.shuffle(train)
    out[size] = {"facts": facts, "train": train}

# 评测集：memorization（500 档训过的实体，换问法）+ generalization（全新实体）
eval_mem = [qa_of(f, EVAL_TEMPLATES) for f in FACT_SETS[500][:60]]
gen_facts = make_facts(20, 10, 10)
eval_gen = [qa_of(f, EVAL_TEMPLATES) for f in gen_facts[:40]]

Path("data").mkdir(exist_ok=True)
for size, d in out.items():
    with open(f"data/knowledge_qa_{size}.json", "w", encoding="utf-8") as f:
        json.dump(d["train"], f, ensure_ascii=False, indent=0)
with open("data/knowledge_eval.json", "w", encoding="utf-8") as f:
    json.dump({"memorization": eval_mem, "generalization": eval_gen,
               "memorization_facts": FACT_SETS[500][:60]},
              f, ensure_ascii=False, indent=1)

print("train sizes:", {k: len(v["train"]) for k, v in out.items()})
print("eval memorization:", len(eval_mem), "| generalization:", len(eval_gen))
print("sample:", json.dumps(out[100]["train"][0], ensure_ascii=False))
