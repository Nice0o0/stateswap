# One-off: inspect candidate catgirl datasets (not part of the package).
import json
import os

for p in [os.path.expandvars(r"%TEMP%\muice.json"), os.path.expandvars(r"%TEMP%\common_v2.json")]:
    try:
        raw = open(p, encoding="utf-8").read()
        d = json.loads(raw)
        print("==", os.path.basename(p), "type:", type(d).__name__, "len:", len(d))
        print("   sample0:", json.dumps(d[0], ensure_ascii=False)[:280])
    except Exception as e:  # noqa: BLE001
        print("==", os.path.basename(p), "ERR", repr(e)[:150])
        try:
            print("   head:", raw[:200])
        except Exception:  # noqa: BLE001
            pass
