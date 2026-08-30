# One-off: inspect WMT newstest CSVs (not part of the package).
import csv
import os

for n in ["wmt18", "wmt19", "wmt20"]:
    p = os.path.expandvars(r"%TEMP%\{}.csv".format(n))
    try:
        rows = list(csv.reader(open(p, encoding="utf-8")))
        print(n, "rows:", len(rows), "| header:", rows[0], "| row1:", rows[1][:2])
    except Exception as e:  # noqa: BLE001
        print(n, "ERR", repr(e)[:120])
