# One-off: reveal what triton's get_def_col_number actually receives.
import triton.runtime.jit as J

orig = J.get_def_col_number


def dbg(raw_src_str):
    try:
        return orig(raw_src_str)
    except ValueError:
        print("=== FAILED def col parse ===")
        print("SRC LEN:", len(raw_src_str))
        print("HEAD:", repr(raw_src_str[:300]))
        print("TAIL:", repr(raw_src_str[-200:]))
        raise


J.get_def_col_number = dbg

import fla.ops.simple_gla.parallel  # noqa: F401  (expected to trigger the bug)

print("import ok")
