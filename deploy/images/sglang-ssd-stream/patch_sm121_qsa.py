from __future__ import annotations

import hashlib
import py_compile
from pathlib import Path

SOURCE = Path(
    "/sgl-workspace/sglang/python/sglang/srt/layers/attention/"
    "qwen_sparse_attn_backend.py"
)
EXPECTED_SHA256 = "c959835d05d0f395ad7eae4330cf264af9f6f7c1bff3d45a39bb953d2536f5f2"
OLD_IMPORT = "from sglang.srt.utils import is_sm100_supported"
NEW_IMPORT = (
    "from sglang.srt.utils import is_sm100_supported, is_sm120_supported"
)
OLD_GATE = "    if not is_sm100_supported():\n        return None"
NEW_GATE = (
    "    if not (is_sm100_supported() or is_sm120_supported()):\n"
    "        return None"
)


source = SOURCE.read_bytes()
actual_sha256 = hashlib.sha256(source).hexdigest()
if actual_sha256 != EXPECTED_SHA256:
    raise RuntimeError(
        "Refusing to patch an unreviewed QSA backend: "
        f"expected {EXPECTED_SHA256}, got {actual_sha256}"
    )

text = source.decode("utf-8")
if text.count(OLD_IMPORT) != 1 or text.count(OLD_GATE) != 1:
    raise RuntimeError("Reviewed SM121 QSA patch anchors are missing or ambiguous")

patched = text.replace(OLD_IMPORT, NEW_IMPORT, 1).replace(OLD_GATE, NEW_GATE, 1)
SOURCE.write_text(patched, encoding="utf-8")
py_compile.compile(str(SOURCE), doraise=True)

print(f"sm121_qsa_patch=ok source_sha256={actual_sha256}")
