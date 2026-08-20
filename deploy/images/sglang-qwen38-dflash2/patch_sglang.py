import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if source.count(old) != 1:
        raise RuntimeError(f"Expected exactly one compatibility target in {path}")
    path.write_text(source.replace(old, new))


utils_path = Path(sys.argv[1])
worker_path = Path(sys.argv[2])

replace_once(
    utils_path,
    "from sglang.srt.speculative.spec_utils import sample_simulated_acc_len",
    "from sglang.srt.speculative.spec_utils import "
    "_sample_simulated_acc_len as sample_simulated_acc_len",
)
replace_once(
    worker_path,
    "from sglang.srt.layers.logprob_processor import compute_spec_logprobs",
    "from sglang.srt.layers.logprob_processor import compute_spec_v2_logprobs",
)
replace_once(
    worker_path,
    """            compute_spec_logprobs(
                batch,
                logits_output,
                out_tokens.reshape(-1),
                chain_stride=block_size,
            )""",
    """            output_indices = torch.arange(
                bs * block_size, dtype=torch.int64, device=device
            ).view(bs, block_size)
            compute_spec_v2_logprobs(
                batch,
                logits_output,
                out_tokens.reshape(-1),
                output_indices,
                block_size - 1,
            )""",
)
