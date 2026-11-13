#!/usr/bin/env python3
"""Fixed perplexity sample: the first 100,000 characters of the WikiText-2 (raw) test split,
cut at a line boundary. The same text is used for every model; token counts differ between
the Llama (128k vocab) and Granite (49k vocab) tokenizers, so perplexities are compared
within a family (quantization effect), not across families.
Output: data/wikitext2_test_100k.txt (+ .sha256)
"""
import hashlib
import pathlib

from huggingface_hub import hf_hub_download

PROJ = pathlib.Path(__file__).resolve().parents[1]
N_CHARS = 100_000

path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                       repo_type="dataset", local_dir=PROJ / "data" / "hf")
import pyarrow.parquet as pq  # noqa: E402

lines = pq.read_table(path).column("text").to_pylist()
text = "".join(lines)
cut = text.rfind("\n", 0, N_CHARS) + 1
sample = text[:cut]
out = PROJ / "data" / "wikitext2_test_100k.txt"
out.write_text(sample)
digest = hashlib.sha256(sample.encode()).hexdigest()
(out.parent / (out.name + ".sha256")).write_text(f"{digest}  {out.name}\n")
print(f"wrote {out}: {len(sample)} chars, {sample.count(chr(10))} lines, sha256 {digest[:16]}...")
