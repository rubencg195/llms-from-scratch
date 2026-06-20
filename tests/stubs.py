"""Offline stand-ins for the network resources the labs download.

Goal: let the whole suite run air-gapped (e.g. on a laptop with no internet, or
in CI) **without masking real bugs**. The fake ``Dataset`` therefore mirrors the
real ``datasets.Dataset`` indexing contract exactly:

    ds[int]   -> dict (one row)
    ds[str]   -> list (one column)
    ds[slice] -> dict of lists (a batch)   <-- same gotcha as the real library
    iter(ds)  -> yields row dicts
    len(ds)   -> row count

Because the semantics match, code that misuses the API (e.g. iterating over a
slice expecting row dicts) fails here just like it would on real data.

These stubs are only installed in ``offline`` mode. In ``online`` mode the real
``datasets`` / ``tiktoken`` packages are used so downloads are exercised for
real.
"""

from __future__ import annotations

import sys
import types
from typing import List


class FakeDataset:
    """Minimal re-implementation of the parts of datasets.Dataset labs use."""

    def __init__(self, rows: List[dict]):
        self.rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.rows[key]
        if isinstance(key, slice):
            batch = self.rows[key]
            if not batch:
                return {col: [] for col in self.column_names}
            return {col: [r[col] for r in batch] for col in batch[0].keys()}
        if isinstance(key, str):
            return [r[key] for r in self.rows]
        raise TypeError(f"FakeDataset indices must be int/str/slice, got {type(key)}")

    def select(self, indices):
        return FakeDataset([self.rows[i] for i in indices])

    def map(self, fn, **kwargs):
        return FakeDataset([{**r, **fn(r)} for r in self.rows])


# --- synthetic corpora -------------------------------------------------------

_TINYSTORIES_SENTENCES = [
    "Once upon a time there was a little girl named Lily.",
    "She had a small red ball and loved to play in the garden.",
    "The sun was warm and the birds were singing in the tall trees.",
    "Lily met a kind cat who wanted to share her shiny golden toy.",
    "They ran and laughed together until the moon came up at night.",
    "A brave little dog helped his friend find the way back home.",
    "The boy was happy because his mother made a sweet apple pie.",
    "Every day the children would read a new story before they slept.",
    "The river was cold but the fish were quick and very clever.",
    "In the end everyone learned that being kind is the best of all.",
]


def _make_tinystories(n: int = 400) -> FakeDataset:
    rows = []
    for i in range(n):
        a = _TINYSTORIES_SENTENCES[i % len(_TINYSTORIES_SENTENCES)]
        b = _TINYSTORIES_SENTENCES[(i + 3) % len(_TINYSTORIES_SENTENCES)]
        c = _TINYSTORIES_SENTENCES[(i + 6) % len(_TINYSTORIES_SENTENCES)]
        rows.append({"text": f"{a} {b} {c}"})
    return FakeDataset(rows)


def _make_gsm8k(n: int = 80) -> FakeDataset:
    rows = []
    for i in range(n):
        x, y = (i % 9) + 1, (i % 5) + 2
        total = x * y
        rows.append(
            {
                "question": f"There are {x} baskets with {y} apples each. How many apples are there in total?",
                "answer": (
                    f"Each basket has {y} apples and there are {x} baskets.\n"
                    f"So we multiply {x} * {y} = {total}.\n"
                    f"#### {total}"
                ),
            }
        )
    return FakeDataset(rows)


def _make_glaive(n: int = 40) -> FakeDataset:
    rows = []
    for i in range(n):
        rows.append(
            {
                "system": "SYSTEM: You are a helpful assistant with access to functions.",
                "chat": (
                    "USER: What is the weather in Paris?\n\n"
                    'ASSISTANT: <functioncall> {"name": "get_weather", '
                    '"arguments": {"city": "Paris"}}\n\n'
                ),
            }
        )
    return FakeDataset(rows)


def _make_generic(n: int = 20) -> FakeDataset:
    return FakeDataset([{"text": f"sample document number {i}"} for i in range(n)])


def fake_load_dataset(path, *args, **kwargs):
    name = (path or "").lower()
    if "tinystories" in name:
        return _make_tinystories()
    if "gsm8k" in name:
        return _make_gsm8k()
    if "glaive" in name or "function-calling" in name:
        return _make_glaive()
    # tiny_image_stories and anything else: minimal generic rows
    return _make_generic()


# --- tiktoken byte-level shim ------------------------------------------------


class _ByteEncoding:
    """Tiny byte-level encoder with the tiktoken surface labs use."""

    name = "gpt2-offline-shim"
    n_vocab = 256

    def encode(self, text, *args, **kwargs):
        return list(text.encode("utf-8"))

    def decode(self, tokens, *args, **kwargs):
        return bytes(t % 256 for t in tokens).decode("utf-8", errors="replace")


def install_offline_stubs():
    """Insert offline fakes into sys.modules / patch real modules in place.

    Safe to call multiple times. Returns the list of names that were stubbed.
    """
    stubbed = []

    # datasets: replace load_dataset with the synthetic generator.
    try:
        import datasets as _real_datasets  # noqa: F401

        _real_datasets.load_dataset = fake_load_dataset  # type: ignore
        stubbed.append("datasets.load_dataset (patched real package)")
    except Exception:
        fake = types.ModuleType("datasets")
        fake.load_dataset = fake_load_dataset  # type: ignore
        fake.Dataset = FakeDataset  # type: ignore
        sys.modules["datasets"] = fake
        stubbed.append("datasets (fake module)")

    # tiktoken: keep the real package if its cache works offline; otherwise shim.
    try:
        import tiktoken as _tk

        try:
            _tk.get_encoding("gpt2")  # works if already cached
        except Exception:
            _tk.get_encoding = lambda *a, **k: _ByteEncoding()  # type: ignore
            stubbed.append("tiktoken.get_encoding (byte-level shim)")
    except Exception:
        fake_tk = types.ModuleType("tiktoken")
        fake_tk.get_encoding = lambda *a, **k: _ByteEncoding()  # type: ignore
        sys.modules["tiktoken"] = fake_tk
        stubbed.append("tiktoken (fake module)")

    return stubbed


if __name__ == "__main__":
    ds = fake_load_dataset("roneneldan/TinyStories", split="train[:1%]")
    print("rows:", len(ds))
    print("row 0:", ds[0])
    print("slice type:", type(ds[:3]))
    print("column:", ds["text"][:1])
