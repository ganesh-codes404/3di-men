import sys

import pytest

from threedimen.external.cap3d import _load_annotations_robust, is_permissive, sample_cap3d


def test_is_permissive_recognizes_common_cc_licenses():
    assert is_permissive("BY")
    assert is_permissive("cc0")
    assert is_permissive("by-sa")


def test_is_permissive_rejects_unknown_or_restrictive():
    assert not is_permissive("all-rights-reserved")
    assert not is_permissive("")


def test_sample_cap3d_raises_clear_error_without_objaverse_installed(monkeypatch):
    # Deterministic regardless of whether `objaverse` happens to be installed
    # in this environment -- setting a module to None in sys.modules makes
    # Python's import system raise ImportError for it immediately.
    monkeypatch.setitem(sys.modules, "objaverse", None)
    with pytest.raises(ImportError, match="external"):
        sample_cap3d(5)


class _FakeObjaverse:
    """Simulates objaverse.load_annotations() raising for the whole batch
    if it contains ANY uid in `bad_uids` -- exactly the real behavior that
    made the flat one-by-one fallback necessary in the first place."""

    def __init__(self, bad_uids: set[str]):
        self.bad_uids = bad_uids
        self.call_count = 0

    def load_annotations(self, uids):
        self.call_count += 1
        if any(u in self.bad_uids for u in uids):
            raise KeyError("simulated: batch contains an unresolvable uid")
        return {u: {"license": "by"} for u in uids}


def test_load_annotations_robust_isolates_bad_uids_without_losing_good_ones():
    uids = [f"uid{i}" for i in range(50)]
    fake = _FakeObjaverse(bad_uids={"uid7", "uid41"})

    result = _load_annotations_robust(fake, uids)

    assert "uid7" not in result
    assert "uid41" not in result
    for uid in uids:
        if uid not in {"uid7", "uid41"}:
            assert uid in result, f"{uid} should have resolved but didn't"


def test_load_annotations_robust_scales_sublinearly_with_batch_size():
    # The whole point of divide-and-conquer over flat one-by-one: with a
    # fixed number of bad uids, call count should grow with log(n), not n,
    # as the surrounding batch grows.
    small_fake = _FakeObjaverse(bad_uids={"uid3"})
    _load_annotations_robust(small_fake, [f"uid{i}" for i in range(20)])

    large_fake = _FakeObjaverse(bad_uids={"uid3"})
    _load_annotations_robust(large_fake, [f"uid{i}" for i in range(2000)])

    # 100x more uids should NOT mean anywhere near 100x more calls -- a flat
    # one-by-one fallback would make large_fake.call_count roughly 2000,
    # dwarfing small_fake's. Divide-and-conquer keeps it within a small
    # constant multiple (bounded by log2(n) extra splits around the one bad uid).
    assert large_fake.call_count < small_fake.call_count * 20


def test_load_annotations_robust_handles_empty_input():
    fake = _FakeObjaverse(bad_uids=set())
    assert _load_annotations_robust(fake, []) == {}
