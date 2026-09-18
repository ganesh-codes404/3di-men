import sys

import pytest

from threedimen.external.cap3d import is_permissive, sample_cap3d


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
