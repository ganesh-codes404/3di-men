import sys

import pytest

from threedimen.external.trellis import _normalize, sample_trellis


def test_normalize_strips_separators_and_case():
    assert _normalize("ObjaverseXL-Sketchfab.csv") == _normalize("objaversexl_sketchfab.CSV")


def test_sample_trellis_raises_clear_error_without_objaverse_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "objaverse", None)
    monkeypatch.setitem(sys.modules, "objaverse.xl", None)
    with pytest.raises(ImportError, match="external"):
        sample_trellis(5)
