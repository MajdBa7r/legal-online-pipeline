"""The name exclusion list stays small, categorized, and never contains anything that can be a person's name."""
import importlib.util
import json
from pathlib import Path

import pytest

from app.pii.masker import _HERE, _load_names
from app.pii.name_postprocess import (EXCLUSION_CATEGORIES, MAX_EXCLUSIONS, is_excluded, load_exclusions,
                                      read_exclusion_entries)
from app.text.arabic import normalize

ROOT = Path(__file__).resolve().parent.parent

# Every name pool used by an evaluation set. Phase 3C adds the held-out pools here.
POOL_SOURCES = [ROOT / "eval" / "build_pii_cases.py"]
POOL_NAMES = ("FIRST_IN_LIST", "FIRST_NOT_IN_LIST", "FAMILY_IN_LIST", "FAMILY_NOT_IN_LIST")


def _pool_words() -> set[str]:
    words = set()
    for path in POOL_SOURCES:
        spec = importlib.util.spec_from_file_location(f"pools_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr in POOL_NAMES:
            for name in getattr(mod, attr, []):
                words.update(name.split())
    return words


def _gold_name_words() -> set[str]:
    words = set()
    with open(ROOT / "eval" / "pii_cases.jsonl", encoding="utf-8") as f:
        for line in f:
            for p in json.loads(line)["pii"]:
                if p["type"] == "name":
                    words.update(p["value"].split())
    return words


def _variants(word: str) -> set[str]:
    w = normalize(word)
    return {w, w[2:]} if w.startswith("ال") and len(w) > 4 else {w}


ENTRIES = read_exclusion_entries()


def test_size_and_categories():
    assert 0 < len(ENTRIES) <= MAX_EXCLUSIONS
    assert all(cat in EXCLUSION_CATEGORIES for _, cat in ENTRIES)
    assert {cat for _, cat in ENTRIES} == set(EXCLUSION_CATEGORIES)


def test_no_duplicates_after_normalization():
    norm = [normalize(w) for w, _ in ENTRIES]
    assert len(norm) == len(set(norm))


@pytest.mark.parametrize("source,words", [
    ("common_names.txt", _load_names(_HERE / "common_names.txt")),
    ("evaluation name pools", _pool_words()),
    ("gold name values in eval/pii_cases.jsonl", _gold_name_words()),
])
def test_no_entry_can_be_a_name(source, words):
    assert words, f"{source} is empty"
    names = {normalize(n) for n in words}
    clashes = [w for w, _ in ENTRIES if _variants(w) & names]
    assert clashes == [], f"exclusion entries that are also names in {source}: {clashes}"
    # and the reverse: no name in the pools is treated as excluded (e.g. through clitic stripping)
    assert [n for n in words if is_excluded(n)] == []


def test_malformed_files_are_rejected(tmp_path):
    for bad in ["المالك", "المالك | colour", "المالك عليه | legal_party"]:
        p = tmp_path / "x.txt"
        p.write_text(bad + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            read_exclusion_entries(p)


@pytest.mark.parametrize("word", ["المالك", "للمالك", "والمالك", "بالمالك", "زوجتي", "وزوجتي", "المستاجر",
                                  "المستأجرة", "مديري", "صاحب"])
def test_forms_are_matched(word):
    assert is_excluded(word, load_exclusions())


@pytest.mark.parametrize("word", ["مالك", "سند", "الزعبي", "ليث", "عمار", "السرحان", "ياسين", "المجالي"])
def test_names_are_not_matched(word):
    assert not is_excluded(word, load_exclusions())
