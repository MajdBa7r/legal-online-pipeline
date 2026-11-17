"""Detector interface for the PII benchmark (evaluation only; the pipeline still uses masker.py directly).

Every detector returns character spans `Span(start, end, type, source)` over the ORIGINAL text, so
rule-based and model-based detectors can be scored with the same harness (app/pii/pii_eval.py).

* RuleDetector   - exactly the spans PIIMasker.mask() would replace (masker.py is used unchanged).
* ModelDetector  - wraps any model that returns spans or entity strings (Hugging Face NER, GLiNER, GLiNER2).
                   Heavy libraries are imported only inside the factory functions, so nothing new is needed
                   in requirements.txt; the model detectors are meant to run in Colab.
* HybridDetector - union of a rule detector and a model detector (optionally only some model types).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from app.pii import masker as M

CORE_TYPES = ("name", "phone", "national_id", "iban", "email")
EXTRA_TYPES = ("address", "plate")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    type: str
    source: str = ""


class RuleDetector:
    """The production rules alone: exactly the spans of PIIMasker.mask() with no name model."""

    name = "rules"

    def __init__(self, masker: M.PIIMasker | None = None):
        self.masker = masker or M.PIIMasker()
        if self.masker.name_model is not None:
            raise ValueError("RuleDetector is rules only; use MaskerDetector for a masker with a name model")

    def detect(self, text: str) -> list[Span]:
        return [Span(s, e, t, "rules") for s, e, t in self.masker.spans(text)]


class MaskerDetector:
    """Exactly what a configured PIIMasker.mask() replaces (rules + optional name model): the production path."""

    def __init__(self, masker: M.PIIMasker, name: str | None = None):
        self.masker = masker
        model = getattr(masker.name_model, "name", None)
        self.name = name or (f"masker[rules+{model}]" if model else "masker[rules]")

    def detect(self, text: str) -> list[Span]:
        return [Span(s, e, t, "masker") for s, e, t in self.masker.spans(text)]


def _locate(text: str, value: str) -> Iterable[tuple[int, int]]:
    start = 0
    while value and (i := text.find(value, start)) != -1:
        yield i, i + len(value)
        start = i + len(value)


class ModelDetector:
    """predict_fn(text) -> iterable of dicts with ('start','end','label') or ('text','label'), or (label, text) pairs."""

    def __init__(self, name: str, predict_fn: Callable[[str], Iterable[Any]], label_map: dict[str, str],
                 keep_types: Iterable[str] | None = None):
        self.name = name
        self.predict_fn = predict_fn
        self.label_map = {k.lower(): v for k, v in label_map.items()}
        self.keep_types = set(keep_types) if keep_types else None

    def _type(self, label: str) -> str | None:
        lab = str(label).lower()
        lab = re.sub(r"^[bi]-", "", lab)
        t = self.label_map.get(lab)
        if t and (self.keep_types is None or t in self.keep_types):
            return t
        return None

    def detect(self, text: str) -> list[Span]:
        out: list[Span] = []
        for ent in self.predict_fn(text) or []:
            if isinstance(ent, dict):
                label = ent.get("label") or ent.get("entity_group") or ent.get("entity") or ""
                t = self._type(label)
                if not t:
                    continue
                if ent.get("start") is not None and ent.get("end") is not None:
                    out.append(Span(int(ent["start"]), int(ent["end"]), t, self.name))
                elif ent.get("text"):
                    out.extend(Span(s, e, t, self.name) for s, e in _locate(text, ent["text"]))
            else:  # (label, value) pair
                label, value = ent
                t = self._type(label)
                if t:
                    out.extend(Span(s, e, t, self.name) for s, e in _locate(text, str(value)))
        return out


class HybridDetector:
    """Rules first, then model spans of the chosen types; a character is masked if either flags it."""

    def __init__(self, rules: RuleDetector, model: ModelDetector, model_types: Iterable[str] = ("name",)):
        self.rules, self.model = rules, model
        self.model_types = set(model_types)
        self.name = f"rules+{model.name}[{','.join(sorted(self.model_types))}]"

    def detect(self, text: str) -> list[Span]:
        spans = list(self.rules.detect(text))
        spans += [s for s in self.model.detect(text) if s.type in self.model_types]
        return spans


# ------------------------------------------------------------------ model factories (Colab)
def hf_token_classification(model_id: str, label_map: dict[str, str], *, name: str | None = None,
                            device: int = -1, keep_types: Iterable[str] | None = None) -> ModelDetector:
    """Any Hugging Face token-classification (NER) model, e.g. CAMeL-Lab/bert-base-arabic-camelbert-da-ner."""
    from transformers import pipeline  # imported lazily: Colab only

    ner = pipeline("token-classification", model=model_id, aggregation_strategy="simple", device=device)
    return ModelDetector(name or model_id.split("/")[-1], lambda t: ner(t), label_map, keep_types)


def gliner_detector(model_id: str, labels: dict[str, str], *, threshold: float = 0.5, name: str | None = None,
                    keep_types: Iterable[str] | None = None) -> ModelDetector:
    """GLiNER models (pip install gliner). `labels` maps the prompt label shown to the model -> our type."""
    from gliner import GLiNER  # imported lazily: Colab only

    model = GLiNER.from_pretrained(model_id)
    prompts = list(labels)
    return ModelDetector(name or model_id.split("/")[-1],
                         lambda t: model.predict_entities(t, prompts, threshold=threshold), labels, keep_types)


def gliner2_detector(model_id: str, labels: dict[str, str], *, name: str | None = None,
                     keep_types: Iterable[str] | None = None) -> ModelDetector:
    """GLiNER2 models (pip install gliner2). Output formats vary by version, so all known shapes are accepted."""
    from gliner2 import GLiNER2  # imported lazily: Colab only

    model = GLiNER2.from_pretrained(model_id)
    prompts = list(labels)

    def predict(text: str):
        out = model.extract_entities(text, prompts)
        ents = out.get("entities", out) if isinstance(out, dict) else out
        items = []
        if isinstance(ents, dict):
            for label, values in ents.items():
                for v in values or []:
                    if isinstance(v, dict):
                        items.append({**v, "label": label})
                    else:
                        items.append((label, v))
        elif isinstance(ents, list):
            items = ents
        return items

    return ModelDetector(name or model_id.split("/")[-1], predict, labels, keep_types)
