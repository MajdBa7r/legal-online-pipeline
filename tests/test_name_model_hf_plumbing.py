"""Our loader with the REAL transformers library and a tiny, randomly built local model (no download).

This checks plumbing only: the pipeline call, character offsets, windows on texts longer than 512 pieces,
label checks, and that the output goes through post-processing. The tiny model knows nothing about names.
Skipped when torch/transformers are not installed (they are optional: requirements-ml.txt).
"""
import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from app.pii.masker import PIIMasker  # noqa: E402
from app.pii.name_model import CamelbertNameModel, NameModelUnavailable, load_name_model  # noqa: E402
from app.pii.name_postprocess import is_subword_span  # noqa: E402

LABELS = ["O", "B-PERS", "I-PERS", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
LETTERS = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"


def _build(tmp_path, labels=LABELS, favour="O"):
    from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list(LETTERS) + ["##" + c for c in LETTERS] \
        + list("0123456789،.,؟?!:") + ["##" + d for d in "0123456789"]
    d = tmp_path / f"tiny-{favour}"
    d.mkdir()
    (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(d / "vocab.txt"), do_lower_case=False, strip_accents=False,
                            tokenize_chinese_chars=False)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=512,
                     id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
    torch.manual_seed(0)
    model = BertForTokenClassification(cfg)
    with torch.no_grad():                       # make the output deterministic: always predict `favour`
        model.classifier.weight.zero_()
        model.classifier.bias.fill_(-10.0)
        model.classifier.bias[labels.index(favour)] = 10.0
    tok.save_pretrained(d)
    model.save_pretrained(d)
    return str(d)


def test_loads_a_local_model_and_reports_its_labels(tmp_path):
    m = load_name_model("camelbert-da", model_id=_build(tmp_path), min_score=0.0)
    assert isinstance(m, CamelbertNameModel)
    assert "PERS" in m.labels and m.load_seconds >= 0


def test_model_without_a_person_label_is_rejected(tmp_path):
    path = _build(tmp_path, labels=["O", "B-LOC", "I-LOC"])
    with pytest.raises(NameModelUnavailable, match="no person label"):
        load_name_model("camelbert-da", model_id=path, min_score=0.0)


def test_missing_local_model_is_unavailable(tmp_path):
    with pytest.raises(NameModelUnavailable):
        load_name_model("camelbert-da", model_id=str(tmp_path / "does-not-exist"), min_score=0.0)


def test_all_o_model_masks_exactly_like_the_rules(tmp_path):
    m = load_name_model("camelbert-da", model_id=_build(tmp_path, favour="O"), min_score=0.0)
    text = "اسمي محمد الزعبي ورقمي 0791234567، المالك بدو يطلعني"
    assert m.detect_names(text) == []
    assert PIIMasker(name_model=m).mask(text).text == PIIMasker().mask(text).text


def test_offsets_from_the_real_pipeline_map_to_whole_words(tmp_path):
    m = load_name_model("camelbert-da", model_id=_build(tmp_path, favour="B-PERS"), min_score=0.0)
    text = "قال المالك براء، رقمه 0791234567 وبيته باربد"
    raw = m.raw_spans(text)
    assert raw and all(0 <= s < e <= len(text) for s, e, _ in raw)
    spans = m.detect_names(text)
    for s, e in spans:
        assert not is_subword_span(text, s, e)
        assert not any(ch.isdigit() for ch in text[s:e]) and "،" not in text[s:e]
        assert text[s:e] != "المالك"                         # exclusion trimming ran on real output
    r = PIIMasker(name_model=m).mask(text)
    assert "[هاتف_1]" in r.text                              # structured rules unchanged
    assert PIIMasker.restore(r.text, r.mapping) == text


def test_text_longer_than_the_model_limit_runs_in_windows(tmp_path):
    m = load_name_model("camelbert-da", model_id=_build(tmp_path, favour="B-PERS"), min_score=0.0)
    text = " ".join(["الشقة"] * 900) + " براء"             # far more than 512 pieces
    assert len(m.piece_offsets(text)) > 512
    wins = m.windows(text)
    assert len(wins) >= 2
    assert all(len(m.piece_offsets(text[a:b])) <= 510 for a, b in wins)
    spans = m.detect_names(text)                             # would raise in the model without windows
    assert (len(text) - 4, len(text)) in spans               # the last word is reached
