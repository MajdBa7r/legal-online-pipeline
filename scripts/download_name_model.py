"""Download CAMeLBERT-da NER once and check that it loads and runs on this machine (CPU).

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements-ml.txt
    python scripts/download_name_model.py

What it prints: where the files are, the commit hash (to pin later with PII_NAME_MODEL_REVISION), size on disk,
load time, memory, the model's labels, and the masked output of three synthetic sentences.
This is a smoke check, NOT an evaluation: accuracy is measured with scripts/eval_pii.py.
Nothing here sends question text anywhere; only the model files are downloaded from Hugging Face.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402

SENTENCES = [  # synthetic, written for this check; not part of any evaluation set
    "تواصلت مع براء الفلاحات من الموارد البشرية وما رد علي",
    "المالك بدو يطلعني من الشقة قبل ما يخلص العقد ورقمه 0791234567",
    "قام صاحب العمل غيث القيسي بإنهاء عقدي دون سابق إنذار",
]


def rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / 2**20


def main():
    s = Settings.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=s.pii_name_model_id)
    ap.add_argument("--revision", default=s.pii_name_model_revision or None)
    args = ap.parse_args()

    try:
        import torch
        import transformers
        from huggingface_hub import snapshot_download
    except ImportError as e:
        sys.exit(f"Missing ML packages ({e.name}). Run:\n"
                 "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                 "  pip install -r requirements-ml.txt")
    print(f"torch {torch.__version__} (CUDA available: {torch.cuda.is_available()}; the app uses CPU), "
          f"transformers {transformers.__version__}")

    print(f"\n1) downloading {args.model_id} (revision: {args.revision or 'default'}) ...")
    t0 = time.perf_counter()
    path = Path(snapshot_download(args.model_id, revision=args.revision))
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20
    print(f"   files:  {path}")
    print(f"   commit: {path.name}   <- to pin, set PII_NAME_MODEL_REVISION={path.name}")
    print(f"   size:   {size:.0f} MB  ({time.perf_counter() - t0:.1f} s)")

    print("\n2) loading on CPU ...")
    from app.pii.masker import PIIMasker
    from app.pii.name_model import load_name_model

    before = rss_mb()
    model = load_name_model("camelbert-da", model_id=args.model_id, min_score=0.0, revision=args.revision)
    after = rss_mb()
    print(f"   loaded in {model.load_seconds:.1f} s; labels: {model.labels}")
    if before is not None:
        print(f"   process memory: {before:.0f} MB before, {after:.0f} MB after (+{after - before:.0f} MB)")

    print("\n3) smoke check on synthetic sentences (min_score=0.0, family expansion off):")
    masker = PIIMasker(name_model=model)
    for text in SENTENCES:
        t0 = time.perf_counter()
        r = masker.mask(text)
        ms = 1000 * (time.perf_counter() - t0)
        raw = [(text[a:b], round(sc, 3)) for a, b, sc in model.raw_spans(text)]
        print(f"   in:   {text}\n   out:  {r.text}\n   raw PERS spans: {raw}\n   {ms:.0f} ms\n")
    print("Done. This was a smoke check only; run scripts/eval_pii.py for the development-set numbers.")


if __name__ == "__main__":
    main()
