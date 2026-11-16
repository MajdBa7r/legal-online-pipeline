"""Send questions to a running server and print type, domain and time.
Also warms the demo cache when the server runs with DEMO_CACHE=1.

    python scripts/smoke_test.py --url https://<your-repl>.replit.app --file questions.txt
    python scripts/smoke_test.py "كم يوم إجازة سنوية بستحق؟"

questions.txt: one question per line.
"""
import argparse
import json
import os
import time

import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="*")
    ap.add_argument("--url", default=os.environ.get("API_URL", "http://localhost:8000"))
    ap.add_argument("--file")
    ap.add_argument("--eval-token", default=os.environ.get("EVAL_TOKEN", ""))
    ap.add_argument("--show", action="store_true", help="print the full JSON response")
    args = ap.parse_args()

    qs = list(args.questions)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            qs += [line.strip() for line in f if line.strip()]
    headers = {"X-Eval-Token": args.eval_token} if args.eval_token else {}

    with httpx.Client(timeout=40) as c:
        for q in qs:
            t0 = time.monotonic()
            r = c.post(f"{args.url}/api/ask", json={"question": q}, headers=headers)
            ms = int((time.monotonic() - t0) * 1000)
            data = r.json()
            print(f"{ms:6d} ms  {data.get('type'):13s} {str(data.get('domain')):20s} {q[:60]}")
            if args.show:
                print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
