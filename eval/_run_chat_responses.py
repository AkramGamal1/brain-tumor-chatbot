"""One-shot runner: hit POST /chat for every prompt in eval/prompts.yaml,
save raw responses to eval/_chat_responses.json. Scoring happens by hand
afterward against the pass_criteria / fail_signals in prompts.yaml.

This script is not part of the regular harness — it exists for the manual
Phase 2 eval session and can be deleted afterward.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
import yaml

PROMPTS = Path(__file__).resolve().parent / "prompts.yaml"
OUT = Path(__file__).resolve().parent / "_chat_responses.json"
URL = "http://localhost:8001/chat"


def main() -> int:
    data = yaml.safe_load(PROMPTS.read_text(encoding="utf-8"))
    prompts = data["prompts"]
    print(f"Running {len(prompts)} prompts against {URL}")

    results = []
    with httpx.Client(timeout=60.0) as client:
        for i, p in enumerate(prompts, 1):
            pid = p["id"]
            msg = p["prompt"]
            print(f"[{i}/{len(prompts)}] {pid}", flush=True)
            try:
                t0 = time.time()
                resp = client.post(URL, json={"message": msg})
                dt = time.time() - t0
                results.append(
                    {
                        "id": pid,
                        "category": p["category"],
                        "gate": p["gate"],
                        "prompt": msg,
                        "status_code": resp.status_code,
                        "response_body": resp.json(),
                        "elapsed_seconds": round(dt, 2),
                    }
                )
            except Exception as e:
                results.append({"id": pid, "error": repr(e)})
                print(f"  ERROR: {e!r}", flush=True)
            # Persist after each call so a mid-run failure doesn't lose progress
            OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
