"""Serve the decision model on localhost for `scout --router local`.

    uv run python serve.py --run out/run [--port 8765]

POST /decide with a question set ({"state": ..., "questions": {...}}) answers every question
in one batch: {"model": ..., "latency_ms": ..., "answers": {key: {option key: probability}}}.
Probabilities use the temperature fitted at training time. Listens on 127.0.0.1 only.
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch

from local_router.model import Scorer
from local_router.prompts import MAX_OPTIONS, options


class Decider:
    def __init__(self, run):
        run = Path(run)
        self.temperature = json.loads((run / "metrics.json").read_text())["temperature"]
        self.scorer = Scorer(adapter=run / "adapter")
        self.scorer.model.eval()
        # scout checks this name against SCOUT_LOCAL_MODEL.
        self.name = run.name

    @torch.no_grad()
    def decide(self, question_set):
        started = time.perf_counter()
        state, questions = question_set["state"], question_set["questions"]
        keys = sorted(questions)
        orders = [options(questions[k]) for k in keys]
        if any(len(o) > MAX_OPTIONS for o in orders):
            raise ValueError(f"a question has more than {MAX_OPTIONS} options")
        logits = self.scorer.set_logits([(state, questions[k], o) for k, o in zip(keys, orders)],
                                        [len(o) for o in orders])
        probs = torch.softmax(logits / self.temperature, dim=-1).cpu()
        answers = {k: {key: round(float(p), 6) for (key, _), p in zip(o, row)}
                   for k, o, row in zip(keys, orders, probs)}
        return {"model": self.name, "latency_ms": round((time.perf_counter() - started) * 1000),
                "answers": answers}


def handler(decider):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/decide":
                return self.reply(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                return self.reply(200, decider.decide(body))
            except (ValueError, KeyError, TypeError) as e:
                return self.reply(400, {"error": str(e)})

        def reply(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="training output directory (adapter/ and metrics.json)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    decider = Decider(args.run)
    print(f"serving {decider.name} (temperature {decider.temperature}) on 127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), handler(decider)).serve_forever()


if __name__ == "__main__":
    main()
