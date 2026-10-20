from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from .common import load_config, safe_output


def main():
    parser = argparse.ArgumentParser(description="Isolated 19_MM synchronization feasibility study")
    parser.add_argument("--config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Read-only input audit and deterministic dataset preparation")
    plan = sub.add_parser("plan", help="List training jobs without running them")
    plan.add_argument("--suite", choices=["core", "temporal", "smoke"], default="core")
    train = sub.add_parser("train", help="Run a resumable suite, or a single job")
    train.add_argument("--suite", choices=["core", "temporal", "smoke"], default="core")
    train.add_argument("--job", type=int, help="Zero-based job index from plan")
    train.add_argument("--max-jobs", type=int, help="Limit this invocation (completed jobs do not count)")
    sub.add_parser("report", help="Rebuild report from completed runs; no training")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        from .prepare import prepare
        from .train import load_prepared
        from .report import report
        prepare(config)
        data, manifest = load_prepared(config)
        report(config, data, manifest)
    elif args.command == "plan":
        from .train import make_plan, job_id
        jobs = make_plan(config, args.suite)
        for i, job in enumerate(jobs):
            print(f"{i:3d}  {job_id(job)}")
        print(f"{len(jobs)} jobs. Run smoke first to check hardware; smoke timings are not convergence estimates.")
    elif args.command == "train":
        # Required by deterministic CUDA matrix multiplication, set before importing torch.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        from .train import load_prepared, make_plan, train_job, job_id
        from .report import report
        data, manifest = load_prepared(config)
        jobs = make_plan(config, args.suite)
        if args.job is not None:
            if not 0 <= args.job < len(jobs):
                parser.error("--job is outside the plan")
            jobs = [jobs[args.job]]
        done = 0
        for job in jobs:
            folder = safe_output(config["output"]) / ("smoke_runs" if job["smoke"] else "runs") / job_id(job)
            exists = (folder / "result.json").exists()
            train_job(data, manifest, config, job)
            if not exists:
                done += 1
            if args.suite != "smoke":
                report(config, manifest=manifest)
            if args.max_jobs is not None and done >= args.max_jobs:
                break
        report(config, data, manifest)
    elif args.command == "report":
        from .report import report
        report(config)
