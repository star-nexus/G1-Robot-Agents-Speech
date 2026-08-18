#!/usr/bin/env python3
"""Capture reproducible host/process/GPU memory data on Jetson.

The script deliberately reports process PSS/RSS and Jetson GPU mappings
separately: on an integrated-memory Jetson they overlap and must not be added.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any


def read_key_values(path: Path) -> dict[str, int | str]:
    result: dict[str, int | str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return result
    for line in lines:
        key, separator, raw_value = line.partition(":")
        if not separator:
            continue
        parts = raw_value.strip().split()
        if parts and parts[0].isdigit():
            value = int(parts[0])
            if len(parts) > 1 and parts[1].lower() == "kb":
                result[f"{key}_kib"] = value
            else:
                result[key] = value
        else:
            result[key] = raw_value.strip()
    return result


def read_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [item.decode("utf-8", errors="replace") for item in raw.split(b"\0") if item]


def read_process(pid: int, *, include_smaps: bool = True) -> dict[str, Any] | None:
    status = read_key_values(Path(f"/proc/{pid}/status"))
    if not status:
        return None
    cmdline = read_cmdline(pid)
    smaps = read_key_values(Path(f"/proc/{pid}/smaps_rollup")) if include_smaps else {}
    return {
        "pid": pid,
        "ppid": int(status.get("PPid", 0)),
        "name": status.get("Name", ""),
        "cmdline": cmdline,
        "rss_kib": int(status.get("VmRSS_kib", 0)),
        "rss_anon_kib": int(status.get("RssAnon_kib", 0)),
        "rss_file_kib": int(status.get("RssFile_kib", 0)),
        "rss_shmem_kib": int(status.get("RssShmem_kib", 0)),
        "vm_size_kib": int(status.get("VmSize_kib", 0)),
        "swap_kib": int(status.get("VmSwap_kib", 0)),
        "pss_kib": int(smaps.get("Pss_kib", 0)),
        "private_clean_kib": int(smaps.get("Private_Clean_kib", 0)),
        "private_dirty_kib": int(smaps.get("Private_Dirty_kib", 0)),
        "shared_clean_kib": int(smaps.get("Shared_Clean_kib", 0)),
        "shared_dirty_kib": int(smaps.get("Shared_Dirty_kib", 0)),
    }


def serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return serializable(dict(value))
    except (TypeError, ValueError):
        return str(value)


def read_jtop() -> dict[str, Any]:
    try:
        from jtop import jtop  # type: ignore[import-not-found]
    except ImportError as exc:
        return {"available": False, "error": str(exc)}

    jetson = jtop()
    try:
        jetson.start()
        if not jetson.ok():
            return {"available": False, "error": "jtop service is not ready"}
        return {
            "available": True,
            "stats": serializable(jetson.stats),
            "memory": serializable(jetson.memory),
            "processes": serializable(jetson.processes),
        }
    except Exception as exc:  # diagnostic utility: retain partial failure
        return {"available": False, "error": repr(exc)}
    finally:
        jetson.close()


def capture_processes(patterns: list[str], *, include_smaps: bool = True) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        process = read_process(int(entry.name), include_smaps=include_smaps)
        if process is None:
            continue
        searchable = " ".join(process["cmdline"] or [str(process["name"])]).lower()
        if any(pattern in searchable for pattern in patterns):
            processes.append(process)
    processes.sort(key=lambda item: int(item["pid"]))
    return processes


def capture_sample(patterns: list[str], *, include_smaps: bool = False) -> dict[str, Any]:
    processes = capture_processes(patterns, include_smaps=include_smaps)
    meminfo = read_key_values(Path("/proc/meminfo"))
    total_kib = int(meminfo.get("MemTotal_kib", 0))
    available_kib = int(meminfo.get("MemAvailable_kib", 0))
    return {
        "timestamp_unix": time.time(),
        "used_excluding_available_kib": total_kib - available_kib,
        "mem_available_kib": available_kib,
        "selected_rss_kib": sum(int(item["rss_kib"]) for item in processes),
        # smaps_rollup is deliberately skipped for high-frequency samples. A
        # missing measurement must not look like a real zero-PSS process tree.
        "selected_pss_kib": (
            sum(int(item["pss_kib"]) for item in processes) if include_smaps else None
        ),
        "selected_swap_kib": sum(int(item["swap_kib"]) for item in processes),
        "processes": processes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        help="case-insensitive cmdline substring; repeatable",
    )
    parser.add_argument("--no-jtop", action="store_true")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()
    patterns = [item.lower() for item in (args.match or ["vllm", "qwen3-tts", "g1-speech", "omni"])]

    if args.samples < 1:
        parser.error("--samples must be at least 1")
    samples: list[dict[str, Any]] = []
    for index in range(args.samples):
        samples.append(capture_sample(patterns, include_smaps=False))
        if index + 1 < args.samples:
            time.sleep(args.interval)
    processes = capture_processes(patterns, include_smaps=True)
    meminfo = read_key_values(Path("/proc/meminfo"))
    total_kib = int(meminfo.get("MemTotal_kib", 0))
    available_kib = int(meminfo.get("MemAvailable_kib", 0))
    result = {
        "schema_version": 2,
        "label": args.label,
        "timestamp_unix": time.time(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": {"executable": sys.executable, "version": sys.version},
        "patterns": patterns,
        "system_memory": {
            "meminfo": meminfo,
            "used_excluding_available_kib": total_kib - available_kib,
        },
        "processes": processes,
        "selected_process_totals": {
            "rss_kib": sum(int(item["rss_kib"]) for item in processes),
            "pss_kib": sum(int(item["pss_kib"]) for item in processes),
            "swap_kib": sum(int(item["swap_kib"]) for item in processes),
        },
        "timeseries": [
            {key: value for key, value in sample.items() if key != "processes"}
            for sample in samples
        ],
        "timeseries_peaks": {
            "used_excluding_available_kib": max(sample["used_excluding_available_kib"] for sample in samples),
            "selected_rss_kib": max(sample["selected_rss_kib"] for sample in samples),
        },
        "jtop": {"available": False, "skipped": True} if args.no_jtop else read_jtop(),
        "notes": [
            "PSS is preferred for process-tree attribution because RSS double-counts shared pages.",
            "Jetson GPU memory uses unified physical RAM and overlaps process/system memory; do not add it to PSS.",
            "System used_excluding_available is MemTotal minus MemAvailable, not a TTS-only measurement.",
            "High-frequency timeseries PSS is null because reading smaps_rollup on every sample would perturb the workload; use selected_process_totals.pss_kib from the final snapshot.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
