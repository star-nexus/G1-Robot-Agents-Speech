from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_shared_deployment_assets_are_vendor_and_host_neutral():
    shared_files = (
        "README.md",
        "README_ZH.md",
        "deploy.env.example",
        "config.example.json",
        "config.qwen3-asr.example.json",
    )
    forbidden = (
        "unitree_sdk",
        "unitree g1",
        "宇树 g1",
        "galbot g1",
        "银河通用 g1",
        "orin_host",
        "spark_host",
        "/home/nvidia",
        "wlp1p1s0",
    )

    for relative_path in shared_files:
        content = (ROOT / relative_path).read_text(encoding="utf-8").lower()
        for marker in forbidden:
            assert marker not in content, f"{marker!r} leaked into {relative_path}"


def test_host_local_and_benchmark_artifacts_are_ignored():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "deploy.env" in ignore
    assert "config.*.local.json" in ignore
    assert "acceptance/data/" in ignore
    assert "acceptance/results/" in ignore
    assert "acceptance/*.local.jsonl" in ignore
    assert "/tests/test_audio_*" in ignore
