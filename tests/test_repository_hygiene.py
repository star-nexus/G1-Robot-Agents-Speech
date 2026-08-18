from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_shared_deployment_assets_are_vendor_and_host_neutral():
    shared_files = (
        "README.md",
        "README_ZH.md",
        "deploy.env.example",
        "configs/examples/config.example.json",
        "configs/examples/config.qwen3-asr.example.json",
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
    assert "tests/acceptance/data/" in ignore
    assert "tests/acceptance/results/" in ignore
    assert "tests/acceptance/*.local.jsonl" in ignore
    assert "/tests/test_audio_*" in ignore


def _deployment_fields(path: Path) -> list[str]:
    fields = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields.append(stripped.partition("=")[0])
    return fields


def test_deployment_example_is_model_neutral_and_matches_local_schema_when_present():
    example = ROOT / "deploy.env.example"
    fields = _deployment_fields(example)

    assert len(fields) == len(set(fields))
    assert "SPEECH_CONFIG_CPU" in fields
    assert "SPEECH_CONFIG_GPU" in fields
    assert "SPEECH_GPU_LIBRARY_PATH" in fields
    assert "AUDIO_INPUT_BACKEND" in fields
    assert "ALSA_INPUT_CARD" in fields
    assert "PULSE_INPUT_DEVICE" in fields
    assert "PULSE_SOURCE" in fields
    assert "MICROPHONE_DEVICE" not in fields
    assert "ASR_BACKEND" not in fields
    assert not any(name.startswith("SENSEVOICE_") for name in fields)
    assert not any(name.startswith("QWEN3_ASR_") for name in fields)

    local = ROOT / "deploy.env"
    if local.exists():
        assert _deployment_fields(local) == fields


def test_example_profiles_are_grouped_and_root_qwen_profiles_are_local():
    example = ROOT / "configs/examples/config.qwen3-asr.example.json"
    root_names = {path.name for path in ROOT.glob("config.qwen3*.json")}

    assert example.is_file()
    assert root_names <= {"config.qwen3-asr.local.json"}
