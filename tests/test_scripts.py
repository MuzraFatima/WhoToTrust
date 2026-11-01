"""Shell scripts are only exercised on Kaggle, so check here that they parse and that every
Hydra override they pass still composes."""

import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_module

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.parametrize("name", ["kaggle_bootstrap.sh", "smoke_gpu.sh", "reproduce_headline.sh"])
def test_shell_scripts_parse(name):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    subprocess.run([bash, "-n", str(SCRIPTS / name)], check=True)


def test_every_smoke_step_composes():
    text = (SCRIPTS / "smoke_gpu.sh").read_text()
    env = dict(re.findall(r'^(\w+)="([^"]*)"$', text, flags=re.M))
    steps = [line for line in text.splitlines() if line.startswith("step ")]
    assert len(steps) >= 10
    with initialize_config_module(config_module="labeltriage.conf", version_base="1.3"):
        for line in steps:
            parts = shlex.split(re.sub(r"\$(\w+)", lambda m: env[m.group(1)], line))
            compose(config_name="config", overrides=parts[2:])
