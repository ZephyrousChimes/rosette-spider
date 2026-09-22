"""Kaggle runner: clone the repo, fine-tune t5-small on the "simple" slice of
Spider train (join/nested/HAVING-free, 3420 examples), then evaluate on
Spider dev and the clause ladder. Writes results to /kaggle/working so
`kaggle kernels output` can retrieve them.
"""
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/ZephyrousChimes/rosette-spider.git"
WORK = Path("/kaggle/working")
REPO = WORK / "rosette-spider"

if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)

subprocess.run(
    ["python", "-u", "src/finetune.py", "--stage", "simple", "--epochs", "3", "--batch_size", "32"],
    cwd=REPO, check=True,
)
subprocess.run(
    ["python", "-u", "src/eval_finetuned.py", "--model_dir", str(REPO / "artifacts" / "model_simple"), "--label", "simple"],
    cwd=REPO, check=True,
)
print("done:", sorted(p.name for p in (REPO / "artifacts").glob("*simple*")))
