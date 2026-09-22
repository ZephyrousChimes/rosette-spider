"""Kaggle runner: clone the repo, execute rosette_spider.ipynb with nbconvert,
and save the executed notebook (with outputs) plus an HTML render to
/kaggle/working, where `kaggle kernels output` can download them.

The Kaggle API returns only files a run writes, not the rendered notebook
itself -- hence this runner.
"""
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/ZephyrousChimes/rosette-spider.git"
WORK = Path("/kaggle/working")
REPO = WORK / "rosette-spider"

if not REPO.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)
subprocess.run(
    ["jupyter", "nbconvert", "--to", "notebook", "--execute", "rosette_spider.ipynb",
     "--output", str(WORK / "rosette_spider_executed.ipynb"),
     "--ExecutePreprocessor.timeout=3600", "--ExecutePreprocessor.kernel_name=python3"],
    cwd=REPO, check=True,
)
subprocess.run(["jupyter", "nbconvert", "--to", "html", str(WORK / "rosette_spider_executed.ipynb"),
                "--output", str(WORK / "rosette_spider_executed.html")], check=True)
print("done:", sorted(p.name for p in WORK.iterdir()))
