"""Packs suite_a/ into a single self-contained Kaggle script in kaggle_suite_a/."""
import os
import base64

root = "/home/manoj/Projects/Research/learning_world_model"
pkg_dir = os.path.join(root, "suite_a")
out_dir = os.path.join(root, "kaggle_suite_a")
os.makedirs(out_dir, exist_ok=True)

files = ["__init__.py", "config.py", "envs.py", "models.py", "eval.py", "train.py", "test_suite_a.py", "run_kaggle_suite_a.py"]
packed = {}
for fname in files:
    with open(os.path.join(pkg_dir, fname), "rb") as f:
        packed[fname] = base64.b64encode(f.read()).decode("ascii")

runner_template = '''"""Auto-generated self-contained Suite A Kaggle Runner (Phase 1 + Phase 2)."""
import os
import sys
import base64
import subprocess

EMBEDDED_FILES = {packed_repr}

os.makedirs("suite_a", exist_ok=True)
for fname, b64_str in EMBEDDED_FILES.items():
    with open(os.path.join("suite_a", fname), "wb") as f:
        f.write(base64.b64decode(b64_str))
print(f"Unpacked {len(EMBEDDED_FILES)} files into suite_a/")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gymnasium", "pytest"], check=True)

print("--- Running Suite A Pre-Flight Unit & Law Tests ---")
test_res = subprocess.run([sys.executable, "-m", "pytest", "suite_a/test_suite_a.py", "-v"])
if test_res.returncode != 0:
    print("Suite A verification failed! Aborting.")
    sys.exit(1)

print("--- Launching Suite A Benchmark (Phase 1 + Phase 2) ---")
res = subprocess.run([sys.executable, "-m", "suite_a.run_kaggle_suite_a"])
sys.exit(res.returncode)
'''

out_path = os.path.join(out_dir, "run_suite_a_kaggle.py")
with open(out_path, "w") as f:
    f.write(runner_template.replace("{packed_repr}", repr(packed)))

print(f"Built self-contained Kaggle bundle: {out_path}")
