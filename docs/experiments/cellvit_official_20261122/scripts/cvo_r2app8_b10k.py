"""cellvit_official_20261122 (PREREG.md §4.3, sensitivity only, not a decision input): run the unchanged hfv_retrain2
r2_app8.py with the window bootstrap at 10,000 draws and seed 42 instead of app 8's 2,000 draws and seed 20261128.
r2_app8.py reads NB and SEED from app8_stats at import, so they are set on that module before r2_app8.py runs.

  python cvo_r2app8_b10k.py SEV R2_APP8_PY <the 10 r2_app8.py arguments after SEV>
"""
import runpy
import sys

sev, script, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, sev)
import app8_stats  # noqa: E402

app8_stats.NB, app8_stats.SEED = 10000, 42
sys.argv = [script, sev] + rest
runpy.run_path(script, run_name="__main__")
