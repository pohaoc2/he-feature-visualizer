import boto3, sys, os
from concurrent.futures import ThreadPoolExecutor
s3 = boto3.client("s3"); B = "bagherilab-working"; P = "pohao/share_space/04_inference/si_a1_a2/tiles/"
ids = [l.strip() for l in open(sys.argv[1]) if l.strip()]; out = sys.argv[2]
jobs = []
for v in ["production", "a2_off_shelf", "a1_concat", "a1_per_channel", "a2_bypass_full_tme", "gt"]:
    os.makedirs(f"{out}/{v}", exist_ok=True)
    for t in ids:
        jobs += [(v, f"{t}.png"), (v, f"{t}_cellvit_instances.json")]
def g(j):
    v, f = j
    try: s3.download_file(B, P + f"{v}/{f}", f"{out}/{v}/{f}"); return 1
    except Exception: return 0
with ThreadPoolExecutor(32) as ex: r = list(ex.map(g, jobs))
print("got", sum(r), "of", len(jobs))
