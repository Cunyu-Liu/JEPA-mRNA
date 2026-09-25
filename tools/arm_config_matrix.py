import json

ARMS = {
    "ff_b4_s0": "rinalmo_ff_b4_s0",
    "ff_b4_s1": "rinalmo_ff_b4_s1",
    "ff_tr1_b4_s0": "rinalmo_ff_tr1_b4_s0",
    "big_b4_s0": "rinalmo_big_b4_s0",
    "bigtr1_b4_s0": "rinalmo_bigtr1_b4_s0",
    "len_b4_s0": "rinalmo_len_b4_s0",
    "sum_b4_s0": "rinalmo_sum_b4_s0",
    "bal_b4_s0": "rinalmo_bal_b4_s0",
    "pw_b4_s0": "rinalmo_pw_b4_s0",
    "bigtr1_b4_s1": "rinalmo_bigtr1_b4_s1",
    "big_b4_s3": "rinalmo_big_b4_s3",
    "ff_tr1_b4_s1": "rinalmo_ff_tr1_b4_s1",
}

print(f"{'arm':16s} {'norm':8s} {'d_z':5s} {'hidden':7s} {'corpus':26s} {'steps':7s} {'seed'}")
for label, arm in ARMS.items():
    try:
        m = json.load(open(f"/mnt/cunyuliu/rna-jepa/runs/{arm}/run_meta.json"))
    except FileNotFoundError:
        print(f"{label:16s} MISSING")
        continue
    cfg = m.get("config", m)
    norm = cfg.get("nll_normalization", "(absent->sum)")
    dz = cfg.get("d_z", "?")
    hid = cfg.get("hidden", "?")
    data = m.get("data", {}).get("source", "?").replace("jsonl:/mnt/cunyuliu/rna-jepa/ss_data/jsonl/", "")
    steps = cfg.get("steps", cfg.get("total_steps", "?"))
    seed = cfg.get("seed", "?")
    print(f"{label:16s} {norm:8s} {str(dz):5s} {str(hid):7s} {data:26s} {str(steps):7s} {seed}")
