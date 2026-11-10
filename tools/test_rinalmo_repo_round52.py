from pathlib import Path

import rinalmo_preflight
import train_plan_a


repo = Path(train_plan_a.__file__).resolve().parents[1]
assert train_plan_a.REPO == repo
assert Path(rinalmo_preflight.__file__).resolve() == repo / "tools/rinalmo_preflight.py"
assert Path(rinalmo_preflight.WEIGHTS).is_dir()
assert (Path(rinalmo_preflight.WEIGHTS) / "model.safetensors").is_file()
print(f"PASS RiNALMo repository and official weight identity: {repo}")
