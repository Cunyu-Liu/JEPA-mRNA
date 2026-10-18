"""Run eval_rinalmo_ft with use_flash_attn forced False (RiNALMo-native path).
Invoked by scripts/rinalmoft_4090_noflash.sh with argv set."""
import sys
from pathlib import Path
sys.path.insert(0, "/root/rna_jev/tools/RiNALMo-main")
sys.path.insert(0, "/root/rna_jev/tools")
import rinalmo.config as rc

_orig = rc.model_config
def patched(name):
    c = _orig(name)
    if isinstance(c, dict) and "model" in c:
        try:
            c["model"]["transformer"]["use_flash_attn"] = False
        except Exception:
            pass
    return c
rc.model_config = patched

# also patch the loader's HAS_FLASH gate if it imported
try:
    import eval_rinalmo_ft as erf
    erf.HAS_FLASH = False
except Exception:
    pass

import runpy
sys.argv = ["eval_rinalmo_ft.py"] + sys.argv[1:]
runpy.run_path("/root/rna_jev/tools/eval_rinalmo_ft.py", run_name="__main__")
