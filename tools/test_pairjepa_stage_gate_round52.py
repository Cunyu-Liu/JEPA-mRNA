from pathlib import Path


def check_text(text, prerequisite_count):
    gate = text.index("until test -f")
    training = text.index('"$PY" "$REPO/tools/train_plan_a.py"')
    assert text.index("base=", gate) < training
    assert text.count(".recheck.json") >= prerequisite_count
    assert text.index("cmp -s", gate) < training
    assert text.index("sleep 300", gate) < training


for filename, count in (("run_rinalmo_j1_block_ema_round52.sh", 2),
                        ("run_rinalmo_pairjepa_target_round52.sh", 1)):
    text = Path(__file__).with_name(filename).read_text()
    check_text(text, count)
    for broken in (text.replace(".recheck.json", ".missing_recheck.json"),
                   text.replace("cmp -s", "false")):
        try:
            check_text(broken, count)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"unsafe stage gate accepted: {filename}")
print("PASS J1/fixed/random recheck-and-cmp gates precede formal training; failure injection rejected")
