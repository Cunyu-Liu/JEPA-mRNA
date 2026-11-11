import json

from analyze_rinalmo_jmlp_confidence_round52 import json_safe


value = {"risk": [0.1, float("nan"), None], "nested": {"aurc": float("nan")}}
safe = json_safe(value)
assert safe == {"risk": [0.1, None, None], "nested": {"aurc": None}}
assert len(safe["risk"]) == 3
json.dumps(safe, allow_nan=False)
try:
    json.dumps(value, allow_nan=False)
except ValueError:
    pass
else:
    raise AssertionError("unsafe NaN serialization accepted")
print("PASS empty confidence buckets remain null with strict JSON; no bucket deletion")
