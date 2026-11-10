from probe_capacity_error_round51 import select_training_subset


records = [(f"SEQ{index}", [(0, 5)]) for index in range(20)]
first = select_training_subset(records, 8, 42)
second = select_training_subset(records, 8, 42)
other = select_training_subset(records, 8, 2026)
assert first == second and first != other
assert len(first) == len({seq for seq, _ in first}) == 8
assert all(record in records for record in first)
assert records == [(f"SEQ{index}", [(0, 5)]) for index in range(20)]
for size, seed in ((0, 42), (21, 42), (8, 0)):
    try:
        select_training_subset(records, size, seed)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid subset configuration must be rejected")
try:
    select_training_subset([("A", []), ("A", [])], 2, 42)
except ValueError:
    pass
else:
    raise AssertionError("duplicate training identities must be rejected")
print("PASS capacity-error subset reproducibility, boundaries, and identity")
