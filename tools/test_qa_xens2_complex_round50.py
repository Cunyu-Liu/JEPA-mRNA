import math

from tools.qa_xens2_complex_round50 import micro_score, sha256


def test_micro_score_uses_precision_recall_harmonic_mean():
    precision, recall, f1 = micro_score(182117, 92271, 107728)
    assert precision == 182117 / (182117 + 92271)
    assert recall == 182117 / (182117 + 107728)
    assert math.isclose(f1, 2 * 182117 / (2 * 182117 + 92271 + 107728),
                        rel_tol=0, abs_tol=1e-15)


def test_micro_score_empty_is_zero():
    assert micro_score(0, 0, 0) == (0.0, 0.0, 0.0)


def test_sha256_accepts_string_paths(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text("{}")
    assert sha256(str(path)) == sha256(path)
