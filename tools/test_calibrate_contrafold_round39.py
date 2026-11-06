import numpy as np
import pytest

from calibrate_contrafold_round39 import parse_bpseq, parse_posteriors


def test_parse_native_bpseq_and_posteriors(tmp_path):
    sequence = "AUGCAU"
    structure = tmp_path / "structure.bpseq"
    structure.write_text("1 A 6\n2 U 5\n3 G 0\n4 C 0\n5 A 2\n6 U 1\n")
    assert parse_bpseq(structure, sequence) == [(0, 5), (1, 4)]
    posteriors = tmp_path / "posteriors.bpseq"
    posteriors.write_text("1 A 6:0.75\n2 U 5:0.5 6:0.25\n3 G\n4 C\n5 A 2:0.5\n6 U 1:0.75 2:0.25\n")
    matrix = parse_posteriors(posteriors, sequence)
    assert matrix[0, 5] == matrix[5, 0] == 0.75
    assert matrix[1, 4] == matrix[4, 1] == 0.5
    assert matrix[1, 5] == matrix[5, 1] == 0.25
    assert np.isfinite(matrix).all() and matrix.max() <= 1


def test_posteriors_reject_invalid_probability(tmp_path):
    path = tmp_path / "posteriors.bpseq"
    path.write_text("1 A 2:1.1\n2 U 1:1.1\n")
    with pytest.raises(AssertionError):
        parse_posteriors(path, "AU")
