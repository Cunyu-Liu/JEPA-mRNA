from types import SimpleNamespace

from rnajepa.train_decision import gradient_exclusions


def test_pairmlp_without_latent_objective_excludes_training_predictor():
    excluded = gradient_exclusions(SimpleNamespace(lambda_jepa=0.0))
    assert "head.pair_jepa.context_predictor" in excluded


def test_latent_objective_keeps_predictor_covered():
    excluded = gradient_exclusions(SimpleNamespace(lambda_jepa=0.1))
    assert "head.pair_jepa.context_predictor" not in excluded
