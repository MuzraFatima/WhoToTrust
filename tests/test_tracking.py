from omegaconf import OmegaConf

from labeltriage.utils import tracking


def test_disabled_tracking_is_a_no_op():
    cfg = OmegaConf.create({"tracking": {"wandb": False}})
    assert tracking.log_result(cfg, {"acc": 0.9}) is False


def test_flatten_keeps_scalars_only():
    flat = tracking._flatten({"a": 1, "b": {"c": 2.0, "d": [1, 2]}, "e": "x"})
    assert flat == {"a": 1, "b/c": 2.0, "e": "x"}
