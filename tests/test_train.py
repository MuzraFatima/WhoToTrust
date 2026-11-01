import numpy as np
import pytest

torch = pytest.importorskip("torch")

from labeltriage.targets import TargetSet  # noqa: E402
from labeltriage.train import image, loop, text  # noqa: E402
from labeltriage.train.loss import weighted_soft_ce  # noqa: E402

pytestmark = pytest.mark.torch
CPU = torch.device("cpu")


def test_soft_ce_matches_cross_entropy_on_one_hot():
    logits = torch.randn(8, 4)
    y = torch.randint(0, 4, (8,))
    onehot = torch.nn.functional.one_hot(y, 4).float()
    expected = torch.nn.functional.cross_entropy(logits, y)
    assert torch.allclose(weighted_soft_ce(logits, onehot), expected, atol=1e-6)


def test_zero_weight_removes_item():
    logits = torch.randn(2, 3)
    t = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
    full = weighted_soft_ce(logits[:1], t[:1])
    masked = weighted_soft_ce(logits, t, torch.tensor([1.0, 0.0]))
    assert torch.allclose(masked, full / 2)


def _tiny_images(n=32, classes=2, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, n)
    x = np.zeros((n, 32, 32, 3), dtype=np.uint8)
    x[y == 1] = 255  # trivially separable: black vs white
    ts = TargetSet(np.arange(n).astype(str), np.eye(classes)[y], np.ones(n))
    return x, y, ts


def test_augment_keeps_shape_and_is_seeded():
    x = torch.randn(4, 3, 32, 32)
    a = image.augment(x, torch.Generator().manual_seed(1))
    b = image.augment(x, torch.Generator().manual_seed(1))
    assert a.shape == x.shape
    assert torch.equal(a, b)


def test_image_fit_learns_separable_data_and_is_reproducible():
    x, y, ts = _tiny_images()
    cfg = image.ImageConfig(
        loop=loop.LoopConfig(epochs=30, batch_size=8, lr=0.05, amp=False), width=4
    )
    model, history = image.fit(x, ts, 2, cfg, seed=0, device=CPU)
    probs = image.predict(model, x, device=CPU)
    assert probs.shape == (32, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
    assert (probs.argmax(1) == y).mean() == 1.0
    assert history[-1] < history[0]

    _, history2 = image.fit(x, ts, 2, cfg, seed=0, device=CPU)
    np.testing.assert_allclose(history, history2, rtol=1e-5)


def test_on_epoch_hook_sees_every_epoch():
    x, _, ts = _tiny_images(8)
    cfg = image.ImageConfig(loop=loop.LoopConfig(epochs=3, batch_size=4, amp=False), width=4)
    seen = []
    image.fit(x, ts, 2, cfg, seed=0, device=CPU, on_epoch=lambda e, m: seen.append(e))
    assert seen == [0, 1, 2]


def test_text_fit_runs_with_a_tiny_random_model():
    transformers = pytest.importorskip("transformers")
    config = transformers.BertConfig(
        vocab_size=50,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=3,
    )
    model = transformers.BertForSequenceClassification(config)
    n = 12
    gen = torch.Generator().manual_seed(0)
    enc = {
        "input_ids": torch.randint(1, 50, (n, 10), generator=gen),
        "attention_mask": torch.ones(n, 10, dtype=torch.long),
    }
    ts = TargetSet(np.arange(n).astype(str), np.full((n, 3), 1 / 3), np.ones(n))
    cfg = text.TextConfig(
        loop=loop.LoopConfig(epochs=2, batch_size=4, optimizer="adamw", lr=1e-3, amp=False)
    )
    model, history = text.fit(model, enc, ts, cfg, seed=0, device=CPU)
    probs = text.predict(model, enc, batch_size=5, device=CPU)
    assert probs.shape == (n, 3)
    assert len(history) == 2


def test_coteaching_trains_two_networks_on_each_others_easy_items():
    x, y, ts = _tiny_images()
    flipped = ts.targets.copy()
    flipped[:4] = flipped[:4, ::-1]  # a few wrong labels
    noisy = TargetSet(ts.item_ids, flipped, ts.weights)
    cfg = image.ImageConfig(
        loop=loop.LoopConfig(epochs=30, batch_size=8, lr=0.05, amp=False), width=4
    )
    model, history = image.fit(x, noisy, 2, cfg, seed=0, device=CPU, coteach=(0.2, 5))
    assert len(history) == 30
    assert (image.predict(model, x, device=CPU).argmax(1) == y).mean() >= 0.9


def test_coteaching_schedule_ramps_then_holds():
    f = loop.coteaching_schedule(0.3, 10)
    assert f(0) == 0.0
    assert f(5) == pytest.approx(0.15)
    assert f(20) == pytest.approx(0.3)


def test_frozen_dino_head_trains_only_the_head():
    from types import SimpleNamespace

    from labeltriage.train.models import FrozenDinoHead

    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(3, 8)

        def forward(self, pixel_values):
            assert pixel_values.shape[-1] == 224
            return SimpleNamespace(pooler_output=self.proj(pixel_values.mean(dim=(2, 3))))

    model = FrozenDinoHead(2, backbone=FakeBackbone(), feature_dim=8)
    model.train()
    assert not model.backbone.training
    before = model.backbone.proj.weight.clone()
    out = model(torch.randn(4, 3, 32, 32))
    out.sum().backward()
    assert out.shape == (4, 2)
    assert model.backbone.proj.weight.grad is None
    assert torch.equal(before, model.backbone.proj.weight)
    assert model.head[0].weight.grad is not None
