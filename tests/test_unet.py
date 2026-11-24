import pytest

torch = pytest.importorskip("torch")

from cosmicweb.models import NORMS, PADDING_MODES, UNet  # noqa: E402


def small(**kw):
    """A tiny U-Net so the tests run fast on CPU."""
    return UNet(**{"base_channels": 8, "depth": 3, **kw})


@pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 3, 32, 96), (3, 3, 256, 256)])
def test_output_shape_matches_input(shape):
    out = small()(torch.randn(shape))
    assert out.shape == (shape[0], 1, *shape[2:]) and out.dtype == torch.float32


def test_default_model_is_compact_and_handles_full_maps():
    model = UNet()
    assert 7.0e6 < model.num_parameters() < 7.5e6  # documented as ~7.2 M
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 512, 512)).shape == (1, 1, 512, 512)


def test_output_is_linear():
    """No final activation: the head is a 1x1 conv and outputs take both signs, unbounded."""
    model = small()
    assert isinstance(model.head, torch.nn.Conv2d) and model.head.kernel_size == (1, 1)
    with torch.no_grad():
        model.head.bias.fill_(-5.0)
        out = model(torch.randn(2, 3, 32, 32))
    assert out.min() < -1 and out.max() < 0  # a ReLU/sigmoid/tanh output could not do this


def test_gradients_reach_every_parameter():
    model = small()
    loss = model(torch.randn(2, 3, 32, 32)).pow(2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("padding_mode", PADDING_MODES)
def test_norm_and_padding_options(norm, padding_mode):
    model = small(norm=norm, padding_mode=padding_mode).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        a, b = model(x), model(x)
    assert torch.equal(a, b) and torch.isfinite(a).all()


def test_eval_output_does_not_depend_on_batch_composition():
    """In eval mode each map's prediction is independent of the others in the batch."""
    model = small().eval()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(x)[1:2], model(x[1:2]), rtol=1e-5, atol=1e-5)


def test_mask_channel_matters():
    model = small().eval()
    x = torch.randn(1, 3, 32, 32)
    y = x.clone()
    y[:, 2] = 0
    with torch.no_grad():
        assert not torch.allclose(model(x), model(y))


@pytest.mark.parametrize("bad", [(1, 2, 32, 32), (1, 3, 30, 32), (3, 32, 32)])
def test_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        small()(torch.randn(bad))


@pytest.mark.parametrize("kwargs", [dict(norm="layer"), dict(padding_mode="wrap"), dict(depth=0),
                                    dict(base_channels=0)])
def test_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        UNet(**kwargs)


def test_can_overfit_one_example():
    """Sanity check that the network can learn: loss on one fixed pair drops a lot."""
    torch.manual_seed(0)
    model = small(depth=2)
    x, y = torch.randn(1, 3, 32, 32), torch.randn(1, 1, 32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = None
    for _ in range(150):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.1 * first


def test_works_under_autocast():
    model = small().eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, 1, 32, 32) and torch.isfinite(out.float()).all()
