import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cosmicweb.data import ShearMapDataset, kappa_to_shear  # noqa: E402


def _ds(built, split="train", noise="high", mask="mid", **kw):
    cfg, _ = built
    return ShearMapDataset(cfg.build.output_dir, split, cfg.observation, noise, mask, **kw)


def test_item_contract(built):
    ds = _ds(built)
    item = ds[0]
    assert item["x"].shape == (3, 64, 64) and item["x"].dtype == torch.float32
    assert item["y"].shape == (1, 64, 64) and item["y"].dtype == torch.float32
    x = item["x"].numpy()
    assert set(np.unique(x[2])) <= {0.0, 1.0} and (x[:2, x[2] == 0] == 0).all()
    assert len(ds) == 6 and len(_ds(built, "val")) == 2


def test_target_is_scaled_zero_mean_kappa(built):
    cfg, _ = built
    ds = _ds(built)
    kappa = np.load(cfg.build.output_dir / "kappa_train.npy")[2].astype(np.float64)
    np.testing.assert_allclose(ds[2]["y"].numpy()[0], 10.0 * (kappa - kappa.mean()), rtol=1e-6, atol=1e-7)


def test_noiseless_unmasked_input_is_exact_shear_even_after_augmentation(built):
    ds = _ds(built, noise="none", mask="none", mode="random", patch_size=64, augment=True)
    for i in range(len(ds)):
        item = ds[i]
        x, y = item["x"].numpy() / 10.0, item["y"].numpy()[0] / 10.0
        np.testing.assert_allclose(x[:2], np.stack(kappa_to_shear(y)), atol=1e-6)
        assert (x[2] == 0.1).all()  # mask channel is not scaled: 1.0 / 10 here


def test_tiles_stitch_back_to_the_full_map(built):
    full, tiles = _ds(built), _ds(built, mode="tiles", patch_size=32)
    assert len(tiles) == 4 * len(full)
    for m in (0, 3):
        x = np.zeros((3, 64, 64), np.float32)
        y = np.zeros((1, 64, 64), np.float32)
        for t in range(4):
            item = tiles[4 * m + t]
            assert item["map_index"] == m
            x[:, item["y0"]:item["y0"] + 32, item["x0"]:item["x0"] + 32] = item["x"].numpy()
            y[:, item["y0"]:item["y0"] + 32, item["x0"]:item["x0"] + 32] = item["y"].numpy()
        np.testing.assert_array_equal(x, full[m]["x"].numpy())
        np.testing.assert_array_equal(y, full[m]["y"].numpy())


def test_val_inputs_are_fixed_and_training_can_resample(built):
    a, b = _ds(built, "val"), _ds(built, "val")
    b.set_epoch(5)
    assert torch.equal(a[0]["x"], b[0]["x"])  # same across instances and epochs

    tr = _ds(built, resample_each_epoch=True)
    first = tr[0]["x"].clone()
    tr.set_epoch(1)
    assert not torch.equal(first, tr[0]["x"])
    tr.set_epoch(0)
    assert torch.equal(first, tr[0]["x"])


def test_random_crops_move_with_epoch(built):
    ds = _ds(built, mode="random", patch_size=32, crops_per_map=3)
    assert len(ds) == 18
    pos0 = [(ds[i]["y0"], ds[i]["x0"]) for i in range(len(ds))]
    ds.set_epoch(1)
    assert pos0 != [(ds[i]["y0"], ds[i]["x0"]) for i in range(len(ds))]


def test_noise_levels_share_draws_and_masks_are_independent_of_noise(built):
    clean, low, high = (_ds(built, noise=n)[1]["x"].numpy().astype(np.float64) for n in ("none", "low", "high"))
    np.testing.assert_array_equal(clean[2], high[2])  # noise level does not change the mask
    ratio = 30 ** 0.5 / 6 ** 0.5  # σ ∝ 1/√n_gal
    valid = clean[2] > 0
    np.testing.assert_allclose((high - clean)[:2, valid], ratio * (low - clean)[:2, valid], rtol=1e-4, atol=1e-5)
    assert (_ds(built, mask="none")[1]["x"].numpy()[2] == 1).all()


def test_pickling_does_not_copy_the_data(built):
    ds = _ds(built)
    ds[0]  # opens the memory map
    clone = pickle.loads(pickle.dumps(ds))
    assert len(pickle.dumps(ds)) < 20_000
    assert torch.equal(clone[0]["x"], ds[0]["x"])


def test_dataloader_with_workers(built):
    loader = torch.utils.data.DataLoader(_ds(built, mode="tiles", patch_size=32), batch_size=4, num_workers=2)
    batches = list(loader)
    assert sum(b["x"].shape[0] for b in batches) == 24 and batches[0]["x"].shape == (4, 3, 32, 32)


def test_bad_arguments(built):
    with pytest.raises(ValueError):
        _ds(built, split="training")
    with pytest.raises(ValueError):
        _ds(built, mode="tiles", patch_size=48)
    with pytest.raises(KeyError):
        _ds(built, noise="extreme")
