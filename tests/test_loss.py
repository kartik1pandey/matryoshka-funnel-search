import jax.numpy as jnp
import pytest

from matryoshka_search.train.loss import info_nce_loss, matryoshka_loss


def test_info_nce_loss_is_lower_for_well_aligned_pairs():
    key_aligned = jnp.eye(4)
    misaligned = jnp.flip(jnp.eye(4), axis=0)

    aligned_loss = info_nce_loss(key_aligned, key_aligned)
    misaligned_loss = info_nce_loss(key_aligned, misaligned)

    assert float(aligned_loss) < float(misaligned_loss)


def test_info_nce_loss_is_scalar():
    image = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    text = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    loss = info_nce_loss(image, text)
    assert loss.shape == ()


def _dense_nonzero_embeddings():
    # Every truncation prefix (down to dim 1) must stay a nonzero vector,
    # or L2-normalization divides by zero -> NaN. eye(4) truncated to 2 dims
    # has all-zero rows for indices 2 and 3, so it can't be used here.
    return jnp.array(
        [
            [3.0, 1.0, 1.0, 1.0],
            [1.0, 3.0, 1.0, 1.0],
            [1.0, 1.0, 3.0, 1.0],
            [1.0, 1.0, 1.0, 3.0],
        ]
    )


def test_matryoshka_loss_sums_across_dims():
    image = _dense_nonzero_embeddings()
    text = _dense_nonzero_embeddings()

    single_dim_loss = info_nce_loss(image[..., :2], text[..., :2])
    total = matryoshka_loss(image, text, dims=[2, 4])
    expected = single_dim_loss + info_nce_loss(image[..., :4], text[..., :4])

    assert float(total) == pytest.approx(float(expected), rel=1e-5)


def test_matryoshka_loss_rejects_invalid_dim():
    image = _dense_nonzero_embeddings()
    text = _dense_nonzero_embeddings()
    with pytest.raises(ValueError):
        matryoshka_loss(image, text, dims=[2, 8])
