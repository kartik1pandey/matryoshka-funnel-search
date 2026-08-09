import functools

import numpy as np
import pytest
from flax import nnx

from matryoshka_search.model.matryoshka_head import MatryoshkaProjectionHead
from matryoshka_search.train.loss import info_nce_loss, matryoshka_loss
from matryoshka_search.train.trainer import TrainConfig, train


def _synthetic_embeddings(n: int, dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Correlated image/text pairs (text = image + small noise) so there is
    # real cross-modal signal for the loss to actually learn from, rather
    # than two independent random clouds.
    image = rng.normal(size=(n, dim)).astype("float32")
    text = (image + 0.1 * rng.normal(size=(n, dim))).astype("float32")
    return image, text


def test_train_loss_decreases_with_matryoshka_loss():
    image_emb, text_emb = _synthetic_embeddings(n=64, dim=16)
    head = MatryoshkaProjectionHead(16, 16, rngs=nnx.Rngs(0))
    loss_fn = functools.partial(matryoshka_loss, dims=[4, 8, 16])
    config = TrainConfig(learning_rate=1e-2, num_epochs=10, batch_size=32)

    history = train(head, image_emb, text_emb, loss_fn, config, seed=0)

    assert len(history) == config.num_epochs
    assert history[-1] < history[0]


def test_train_works_with_single_dimension_baseline_loss():
    image_emb, text_emb = _synthetic_embeddings(n=64, dim=16)
    head = MatryoshkaProjectionHead(16, 16, rngs=nnx.Rngs(0))
    config = TrainConfig(learning_rate=1e-2, num_epochs=10, batch_size=32)

    history = train(head, image_emb, text_emb, info_nce_loss, config, seed=0)

    assert len(history) == config.num_epochs
    assert history[-1] < history[0]


def test_train_rejects_batch_size_larger_than_dataset():
    image_emb, text_emb = _synthetic_embeddings(n=8, dim=4)
    head = MatryoshkaProjectionHead(4, 4, rngs=nnx.Rngs(0))
    config = TrainConfig(num_epochs=1, batch_size=32)

    with pytest.raises(ValueError):
        train(head, image_emb, text_emb, info_nce_loss, config)


def test_train_is_reproducible_given_same_seed():
    image_emb, text_emb = _synthetic_embeddings(n=64, dim=16)
    config = TrainConfig(learning_rate=1e-2, num_epochs=5, batch_size=16)

    head_a = MatryoshkaProjectionHead(16, 16, rngs=nnx.Rngs(0))
    history_a = train(head_a, image_emb, text_emb, info_nce_loss, config, seed=42)

    head_b = MatryoshkaProjectionHead(16, 16, rngs=nnx.Rngs(0))
    history_b = train(head_b, image_emb, text_emb, info_nce_loss, config, seed=42)

    assert history_a == pytest.approx(history_b)
