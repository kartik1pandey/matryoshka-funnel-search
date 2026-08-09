import jax
import jax.numpy as jnp
from flax import nnx

from matryoshka_search.model.matryoshka_head import MatryoshkaProjectionHead


def test_output_shape():
    head = MatryoshkaProjectionHead(512, 512, rngs=nnx.Rngs(0))
    x = jnp.ones((4, 512))
    out = head(x)
    assert out.shape == (4, 512)


def test_supports_non_square_projection():
    head = MatryoshkaProjectionHead(512, 256, rngs=nnx.Rngs(0))
    out = head(jnp.ones((3, 512)))
    assert out.shape == (3, 256)


def test_forward_pass_does_not_mutate_parameters():
    # Applying the same head to both the image and text tower's embeddings
    # (docs/02_architecture.md's shared-weights design) only works if a
    # forward pass is pure — calling it repeatedly must never change the
    # weights out from under a later call.
    head = MatryoshkaProjectionHead(8, 8, rngs=nnx.Rngs(0))
    kernel_before = jnp.array(head.linear.kernel[...])

    head(jnp.ones((2, 8)))
    head(jnp.ones((3, 8)) * 5.0)

    kernel_after = jnp.array(head.linear.kernel[...])
    assert jnp.array_equal(kernel_before, kernel_after)


def test_different_seeds_give_independent_weights():
    head_a = MatryoshkaProjectionHead(8, 8, rngs=nnx.Rngs(0))
    head_b = MatryoshkaProjectionHead(8, 8, rngs=nnx.Rngs(1))
    assert not jnp.allclose(head_a.linear.kernel[...], head_b.linear.kernel[...])


def test_is_differentiable_end_to_end():
    head = MatryoshkaProjectionHead(4, 4, rngs=nnx.Rngs(0))
    x = jnp.array([[1.0, 2.0, 3.0, 4.0]])

    graphdef, params = nnx.split(head)

    def loss_fn(params):
        model = nnx.merge(graphdef, params)
        out = model(x)
        return jnp.sum(out**2)

    grads = jax.grad(loss_fn)(params)
    kernel_grad = grads["linear"]["kernel"][...]
    assert kernel_grad.shape == (4, 4)
    assert not jnp.allclose(kernel_grad, jnp.zeros_like(kernel_grad))
