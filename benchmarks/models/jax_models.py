"""JAX benchmark model architectures across scale tiers."""

from __future__ import annotations

from typing import Any

import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp


def build_jax_convnext(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[Any, tuple[Any, ...]]:
    if scale == "micro":
        img_size = 32
        dims = [32, 64]
        stem_k, stem_s = 2, 2
    elif scale == "standard":
        img_size = 224
        dims = [96, 192]
        stem_k, stem_s = 4, 4
    else:  # production
        img_size = 224
        dims = [128, 256]
        stem_k, stem_s = 4, 4

    w_stem = jnp.ones((dims[0], 3, stem_k, stem_k))
    w_dw1 = jnp.ones((dims[0], 1, 7, 7))
    w_pw1_1 = jnp.ones((dims[0] * 4, dims[0], 1, 1))
    w_pw1_2 = jnp.ones((dims[0], dims[0] * 4, 1, 1))

    w_down = jnp.ones((dims[1], dims[0], 2, 2))
    w_dw2 = jnp.ones((dims[1], 1, 7, 7))
    w_pw2_1 = jnp.ones((dims[1] * 4, dims[1], 1, 1))
    w_pw2_2 = jnp.ones((dims[1], dims[1] * 4, 1, 1))
    w_fc = jnp.ones((dims[1], 10))

    def model(
        x,
        _w_stem=w_stem,
        _w_dw1=w_dw1,
        _w_pw1_1=w_pw1_1,
        _w_pw1_2=w_pw1_2,
        _w_down=w_down,
        _w_dw2=w_dw2,
        _w_pw2_1=w_pw2_1,
        _w_pw2_2=w_pw2_2,
        _w_fc=w_fc,
    ):
        # Stem
        h = lax.conv_general_dilated(
            x, _w_stem, (stem_s, stem_s), "VALID", dimension_numbers=("NCHW", "OIHW", "NCHW")
        )
        # Block 1
        res = h
        h = lax.conv_general_dilated(
            h,
            _w_dw1,
            (1, 1),
            "SAME",
            feature_group_count=dims[0],
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        h = lax.conv_general_dilated(
            h, _w_pw1_1, (1, 1), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW")
        )
        h = jnn.gelu(h)
        h = lax.conv_general_dilated(
            h, _w_pw1_2, (1, 1), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW")
        )
        h = res + h

        # Downsample
        h = lax.conv_general_dilated(
            h, _w_down, (2, 2), "VALID", dimension_numbers=("NCHW", "OIHW", "NCHW")
        )

        # Block 2
        res = h
        h = lax.conv_general_dilated(
            h,
            _w_dw2,
            (1, 1),
            "SAME",
            feature_group_count=dims[1],
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        h = lax.conv_general_dilated(
            h, _w_pw2_1, (1, 1), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW")
        )
        h = jnn.gelu(h)
        h = lax.conv_general_dilated(
            h, _w_pw2_2, (1, 1), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW")
        )
        h = res + h

        return h.mean(axis=(2, 3)) @ _w_fc

    x = jnp.ones((batch, 3, img_size, img_size))
    return model, (x, w_stem, w_dw1, w_pw1_1, w_pw1_2, w_down, w_dw2, w_pw2_1, w_pw2_2, w_fc)


def build_jax_vit(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[Any, tuple[Any, ...]]:
    if scale == "micro":
        img_size, patch_size, embed_dim, num_heads = 32, 4, 128, 4
    elif scale == "standard":
        img_size, patch_size, embed_dim, num_heads = 224, 16, 384, 6
    else:  # production
        img_size, patch_size, embed_dim, num_heads = 224, 16, 768, 12

    head_dim = embed_dim // num_heads
    num_patches = (img_size // patch_size) ** 2
    w_patch = jnp.ones((embed_dim, 3, patch_size, patch_size))
    w_pos = jnp.ones((1, num_patches, embed_dim))

    wq = jnp.ones((embed_dim, embed_dim))
    wk = jnp.ones((embed_dim, embed_dim))
    wv = jnp.ones((embed_dim, embed_dim))
    wo = jnp.ones((embed_dim, embed_dim))
    w_mlp1 = jnp.ones((embed_dim, embed_dim * 4))
    w_mlp2 = jnp.ones((embed_dim * 4, embed_dim))
    w_head = jnp.ones((embed_dim, 10))

    def model(
        x,
        _w_patch=w_patch,
        _w_pos=w_pos,
        _wq=wq,
        _wk=wk,
        _wv=wv,
        _wo=wo,
        _w_mlp1=w_mlp1,
        _w_mlp2=w_mlp2,
        _w_head=w_head,
    ):
        # Patch embedding
        h = lax.conv_general_dilated(
            x,
            _w_patch,
            (patch_size, patch_size),
            "VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        b, d, gh, gw = h.shape
        h = h.reshape((b, d, gh * gw)).transpose((0, 2, 1))
        h = h + _w_pos

        # Self-attention
        q = (h @ _wq).reshape((b, num_patches, num_heads, head_dim))
        k = (h @ _wk).reshape((b, num_patches, num_heads, head_dim))
        v = (h @ _wv).reshape((b, num_patches, num_heads, head_dim))
        attn_out = jax.nn.dot_product_attention(q, k, v).reshape((b, num_patches, embed_dim)) @ _wo
        h = h + attn_out

        # MLP
        mlp_out = jnn.gelu(h @ _w_mlp1) @ _w_mlp2
        h = h + mlp_out

        return h.mean(axis=1) @ _w_head

    x = jnp.ones((batch, 3, img_size, img_size))
    return model, (x, w_patch, w_pos, wq, wk, wv, wo, w_mlp1, w_mlp2, w_head)


def build_jax_deep_dnn(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[Any, tuple[Any, ...]]:
    if scale == "micro":
        in_dim, hidden_dim, out_dim = 784, 128, 128
    elif scale == "standard":
        in_dim, hidden_dim, out_dim = 1024, 4096, 1024
    else:  # production
        in_dim, hidden_dim, out_dim = 2048, 8192, 2048

    w1 = jnp.ones((in_dim, hidden_dim))
    w2 = jnp.ones((hidden_dim, hidden_dim))
    w3 = jnp.ones((hidden_dim, out_dim))
    w_head = jnp.ones((out_dim, 10))

    def model(x, _w1=w1, _w2=w2, _w3=w3, _w_head=w_head):
        x = jnn.gelu(x @ _w1)
        x = jnn.gelu(x @ _w2)
        x = jnn.gelu(x @ _w3)
        return x @ _w_head

    x = jnp.ones((batch, in_dim))
    return model, (x, w1, w2, w3, w_head)


def build_jax_transformer(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[Any, tuple[Any, ...]]:
    if scale == "micro":
        seq_len, embed_dim, num_heads = 32, 128, 4
    elif scale == "standard":
        seq_len, embed_dim, num_heads = 512, 768, 12
    else:  # production
        seq_len, embed_dim, num_heads = 1024, 1024, 16

    head_dim = embed_dim // num_heads
    wq = jnp.ones((embed_dim, embed_dim))
    wk = jnp.ones((embed_dim, embed_dim))
    wv = jnp.ones((embed_dim, embed_dim))
    wo = jnp.ones((embed_dim, embed_dim))
    w_mlp1 = jnp.ones((embed_dim, embed_dim * 4))
    w_mlp2 = jnp.ones((embed_dim * 4, embed_dim))
    w_head = jnp.ones((embed_dim, 10))

    def model(x, _wq=wq, _wk=wk, _wv=wv, _wo=wo, _w_mlp1=w_mlp1, _w_mlp2=w_mlp2, _w_head=w_head):
        b, l, d = x.shape
        q = (x @ _wq).reshape((b, l, num_heads, head_dim))
        k = (x @ _wk).reshape((b, l, num_heads, head_dim))
        v = (x @ _wv).reshape((b, l, num_heads, head_dim))
        attn_out = jax.nn.dot_product_attention(q, k, v).reshape((b, l, d)) @ _wo
        x = x + attn_out
        mlp_out = jnn.gelu(x @ _w_mlp1) @ _w_mlp2
        x = x + mlp_out
        return x.mean(axis=1) @ _w_head

    x = jnp.ones((batch, seq_len, embed_dim))
    return model, (x, wq, wk, wv, wo, w_mlp1, w_mlp2, w_head)


def build_jax_cnn(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[Any, tuple[Any, ...]]:
    img_size = 32 if scale == "micro" else 224
    embed_dim = 128
    k1 = jnp.ones((embed_dim // 2, 3, 3, 3))
    k2 = jnp.ones((embed_dim, embed_dim // 2, 3, 3))
    wfc = jnp.ones((embed_dim, 10))

    def model(x, _k1=k1, _k2=k2, _wfc=wfc):
        y = jnp.tanh(
            lax.conv_general_dilated(
                x, _k1, (1, 1), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW")
            )
        )
        y = jnp.tanh(
            lax.conv_general_dilated(
                y, _k2, (2, 2), "SAME", dimension_numbers=("NCHW", "OIHW", "NCHW")
            )
        )
        return y.mean(axis=(2, 3)) @ _wfc

    x = jnp.ones((batch, 3, img_size, img_size))
    return model, (x, k1, k2, wfc)


def build_jax_rnn(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[Any, tuple[Any, ...]]:
    seq_len = 32 if scale == "micro" else 128
    dim = 128 if scale == "micro" else 256
    wih = jnp.ones((dim, dim))
    whh = jnp.ones((dim, dim))
    wfc = jnp.ones((dim, 10))

    def model(x, _wih=wih, _whh=whh, _wfc=wfc):
        h = jnp.zeros((x.shape[0], dim))
        for t in range(seq_len):
            h = jnp.tanh(x[:, t, :] @ _wih + h @ _whh)
        return h @ _wfc

    x = jnp.ones((batch, seq_len, dim))
    return model, (x, wih, whh, wfc)


def build_jax_lstm(
    scale: str = "standard", batch: int = 1, **kwargs: Any
) -> tuple[Any, tuple[Any, ...]]:
    seq_len = 32 if scale == "micro" else 128
    dim = 128 if scale == "micro" else 256
    wih = jnp.ones((dim, dim * 4))
    whh = jnp.ones((dim, dim * 4))
    wfc = jnp.ones((dim, 10))

    def model(x, _wih=wih, _whh=whh, _wfc=wfc):
        h = jnp.zeros((x.shape[0], dim))
        c = jnp.zeros((x.shape[0], dim))
        for t in range(seq_len):
            gates = x[:, t, :] @ _wih + h @ _whh
            i_g = jnn.sigmoid(gates[:, :dim])
            f_g = jnn.sigmoid(gates[:, dim : 2 * dim])
            g_g = jnp.tanh(gates[:, 2 * dim : 3 * dim])
            o_g = jnn.sigmoid(gates[:, 3 * dim :])
            c = f_g * c + i_g * g_g
            h = o_g * jnp.tanh(c)
        return h @ _wfc

    x = jnp.ones((batch, seq_len, dim))
    return model, (x, wih, whh, wfc)


JAX_BUILDERS: dict[str, Any] = {
    "convnext": build_jax_convnext,
    "vit": build_jax_vit,
    "deep_dnn": build_jax_deep_dnn,
    "transformer": build_jax_transformer,
    "cnn": build_jax_cnn,
    "rnn": build_jax_rnn,
    "lstm": build_jax_lstm,
}
