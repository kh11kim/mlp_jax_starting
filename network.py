from jax import lax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Sequence
import orbax
from flax.training import orbax_utils

class LipLinear(nn.Module):
    features: int
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, inputs):
        kernel = self.param('kernel',
            self.kernel_init,
            (jnp.shape(inputs)[-1], self.features))
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,))
        else:   bias = None
        c = self.param('c', self.bias_init, 1)
        absrowsum = jnp.sum(jnp.abs(kernel), axis=0)
        scale = jnp.minimum(1.0, nn.softplus(c)/absrowsum)
        y = lax.dot_general(inputs, kernel*scale,
                        (((inputs.ndim - 1,), (0,)), ((), ())))
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y

class MLP(nn.Module):
    dims: Sequence[int]
    skip_layer: int = 0 #if 0, no skip connection
    linear: nn.Module = nn.Dense
    actv_fn: Callable = nn.relu
    out_actv_fn: Callable = None

    def setup(self):
        self.nin = self.dims[0]
        layers = []
        for i, dim in enumerate(self.dims[1:]):
            if i == self.skip_layer and i != 0:
                dim += self.nin
            layers += [self.linear(dim)]
        self.layers = layers
    
    def __call__(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer and i != 0:
                x = jnp.hstack([x, inputs])
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.actv_fn(x)

        if self.out_actv_fn is not None:
            x = self.out_actv_fn(x)
        return x

def save(path, params):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    orbax_checkpointer.save(path, params, save_args=save_args)

def load(path):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    return orbax_checkpointer.restore(path)
