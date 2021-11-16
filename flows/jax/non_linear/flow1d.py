import flax
from flax import linen as nn
from flax import optim
from flax.training import train_state
from flax.core import unfreeze
import optax

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, random, lax
from jax.experimental import optimizers

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

import torch
from torch.distributions import Normal, Laplace

import seaborn as sns
from tqdm import tqdm
from typing import Any, Callable, Sequence, List, Tuple
import matplotlib.pyplot as plt
# Generate key which is used to generate random numbers
key = random.PRNGKey(1)


def get_dataset(data_size, batch_size):

    def laplace(data_size):

        laplace_dist = Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
        data = laplace_dist.sample([1000, ])
        # Standardize data
        for i in range(data.shape[1]):
            data[:, i] = (data[:, i] - torch.mean(data[:, i])) / \
                torch.std(data[:, i])
        sns.distplot(data)
        plt.title('Training data')
        plt.show()
        return data
    x = laplace(data_size)
    dataset = tf.data.Dataset.from_tensor_slices(tf.dtypes.cast(x,
                                                                dtype=tf.float64)).batch(batch_size,
                                                                                         drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = tfds.as_numpy(dataset)
    return dataset


class TransformationLayer1D(nn.Module):

    features: int
    alpha_init: Callable = nn.initializers.normal()
    beta_init: Callable = nn.initializers.normal()

    def setup(self):
        self.alpha = self.param('alpha',
                                self.alpha_init,
                                (self.features,))
        self.beta = self.param('beta', self.beta_init, (self.features,))

    def f(self, _x):
        """Invertible and differentiable function, f

        Args:
            _x ([type]): [description]

        Returns:
        vector: _x transformed to other domain z
        """
        return jnp.where(_x > 0, _x, jnp.exp(_x)-1)

    def f_deriv(self, x):
        """Partial derivative of function f, df/dx: Jacobian of the function f during forward flow; Z = f(X)

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        return jnp.where(x > 0, self.alpha, jnp.exp(x))

    def f_inv(self, y):
        """Inverse of df/dz

        Args:
            y ([type]): [description]

        Returns:
            [type]: [description]
        """
        return jnp.where(y > 0, (y-self.beta)/self.alpha, (jnp.log(y+1)-self.beta)/self.alpha)

    def __call__(self, x):

        # Convert the laplace data into gaussian distribution f:
        z = self.f(self.alpha*x + self.beta)
        # Change of volume induced by the transformation of the normalizing flows
        log_det = jnp.log(jnp.abs(self.f_deriv(x)))
        return z, log_det


class NormalizingFlow(nn.Module):
    n_flows: int
    features: int
    train: bool

    def setup(self):
        self.flows = [TransformationLayer1D(
            features=self.features) for _ in range(self.n_flows)]

    def _shared_modules(self):
        return [flow for flow in self.flows]

    def __call__(self, x):
        log_det = jnp.zeros(x.shape[0])
        prior_logprob = jax.scipy.stats.multivariate_normal.logpdf(x, 0, 1)
        flows = self._shared_modules()
        for flow in flows:
            z, ld = flow(x)
            log_det += ld

        if not self.train:
            x = self.sample(n_samples=x.shape[0])
            return x, z, prior_logprob, log_det
        else:
            return z, prior_logprob, log_det

    def inverse(self, z):
        for flow in self.flows[::-1]:
            x = flow.f_inv(z)
        return x

    def sample(self, n_samples):
        z = jax.random.normal(key, (n_samples, 1))
        x = self.inverse(z)
        return x


@jax.jit
def train_step(state, x):
    def loss_fn(params):
        """Compute log likelihood loss using the prior log probability and the determinants
        Args:
            params (tuple): Parameters of each flow
            x (vector): Observables
        Returns:
            float: Negative log likelihood
        """
        z, prior_logprob, log_det = NormalizingFlow(
            n_flows=10, features=1, train=True).apply({'params': params}, x)
        loss = jnp.mean(-prior_logprob - log_det)
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state


def train_epoch(state, train_ds):
    epoch_loss = 0
    for x in train_ds:
        loss, state = train_step(state, x)
    epoch_loss += loss
    print("Epoch loss ", epoch_loss/len(list(train_ds)))
    return state


@jax.jit
def eval_step(params, x):
    z, _, _ = NormalizingFlow(
        n_flows=10, features=1, train=True).apply({'params': params}, x)
    _, x, _, _ = NormalizingFlow(
        n_flows=10, features=1, train=False).apply({'params': params}, z)
    return z, x


if __name__ == "__main__":

    dataset = get_dataset(1000, 100)

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    model = NormalizingFlow(n_flows=10, features=1, train=True)
    params = model.init(init_rng, jnp.ones([4, 1]))['params']

    lr = 0.001
    nesterov_momentum = 0.9

    tx = optax.sgd(learning_rate=0.001, nesterov=nesterov_momentum)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

    num_epochs = 10
    batch_size = 100

    for epoch in range(1, num_epochs + 1):
        # Use a separate PRNG key to permute image data during shuffling
        rng, input_rng = jax.random.split(rng)
        # Run an optimization step over a training batch
        state = train_epoch(state, dataset)
        if epoch == num_epochs:
            z, x = eval_step(state.params, list(dataset)[0])

            sns.distplot(z, label='Z')
            sns.distplot(x, label='X')
            plt.legend()
