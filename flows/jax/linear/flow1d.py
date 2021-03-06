import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, random, value_and_grad
import tensorflow_probability as tfp
from jax.experimental import optimizers
import tensorflow as tf
import seaborn as sns
from tqdm import tqdm
import os
import sys
import time
import tensorflow_datasets as tfds
from flows import utils
from flows.jax import helpers
from flows.helpers import *

# Generate key which is used to generate random numbers
key = random.PRNGKey(1)


def initialize_parameters(n_flows, key):
    """ Initialize the weights of all layers of a linear layer network """
    keys = random.split(key, n_flows)
    # Initialize a single layer with Gaussian weights -  helper function

    def initialize_layer(key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (1,)), scale * random.normal(b_key, (1,))
    return [initialize_layer(k) for k in keys]


def normalizing_direction(x, param):
    """Normalizing flow direction from some complex distribution to simple known
    distribution (Normal). Compute Z = f(X)

    Args:
        x (tensor): Sample of 1D data from a complex distribution(Laplace)
        param (dict): With parameter alpha and beta of the transformation layer

    Returns:
        _z (vector): Normalized vector of x
        log_det (int): Log absolute value of the determinant of the jacobian of f with respect to x
    """

    # Convert the laplace data into gaussian distribution f:
    _z = param[0]*x + param[0]

    # Jacobian of f with respect to x
    log_det = jnp.log(jnp.abs(param[0]))

    return _z, log_det


def generative_direction(z, param):
    """Generative distribution from the base distribution to some complex distribution
        X = g(Z)

    Args:
        z (vector): Samples from the base distribution
        param (dict): With parameter alpha and beta of the transformation layer
        """
    _x = (z - param[1])/param[0]
    return _x


def forward(_x, params):
    """Compute Normalizing direction of the flow applying given number of bijective transformations

    Args:
        _x (vector): Sample of 1D data from a complex distribution
        params (tuple): Parameters of each flow as a tuple

    Returns:
        z (vector): Latent state
        prior_logprob (vector): Log prior probability of observables with repect to latent prior
        log_det (vector): Log absolute value of the determinant of the jacobian of f with respect to x
    """
    log_det = 0.
    for param in params[:-1]:  # Forward flow
        _x, ld = normalizing_direction(_x, param)
        log_det += ld
    prior_logprob = jax.scipy.stats.norm.logpdf(_x, loc=0, scale=1)
    z = _x
    return z, jnp.nan_to_num(prior_logprob), log_det


def inverse(_z, params):
    """Compute the generative direction of the flow given the number of bijective transformations

    Args:
        _z (vector): Latent samples
        params (tuple): Parameters of each flow as a tuple

    Returns:
        x (vector): Generated observables
    """
    for param in params[::-1]:  # Reverse flow
        _z = generative_direction(_z, param)
    x = _z
    return x


def sample(n_samples, params):
    """Sample data from the latent and generate data using the inverse flow model

    Args:
        n_samples (int): Sample length
        params (tuple): Parameters of each flow as a tuple

    Returns:
        x (vector): Generated observables
    """
    z = jax.random.normal(key, (n_samples,))
    x = inverse(z, params)
    return x


def loss(params, x):
    """Computelog likelihood loss using the prior log probability and the determinants

    Args:
        params (tuple): Parameters of each flow as a tuple
        x (vector): Observables

    Returns:
        float: Negative log likelihood
    """
    _, prior_logprob, log_det = forward(x, params)
    loss_ = jnp.mean(-prior_logprob - log_det)
    return loss_


@jit
def update(params, x, opt_state):
    """Wrappper to update parameters of the model

    Args:
        params (tuple): Parameters of each flow as a tuple
        x (vector): Observables
        opt_state (tuple): State of the optimizer at the end of each update

    Returns:
        params (tuple): Updated parameters of the model
    """
    value, grads = value_and_grad(loss)(params, x)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, value


def train(num_epochs, params):
    """Training Loop

    Args:
        num_epochs (int): Number of iterations through data
        params (tuple): params (tuple): Parameters of each flow as a tuple
    """

    # Dataset
    X = helpers.get_laplace(1000)
    DATASET = tf.data.Dataset.from_tensor_slices(tf.dtypes.cast(X,
                                                                dtype=tf.float64)).batch(100,
                                                                                         drop_remainder=True)
    DATASET = DATASET.prefetch(tf.data.experimental.AUTOTUNE)
    DATASET = tfds.as_numpy(DATASET)

    # Define optimizer
    lr = 0.001
    opt_init, _, get_params = optimizers.adam(lr)
    opt_state = opt_init(params)

    # Train Loop
    params = get_params(opt_state)
    start_time = time.time()
    for _ in range(num_epochs):
        for x in DATASET:
            x = jnp.array(x)
            params, opt_state, loss_ = update(params, x, opt_state)
    epoch_time = time.time() - start_time
    print('Time', epoch_time)


if __name__ == "__main__":

    n_flows = 100
    params = initialize_parameters(100, key)
    train(num_epochs=1000, params=params)
