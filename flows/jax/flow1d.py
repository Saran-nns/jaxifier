import jax
import jax.numpy as jnp
from jax import jit, random
import tensorflow_probability as tfp

# Generate key which is used to generate random numbers
key = random.PRNGKey(1)


def f(x, params):
    """Fully differentiable and invertible transformation function mapping domain x to z

    Args:
        x (vector): Sample of 1D data from a complex distribution

    Returns:
        vector: X transformed to other domain z
    """
    return (jnp.where(params[0]*x+params[1] > 0, x, jnp.exp(x)-1))


def f_deriv(x, params):
    """

    Args:
        x (vector): Sample of 1D data from a complex distribution
        param (dict): With parameter alpha and beta of the transformation layer

    Returns:
        [type]: [description]
    """
    return jnp.where(x > 0, params[0], jnp.exp(x))


def f_inv(z, param):
    """Inverse of df/dz

    Args:
        z (vector): Sample of 1D data from a simple distribution
        param (dict): With parameter alpha and beta of the transformation layer

    Returns:
        vector: Generate the training data x back from z
    """

    return jnp.where(z > 0,
                     z-param[1]/param[0], (jnp.log(z+1)-param[1])/param[0])


def jac_f(x):
    """ Compute the jacobian of f with respect to x
    Args:
        x (vector): Sample of 1D data from a complex distribution

    Returns:
        det (int): Determinant of the jacobian of function f with respect to given x
    """
    # Jax placeholder for the transformation function f
    jbn = jax.jacobian(f)
    # Create a function which maps jaxified function f over the axes
    vmap_jbn = jax.vmap(jbn)
    # Compute product of the determinant of the jacobian of f given x
    det = jnp.linalg.det(vmap_jbn(x))
    return det


# Jitify computations
jit_f = jit(f)  # Transformation function
jit_f_deriv = jit(f_deriv)  # Derivative of f
jit_f = jit(f_inv)  # Inverse of f
jit_jac_f = jit(jac_f)  # Jacobian of f with respect to x


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
    _z = jit_f(param[0]*x+param[1])

    # Jacobian of f with respect to x
    jac_fx = jit_jac_f(x)

    # Compute the determinant of the jacobians,i.e, Change of volume induced
    # by the transformation of the normalizing flows
    det = jnp.linalg.det(jac_fx)

    # Log absolute value of the determinant
    log_det = jnp.log(jnp.abs(det))

    return _z, log_det


def generative_direction(z, param):
    """Generative distribution from the base distribution to some complex distribution
        X = g(Z)

    Args:
        z (vector): Samples from the base distribution
        param (dict): With parameter alpha and beta of the transformation layer
        """
    _x = f_inv(z, param)
    return _x


class NormalizingFlowModel:

    def __init__(self):
        self.prior = tfp.distributions.Normal(
            jnp.zeros(1), jnp.ones(1))

    @staticmethod
    def initialize_param(key):
        alpha_key, beta_key = jax.random.split(key)
        return random.normal(alpha_key, (1,)), random.normal(beta_key, (1,))

    def forward(self, _x, params):

        log_det = jnp.zeros(_x.shape[0])
        for flow, _ in enumerate(params):
            _x, ld = normalizing_direction(_x, params[flow])
            log_det += ld
        prior_logprob = self.prior.log_prob(_x)
        z = _x
        return z, prior_logprob, log_det

    def inverse(self, _z, params):
        for param in params[::-1]:  # Reverse flow
            _z = generative_direction(_z, param)
        x = _z
        return x

    def sample(self, n_samples, params):
        z = self.prior.sample((n_samples,))
        x = self.inverse(z, params)
        return x


def train(model, _x, n_flows, key):

    params = [initialize_param(key) for _ in range(n_flows)]

    # Forward pass
    z, prior_logprob, log_det = nf_model.forward(_x, params)

    return loss
