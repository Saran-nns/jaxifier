import jax
import jax.numpy as jnp
from jax import jit, random

# Generate key which is used to generate random numbers
key = random.PRNGKey(1)


def f(x, params):
    """Fully differentiable and invertible transformation function mapping domain x to z

    Args:
        x (vector): Sample of 1D data from a complex distribution

    Returns:
        vector: X transformed to other domain z
    """
    return (jnp.where(params['alpha']*x+params['beta'] > 0, x, jnp.exp(x)-1))


def f_deriv(x, params):
    """

    Args:
        x (vector): Sample of 1D data from a complex distribution
        param (dict): With parameter alpha and beta of the transformation layer

    Returns:
        [type]: [description]
    """
    return jnp.where(x > 0, params['alpha'], jnp.exp(x))


def f_inv(z, params):
    """Inverse of df/dz

    Args:
        z (vector): Sample of 1D data from a simple distribution
        params (dict): With parameter alpha and beta of the transformation layer

    Returns:
        vector: Generate the training data x back from z
    """

    return (z > 0).type(jnp.float32)*(
        z-params['beta'])/params['alpha'] + (z <= 0).type(jnp.float32)*(jnp.log(z+1)-params['beta'])/params['alpha']


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


def normalizing_direction(x, params):
    """Normalizing flow direction from some complex distribution to simple known
    distribution (Normal). Compute Z = f(X)

    Args:
        x (tensor): Sample of 1D data from a complex distribution(Laplace)
        params (dict): With parameter alpha and beta of the transformation layer

    Returns:
        _z (vector): Normalized vector of x
        log_det (int): Log absolute value of the determinant of the jacobian of f with respect to x
    """

    # Convert the laplace data into gaussian distribution f:
    _z = jit_f(params['alpha']*x+params['beta'])

    # Jacobian of f with respect to x
    jac_fx = jit_jac_f(x)

    # Compute the determinant of the jacobians,i.e, Change of volume induced
    # by the transformation of the normalizing flows
    det = jnp.linalg.det(jac_fx)

    # Log absolute value of the determinant
    log_det = jnp.log(jnp.abs(det))

    return _z, log_det


def generative_direction(z, params):
    """Generative distribution from the base distribution to some complex distribution
        X = g(Z)

    Args:
        z (vector): Samples from the base distribution
        params (dict): With parameter alpha and beta of the transformation layer
        """
    _x = f_inv(z, params)
    return _x


def loss(x, params):

    # Forward pass

    return loss
