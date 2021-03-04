import jax.numpy as jnp


def get_laplace(n_samples):
    laplace_dist = tfp.distributions.Laplace(0.0, 1.0)
    data = laplace_dist.sample([1000, ])
    return data


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


def jac_f(x, param):
    """ Compute the jacobian of f with respect to x
    Args:
        x (vector): Sample of 1D data from a complex distribution

    Returns:
        det (int): Determinant of the jacobian of function f with respect to given x
    """
    jcb = jnp.where(x > 0, param[0], jnp.exp(x))
    return jcb
