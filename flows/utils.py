import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import seaborn as sns
import matplotlib.pyplot as plt


def get_laplace(n_samples):
    laplace_dist = tfp.distributions.Laplace(0.0, 1.0)
    data = laplace_dist.sample([1000, ])
    return data


def get_gaussian(n_samples):
    base_mu, base_cov = tf.zeros(1), tf.eye(1)
    base_dist = tfp.distributions.Normal(base_mu, base_cov)
    z = base_dist.rsample(sample_shape=(1000,))
    return z
