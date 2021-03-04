import sys
import os
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf
import seaborn as sns
from tqdm import tqdm

try:
    from flows import utils
except:
    sys.path.insert(1, 'N:/jaxifier/')
    from flows import utils


class TransformationLayer1D(tf.keras.layers.Layer):

    def __init__(self):
        super(TransformationLayer1D, self).__init__()

        alpha_init = tf.random_normal_initializer()
        self.alpha = tf.Variable(initial_value=alpha_init(shape=(1,), dtype="float32"),
                                 trainable=True)
        beta_init = tf.random_normal_initializer()
        self.beta = tf.Variable(initial_value=beta_init(shape=(1,), dtype="float32"),
                                trainable=True)

    def _f(self, _x):
        """Invertible and dif ferentiable function, f

        Args:
            _x ([type]): [description]

        Returns:
            [type]: [description]
        """
        return tf.where(_x > 0, _x, tf.exp(_x)-1)

    def f_deriv(self, x):
        """Partial derivative of function f, df/dx: Jacobian of the function f during forward flow; Z = f(X)

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        return tf.where(x > 0, self.alpha, tf.math.exp(x))

    def f_inv(self, y):
        """Inverse of df/dz

        Args:
            y ([type]): [description]

        Returns:
            [type]: [description]
        """
        return tf.where(y > 0, (y-self.beta)/self.alpha, (tf.math.log(y+1)-self.beta)/self.alpha)

    def call(self, x):
        """Normalizing flow direction from some complex distribution to simple known
        distribution (Normal) i.e, Z = f(X)

        Forward function compute the bijective transformation from complex distribution to gaussian distribution.
        Also the determinant of partial derivative of function f given x; i.e., df/dx
        Args:
            x (tensor): Sample of 1D data from a complex distribution(Laplace)
        """
        # Convert the laplace data into gaussian distribution f:
        _z = tf.math.exp(self.alpha)*x+self.beta
        # Change of volume induced by the transformation of the normalizing flows
        log_det = -self.alpha
        return _z, log_det

    def inverse(self, z):
        """Generative distribution from the base distribution to some complex distribution
        X = g(Z)

        """
        x = (z - self.beta)*tf.math.exp(-self.alpha)
        return x


class NormalizingFlowModel(tf.keras.Model):

    def __init__(self, num_flows):
        super(NormalizingFlowModel, self).__init__()

        # Source distribution
        self.prior = tfp.distributions.Normal(tf.zeros(1), tf.eye(1))
        self.flows = [TransformationLayer1D() for _ in range(num_flows)]

    def call(self, x):

        log_det = tf.zeros(x.shape[0])
        for flow in self.flows:
            x, ld = flow(x)
            log_det += ld
        prior_logprob = self.prior.log_prob(x)
        z = x
        return z, prior_logprob, log_det

    def inverse(self, z):
        for flow in self.flows[::-1]:
            z = flow.inverse(z)
        x = z
        return x

    def sample(self, n_samples):
        z = tf.squeeze(self.prior.sample((n_samples,)))
        x = self.inverse(z)
        return x


def loss(model, x):
    """Compute the negative log likelihood of x

    Args:
        model (tf.keras.Model): Instance of Normalizing Flow 1D model
        x (vector): Sample of data from complex distribution X (1D)

    Returns:
        _z (vector): Normalized vector of x
        loss (int): Negative log likelihood scaled by the determinant of the transformation function with respect to x
    """
    z, prior_logprob, log_det = model(x)
    return z, tf.math.reduce_mean(-prior_logprob - log_det)


def grad(model, x):
    with tf.GradientTape() as tape:
        z, loss_value = loss(model, x)
        return z, loss_value, tape.gradient(loss_value, model.trainable_variables)


def train(model, X, optimizer, num_epochs):

    # Keep results for plotting
    train_loss_results = []
    num_epochs = 201

    for epoch in tqdm(range(num_epochs)):

        epoch_loss_avg = tf.keras.metrics.Mean()
        for x in X:
            # Compute gradients
            z, loss_value, grads = grad(model, x)
            # Optimize the model
            optimizer.apply_gradients((grad, var) for (grad, var) in zip(
                grads, model.trainable_variables) if grad is not None)
            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # End epoch
        train_loss_results.append(epoch_loss_avg.result())

        if epoch % 50 == 0:
            print(" Epoch {:03d}: Loss: {:.3f}".format(
                epoch, epoch_loss_avg.result()))

    plt.figure(figsize=(16, 6))
    plt.subplot(2, 3, 1)
    sns.distplot(x)
    plt.title("Training data distribution")
    plt.subplot(2, 3, 2)
    sns.distplot(z)
    plt.title("Latent distribution")
    plt.subplot(2, 3, 3)
    samples = model.sample(len(z))
    sns.distplot(samples)
    plt.title("Generated data distribution")
    plt.subplot(2, 3, 4)
    plt.plot(x)
    plt.title('1D training data')
    plt.subplot(2, 3, 5)
    plt.plot(z)
    plt.title('1D latent data')
    plt.subplot(2, 3, 6)
    plt.plot(samples)
    plt.title('1D generated data')
    plt.show()

    plt.show()


if __name__ == '__main__':

    # Training data
    X = utils.get_laplace(1000)

    # Dataset instance
    data = tf.data.Dataset.from_tensor_slices(X).batch(1000)

    # Model instance
    model = NormalizingFlowModel(num_flows=100)

    # Optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    # Training loop
    train(model, data, optimizer, 1000)
