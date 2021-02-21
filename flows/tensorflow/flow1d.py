import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
import matplotlib.pyplot as plt
from flows import utils


class TransformationLayer1D(tf.keras.layers.Layer):
    """

    Args:
        tf (instance): Custom tf keras layer instance
    """

    def __init__(self):
        super(TransformationLayer1D, self).__init__()

        alpha_init = tf.random_normal_initializer()
        self.alpha = tf.Variable(initial_value=alpha_init(shape=(1,), dtype="float32"),
                                 trainable=True)
        beta_init = tf.random_normal_initializer()
        self.beta = tf.Variable(initial_value=beta_init(shape=(1,), stype="float32"),
                                trainable=True)

        # Invertible and dif ferentiable function, f
        self.f = lambda x: (x > 0).type(tf.float32) * \
            x + (x <= 0).type(tf.float32) * (tf.exp(x)-1)

        # Partial derivative of function f, df/dx: Jacobian of the function f during forward flow; Z = f(X)
        self.df_dx = lambda x, alpha: (x > 0).type(
            tf.float32)*alpha + (x <= 0).type(tf.float32) * tf.math.exp(x)

        # Inverse of df/dz
        self.f_inv = lambda y, alpha, beta: (y > 0).type(tf.float32)*(
            y-beta)/alpha + (y <= 0).type(tf.float32)*(tf.math.log(y+1)-beta)/alpha

    def call(self, x):
        """Normalizing flow direction from some complex distribution to simple known
        distribution (Normal) i.e, Z = f(X)

        Args:
            x (tensor): Sample of 1D data from a complex distribution(Laplace)
        """
        # Convert the laplace data into gaussian distribution f:
        _z = tf.keras.activations.elu(self.alpha*x+self.beta, 1)
        # Determinant of partial derivative of function f given x; i.e., df/dx
        # Change of volume induced by the transformation of the normalizing flows
        log_det = tf.math.log(tf.math.abs(self.df_dx(x, self.alpha)))
        return _z, log_det

    def inverse(self, z):
        """Generative distribution from the base distribution to some complex distribution
        X = g(Z)

        """
        x = self.f_inv(z, self.alpha, self.beta)
        return x


class NormalizingFlowModel(tf.keras.Model):

    def __init__(self, num_flows):
        super(NormalizingFlowModel, self).__init__()

        # Source distribution
        self.prior = tfp.Normal(tf.zeros(1), tf.eye(1))
        self.flows = [TransformationLayer1D() for _ in range(num_flows)]

    def call(self, x):
        log_det = tf.zeros(x.shape[0])
        for flow in self.flows:
            z, ld = flow(x)
            log_det += ld
        prior_logprob = self.prior.log_prob(x)
        return z, prior_logprob, log_det

    def inverse(self, z):
        log_det = tf.zeros(z.shape)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, n_samples):
        z = self.prior.sample((n_samples,)).squeeze()
        x, _ = self.inverse(z)
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
    return z, (-prior_logprob - log_det).mean()


def grad(model, x):
    with tf.GradientTape() as tape:
        z, loss_value = loss(model, x)
        return z, loss_value, tape.gradient(loss_value, model.trainable_variables)


def train(model, X, optimizer, num_epochs):

    # Keep results for plotting
    train_loss_results = []
    num_epochs = 201

    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        for x in X:
            # Optimize the model
            _, loss_value, grads = grad(model, x)
            optimizer.apply_gradients(zip(grads,
                                          model.trainable_variables))
            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # End epoch
        train_loss_results.append(epoch_loss_avg.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}".format(
                epoch, epoch_loss_avg.result()))

    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    plt.show()


if __name__ == '__main__':

    # Training data
    X = utils.get_laplace(1000)
    # Model instance
    model = NormalizingFlowModel(num_flows=100)
    # Optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    # Training loop
    train(model, X, optimizer, 1000)
