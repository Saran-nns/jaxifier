import unittest
from flows.jax.flow1d import train as jflow
from flows.tensorflow.flow1d import train as tflow
from flows.pytorch.flow1d import train as pflow


class TestJaxifier(unittest.TestCase):
    def test_runsorn(self):
        self.assertRaises(
            Exception, tflow(t_model, t_X, t_optim, num_epochs)
        )
        self.assertRaises(
            Exception, pflow(p_model, x, p_optim, epochs)
        )
        self.assertRaises(
            Exception, jflow(num_epochs=2, params)
        )
