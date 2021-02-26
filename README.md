# jaxifier

Normalizing Flow models using Pytorch, Tensorflow and XLA accellarated JAX which allows for automatically parallelising code across multiple accelerators such as GPUs and TPUs.

The repository is under active developement.

- [ ] Linear Flows [[Pytorch](https://colab.research.google.com/drive/1S-bVMrnnBTIoQZ1OI_5Cd13FrVyKfJ2z#scrollTo=yHrdghn5W4Ca)]        [[TensorFlow]()]   [[JAX](https://colab.research.google.com/drive/1AiKQK5q-7Xy9-N6TfMjoxbzgKQy5BYG6#scrollTo=NIovaeRtvkSI)]
- [ ] Planar Flows
- [ ] Radial Flows
- [ ] Coupling and Autoregressive Flows
- [ ] RealNVPs
- [ ] GLOW

# Training time: PyTorch vs Tensorflow vs JAX
| Model | Dataset  | Data Size  | Flow Length  | Num_Epochs  | Pytorch (sec)  |Tensorflow (sec)  |JAXs (sec)
| :---: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Linear Flow | 1D | 1000 | 100 | 1000 | ~72 | ~160 | ~68