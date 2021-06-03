# [Quantum Self-Supervised Learning](https://arxiv.org/abs/2103.14653)

This repository contains the code used to generate results in https://arxiv.org/abs/2103.14653. This is achieved using a
PyTorch implementation of [SimCLR](https://arxiv.org/abs/2002.05709) based on https://github.com/facebookresearch/moco,
adapted so that the encoder consists of ResNet-18 followed by a representation network.

![image](https://user-images.githubusercontent.com/14994219/120330295-9569c100-c2e4-11eb-8710-b4c2e284676a.png)

### Self-supervised Training with Classical Representation Network

```
python train_simclr.py --gpu 0 --lr 1e-3 -b 256 -d data/ -w 8 

Optional arguments:
--gpu            gpu_id
--lr             learning rate
-b               batch_size
-d               data_dir
-w               width of representation network
```

### Self-supervised Training with Quantum Representation Network

**NOTE** You must clone [quantum-neural-network](https://github.com/bjader/quantum-neural-network) and
add it to your sys/python path first.

```
python train_simclr.py -q --q_backend qasm_simulator --q_ansatz sim_circ_14_half -w 8 --classes 5 --save-batches --epochs 2

Optional arguments:
-q               Flag to use a quantum representation network
--q_backend      Qiskit backend to use (can include real quantum devices)
--q_ansatz       Variational ansatz for quantum neural network.
-w               Width of representation network, in this case the number of qubits.
--classes        Use the first N classes of the dataset
--save-batches   Save the model after each batch (rather than epoch by default)
--epochs         Number of epochs to train for (quantum training takes a long time)
```

The so called "ring" and "all-to-all" ansatzes used in the paper correspond to `--q_anzatz sim_circ_14_half` and `abbas`
in these options respectively.

### Linear Probing the Above Quantum Model

```
python linear_probe_simclr.py --pretrained model/selfsup/path_to_checkpoint_0000.path.tar -q --q_backend qasm_simulator --q_ansatz sim_circ_14_half -w 8 --classes 5

Optional arguments:
--pretrained     path_to_self_sup_model
```

Running the above code block for each checkpoint in the trained model will produce results comparable to the purple line
in Fig. 5 of [Quantum Self-Supervised Learning](https://arxiv.org/pdf/2103.14653.pdf)

![image](https://user-images.githubusercontent.com/14994219/120349125-bf77af00-c2f5-11eb-957e-853c9e5e9f53.png)

Producing the orange line can be done by changing `--q_backend qasm_simulator` to `--q_backend statevector_simulator`.

## Usage and citation

We kindly ask any publication, whitepaper or project using this code to cite:
```
Jaderberg, B., Anderson, L.W., Xie, W., Albanie, S., Kiffner, M. and Jaksch, D., 2021. Quantum Self-Supervised Learning. arXiv preprint arXiv:2103.14653.
```