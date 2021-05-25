# memory-network-pytorch

Several memory network implemented with pytorch

## Neural Turing Machine

https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html

NTM(Neural Turing Machine)是一个结合神经网络和记忆装置的模型结构。这个结构模仿了图灵机的纸带结构，使用神经网络来控制纸带的读和写的操作。

NTM由两个结构组成，一个控制器，一个记忆仓库。控制器用来操作记忆仓库，它可以是任意的神经网络。记忆仓库用来存储信息，它是一个 $N \times M$ 的矩阵，包含 N 个行，每行有M个维度。控制器的每一次操作是并行的一组读和写的操作。读和写操作使用注意力机制来从记忆仓库中寻找地址。

## Memory Augmented Networks for Garment Recommendation

* MANTRA: Memory Augmented Networks for Multiple Trajectory Prediction
* Garment Recommendation with Memory Augmented Neural Networks

