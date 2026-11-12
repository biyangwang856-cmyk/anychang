# Vendored AnyChange source

The `torchange/models/segment_any_change` subtree comes from
[pytorch-change-models](https://github.com/Z-Zheng/pytorch-change-models/tree/190b07a2eb4c389a5dc3f72802c67f428cdfd6b7/torchange/models/segment_any_change)
at commit `190b07a2eb4c389a5dc3f72802c67f428cdfd6b7`. It includes the SAM
implementation incorporated by the upstream project. The upstream license is
provided in this directory, and the original file headers are preserved.

Local modifications allow explicit CPU or CUDA selection and load SAM
checkpoints on CPU using PyTorch's weights-only mode. The matching and
candidate-generation logic follows the upstream implementation.

- [Segment Any Change](https://arxiv.org/abs/2402.01188), Zheng et al.
- [Segment Anything](https://arxiv.org/abs/2304.02643), Kirillov et al.
