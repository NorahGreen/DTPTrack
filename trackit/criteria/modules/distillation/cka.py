'''
Centered Kernel Alignment (CKA) [1] is a similarity index between representations of features in neural networks,
based on the Hilbert-Schmidt Independence Criterion (HSIC) [2].
Given a set of examples, CKA compares the representations of examples passed through the layers that we want to compare.

[1] Kornblith, Simon, et al. "Similarity of neural network representations revisited." International Conference on Machine Learning. PMLR, 2019.
[2] Wang, Tinghua, Xiaolu Dai, and Yuze Liu. "Learning with Hilbert–Schmidt independence criterion: A review and new perspectives." Knowledge-based systems 234 (2021): 107567.
'''


"""Module that implements both base and mini-batch CKA."""
# https://github.com/RistoAle97/centered-kernel-alignment/blob/main/src/ckatorch/core.py
from typing import Literal

import torch


def cka_base(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: Literal["linear", "rbf"] = "linear",
    unbiased: bool = False,
    threshold: float = 1.0,
    method: Literal["fro_norm", "hsic"] = "fro_norm",
) -> torch.Tensor:
    """Computes the Centered Kernel Alignment (CKA) between two given matrices.

    Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.

    Args:
        x: tensor of shape (n, j).
        y: tensor of shape (n, k).
        kernel: the kernel used to compute the Gram matrices, must be "linear" or "rbf" (default="linear).
        unbiased: whether to use the unbiased version of CKA (default=False).
        threshold: the threshold used by the RBF kernel (default=1.0).
        method: the method used to compute the CKA value, must be "fro_norm" (Frobenius norm) or "hsic"
            (Hilbert-Schmidt Independence Criterion). Note that the choice does not influence the output
            (default="fro_norm").

    Returns:
        a float tensor in [0, 1] that is the CKA value between the two given matrices.

    Raises:
        ValueError: if ``kernel`` is not "linear" or "rbf" or if ``method`` is not "fro_norm" or "hsic".
    """
    if kernel not in ["linear", "rbf"]:
        raise ValueError("The chosen kernel must be either 'linear' or 'rbf'.")

    if method not in ["hsic", "fro_norm"]:
        raise ValueError("The chosen method must be either 'hsic' or 'fro_norm'.")

    x = x.type(torch.float64) if not x.dtype == torch.float64 else x
    y = y.type(torch.float64) if not y.dtype == torch.float64 else y

    # Build the Gram matrices by applying the kernel
    gram_x = linear_kernel(x) if kernel == "linear" else rbf_kernel(x, threshold)
    gram_y = linear_kernel(y) if kernel == "linear" else rbf_kernel(y, threshold)

    # Compute CKA by either using HSIC or the Frobenius norm
    if method == "hsic":
        hsic_xy = hsic0(gram_x, gram_y)
        hsic_xx = hsic0(gram_x, gram_x)
        hsic_yy = hsic0(gram_y, gram_y)
        cka = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
    else:
        gram_x = center_gram_matrix(gram_x, unbiased)
        gram_y = center_gram_matrix(gram_y, unbiased)
        norm_xy = gram_x.ravel().dot(gram_y.ravel())
        norm_xx = torch.linalg.norm(gram_x, ord="fro")
        norm_yy = torch.linalg.norm(gram_y, ord="fro")
        cka = norm_xy / (norm_xx * norm_yy)

    return cka


def cka_batch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the minibatch version of CKA from Nguyen et al. (https://arxiv.org/abs/2010.15327).

    This computation is performed with linear kernel and by calculating HSIC_1.

    Args:
        x: tensor of shape (bsz, n, j).
        y: tensor of shape (bsz, n, k).

    Returns:
        a float tensor in [0, 1] that is the CKA value between the two given tensors.
    """
    x = x.type(torch.float64) if not x.dtype == torch.float64 else x
    y = y.type(torch.float64) if not y.dtype == torch.float64 else y

    # Build the Gram matrices by applying the linear kernel
    gram_x = torch.bmm(x, x.transpose(1, 2))
    gram_y = torch.bmm(y, y.transpose(1, 2))

    # Compute the HSIC values for the entire batches
    hsic1_xy = hsic1(gram_x, gram_y)
    hsic1_xx = hsic1(gram_x, gram_x)
    hsic1_yy = hsic1(gram_y, gram_y)

    # Compute the CKA value
    cka = hsic1_xy.sum() / (hsic1_xx.sum() * hsic1_yy.sum()).sqrt()
    return cka

# https://github.com/RistoAle97/centered-kernel-alignment/blob/main/src/ckatorch/hsic.py
"""Module for computing HSIC (Hilbert-Schmidt Independence Criterion), both its standard and mini-batch versions."""

import torch


def hsic0(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    """Compute the Hilbert-Schmidt Independence Criterion on two given Gram matrices.

    Args:
        gram_x: Gram matrix of shape (n, n), this is equivalent to K from the original paper.
        gram_y: Gram matrix of shape (n, n), this is equivalent to L from the original paper.

    Returns:
        a tensor with the Hilbert-Schmidt Independence Criterion values.

    Raises:
        ValueError: if ``gram_x`` and ``gram_y`` are not symmetric.
    """
    if not torch.allclose(gram_x, gram_x.T) and not torch.allclose(gram_y, gram_y.T):
        raise ValueError("The given matrices must be symmetric.")

    # Build the identity matrix
    n = gram_x.shape[0]
    identity = torch.eye(n, n, dtype=gram_x.dtype, device=gram_x.device)

    # Build the centering matrix
    h = identity - torch.ones(n, n, dtype=gram_x.dtype, device=gram_x.device) / n

    # Compute k * h and l * h
    kh = torch.mm(gram_x, h)
    lh = torch.mm(gram_y, h)

    # Compute the trace of the product kh * lh
    trace = torch.trace(kh.mm(lh))
    return trace / (n - 1) ** 2


def hsic1(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    """Compute the batched version of the Hilbert-Schmidt Independence Criterion on Gram matrices.

    This version is based on
    https://github.com/numpee/CKA.pytorch/blob/07874ec7e219ad29a29ee8d5ebdada0e1156cf9f/cka.py#L107.

    Args:
        gram_x: batch of Gram matrices of shape (bsz, n, n).
        gram_y: batch of Gram matrices of shape (bsz, n, n).

    Returns:
        a tensor with the unbiased Hilbert-Schmidt Independence Criterion values.

    Raises:
        ValueError: if ``gram_x`` and ``gram_y`` do not have the same shape or if they do not have exactly three
        dimensions.
    """
    if len(gram_x.size()) != 3 or gram_x.size() != gram_y.size():
        raise ValueError("Invalid size for one of the two input tensors.")

    n = gram_x.shape[-1]
    gram_x = gram_x.clone()
    gram_y = gram_y.clone()

    # Fill the diagonal of each matrix with 0
    gram_x.diagonal(dim1=-1, dim2=-2).fill_(0)
    gram_y.diagonal(dim1=-1, dim2=-2).fill_(0)

    # Compute the product between k (i.e.: gram_x) and l (i.e.: gram_y)
    kl = torch.bmm(gram_x, gram_y)

    # Compute the trace (sum of the elements on the diagonal) of the previous product, i.e.: the left term
    trace_kl = kl.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)

    # Compute the middle term
    middle_term = gram_x.sum((-1, -2), keepdim=True) * gram_y.sum((-1, -2), keepdim=True)
    middle_term /= (n - 1) * (n - 2)

    # Compute the right term
    right_term = kl.sum((-1, -2), keepdim=True)
    right_term *= 2 / (n - 2)

    # Put all together to compute the main term
    main_term = trace_kl + middle_term - right_term

    # Compute the hsic values
    out = main_term / (n**2 - 3 * n)
    return out.squeeze(-1).squeeze(-1)

# https://github.com/RistoAle97/centered-kernel-alignment/blob/main/src/ckatorch/utils.py
"""Utilities for computing and centering Gram matrices."""

import torch


def linear_kernel(x: torch.Tensor) -> torch.Tensor:
    """Computes the Gram (kernel) matrix for a linear kernel.

    Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.

    Args:
        x: tensor of shape (n, m).

    Returns:
        a Gram matrix which is a tensor of shape (n, n).
    """
    return torch.mm(x, x.T)


def rbf_kernel(x: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Computes the Gram (kernel) matrix for an RBF kernel.

    Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.

    Args:
        x: tensor of shape (n, m).
        threshold: fraction of median Euclidean distance to use as RBF kernel bandwidth (default=1.0).

    Returns:
        a Gram matrix which is a tensor of shape (n, n).
    """
    dot_products = torch.mm(x, x.T)
    sq_norms = torch.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = torch.median(sq_distances)
    return torch.exp(-sq_distances / (2 * threshold**2 * sq_median_distance))


def center_gram_matrix(gram_matrix: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    """Centers a given Gram matrix.

    Adapted from the one made by Kornblith et al.
    https://github.com/google-research/google-research/tree/master/representation_similarity.

    Args:
        gram_matrix: tensor of shape (n, n).
        unbiased: whether to use the unbiased version of the centering (default=False).

    Returns:
        the centered version of the given Gram matrix.
    """
    if not torch.allclose(gram_matrix, gram_matrix.T):
        raise ValueError("The given matrix must be symmetric.")

    gram_matrix = gram_matrix.detach().clone()
    if unbiased:
        n = gram_matrix.shape[0]
        gram_matrix.fill_diagonal_(0)
        means = torch.sum(gram_matrix, dim=0, dtype=torch.float64) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram_matrix -= means[:, None]
        gram_matrix -= means[None, :]
        gram_matrix.fill_diagonal_(0)
    else:
        means = torch.mean(gram_matrix, dim=0, dtype=torch.float64)
        means -= torch.mean(means) / 2
        gram_matrix -= means[:, None]
        gram_matrix -= means[None, :]

    return gram_matrix