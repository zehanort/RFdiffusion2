from dataclasses import dataclass
from typing import Union

from einops import repeat
from torch import Tensor

import chai_lab.ranking.utils as rutils
from chai_lab.utils.tensor_utils import masked_mean
from chai_lab.utils.typing import Bool, Float, Int, typecheck


@typecheck
@dataclass
class PAEScores:
    """
    complex_pae: Overall PAE matrix for the entire complex.
    per_chain_pae: PAE matrices for each individual chain.
    interface_pae: PAE values specifically at the interfaces between chains.
    """
    complex_pae: Float[Tensor, "... N N"]
    per_chain_pae: Float[Tensor, "... C N N"]
    interface_pae: Float[Tensor, "..."]


@typecheck
def pae(
    logits: Float[Tensor, "... N N Bins"],
    mask: Bool[Tensor, "... N N"],
    bin_centers: Float[Tensor, "Bins"],
    per_pair: bool = False,
) -> Union[Float[Tensor, "... N N"], Float[Tensor, "... N N Bins"]]:
    """
    Computes the PAE by taking the expectation over the bin logits.
    
    Args:
        logits: Logits tensor with shape (..., N, N, Bins).
        mask: Boolean mask tensor with shape (..., N, N).
        bin_centers: Tensor of bin center values with shape (Bins,).
        per_pair: If True, returns PAE per residue pair; otherwise, aggregates over all pairs.
    
    Returns:
        PAE tensor.
    """
    expectations = rutils.expectation(logits, bin_centers)
    if per_pair:
        return expectations  # Shape: (..., N, N)
    else:
        # Aggregate PAE over all residue pairs, optionally masked
        return masked_mean(mask, expectations, dim=(-2, -1))  # Shape: (...)


@typecheck
def per_chain_pae(
    logits: Float[Tensor, "... N N Bins"],
    atom_mask: Bool[Tensor, "... N"],
    asym_id: Int[Tensor, "... N"],
    bin_centers: Float[Tensor, "Bins"],
) -> Float[Tensor, "... C N N"]:
    """
    Computes PAE for each chain individually.
    
    Args:
        logits: Logits tensor with shape (..., N, N, Bins).
        atom_mask: Boolean mask for atoms with shape (..., N).
        asym_id: Asymmetrical IDs indicating chain membership with shape (..., N).
        bin_centers: Tensor of bin center values with shape (Bins,).
    
    Returns:
        Per-chain PAE tensor with shape (..., C, N, N).
    """
    chain_masks, _ = rutils.get_chain_masks_and_asyms(asym_id, atom_mask)
    # Repeat logits for each chain
    logits_repeated = repeat(logits, "... n1 n2 b -> ... c n1 n2 b", c=chain_masks.shape[-2])
    # Create mask for intra-chain residue pairs
    intra_chain_mask = chain_masks.unsqueeze(-1) & chain_masks.unsqueeze(-2)
    return pae(logits_repeated, intra_chain_mask, bin_centers, per_pair=True)  # Shape: (..., C, N, N)


@typecheck
def interface_pae(
    complex_pae: Float[Tensor, "... N N"],
    asym_id: Int[Tensor, "... N"],
) -> Float[Tensor, "..."]:
    """
    Computes the average PAE at the interfaces between different chains.
    
    Args:
        complex_pae: Overall PAE matrix with shape (..., N, N).
        asym_id: Asymmetrical IDs indicating chain membership with shape (..., N).
    
    Returns:
        Interface PAE scalar tensor with shape (...).
    """
    # Create a mask for inter-chain residue pairs
    chain_ids = asym_id
    inter_chain_mask = chain_ids.unsqueeze(-1) != chain_ids.unsqueeze(-2)  # Shape: (..., N, N)
    # Compute the masked mean PAE for interface residues
    return masked_mean(inter_chain_mask, complex_pae, dim=(-2, -1))  # Shape: (...)


@typecheck
def get_scores(
    pae_logits: Float[Tensor, "... N N Bins"],
    atom_mask: Bool[Tensor, "... N"],
    atom_asym_id: Int[Tensor, "... N"],
    bin_centers: Float[Tensor, "Bins"],
) -> PAEScores:
    """
    Aggregates PAE scores for the complex, per chain, and at interfaces.
    
    Args:
        pae_logits: Logits tensor with shape (..., N, N, Bins).
        atom_mask: Boolean mask for atoms with shape (..., N).
        atom_asym_id: Asymmetrical IDs indicating chain membership with shape (..., N).
        bin_centers: Tensor of bin center values with shape (Bins,).
    
    Returns:
        PAEScores dataclass instance containing complex, per-chain, and interface PAE.
    """
    # Compute overall complex PAE
    complex_pae = pae(
        logits=pae_logits,
        mask=atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2),
        bin_centers=bin_centers,
        per_pair=False,
    )  # Shape: (...)

    # Compute per-chain PAE
    per_chain_pae_scores = per_chain_pae(
        logits=pae_logits,
        atom_mask=atom_mask,
        asym_id=atom_asym_id,
        bin_centers=bin_centers,
    )  # Shape: (..., C, N, N)

    # Compute interface PAE
    interface_pae_score = interface_pae(
        complex_pae=complex_pae,
        asym_id=atom_asym_id,
    )  # Shape: (...)

    return PAEScores(
        complex_pae=complex_pae,
        per_chain_pae=per_chain_pae_scores,
        interface_pae=interface_pae_score,
    )
