from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

import chai_lab.ranking.clashes as clashes
import chai_lab.ranking.plddt as plddt
import chai_lab.ranking.ptm as ptm
import chai_lab.ranking.utils as rutils
from chai_lab.utils.typing import Bool, Float, Int, typecheck


@typecheck
@dataclass
class SampleRanking:
    """Sample Ranking Data
    asym ids: a tensor of shape (c,) containing the unique asym ids for
        each chain in the sample. The asym ids are sorted numerically.
    aggregate_score: a tensor of shape (...) containing the aggregate ranking
        score for the sample
    ptm_scores: see ptm.get_scores for a description of the ptm scores
    clash_scores: a dictionary of clash scores
    plddt_scores: see plddt.PLDDTScores for a description of the plddt scores
    """

    asym_ids: Int[Tensor, "c"]
    aggregate_score: Float[Tensor, "..."]
    ptm_scores: ptm.PTMScores
    clash_scores: clashes.ClashScores
    plddt_scores: plddt.PLDDTScores
    pae_scores: Float[Tensor, "... c c"]

def compute_expected_pae(
    pae_logits: Float[Tensor, "... n n pae_bins"],
    pae_bin_centers: Float[Tensor, "pae_bins"],
) -> Float[Tensor, "... n n"]:
    """
    Compute the expected PAE from PAE logits.

    Args:
        pae_logits: Tensor of shape (..., n, n, pae_bins) representing PAE logits.
        pae_bin_centers: Tensor of shape (pae_bins,) representing the centers of PAE bins.

    Returns:
        Tensor of shape (..., n, n) representing the expected PAE.
    """
    pae_probs = torch.softmax(pae_logits, dim=-1)  # Convert logits to probabilities
    pae_expected = torch.sum(pae_probs * pae_bin_centers, dim=-1)  # Expected PAE
    return pae_expected


@typecheck
def rank(
    atom_coords: Float[Tensor, "... a 3"],
    atom_mask: Bool[Tensor, "... a"],
    atom_token_index: Int[Tensor, "... a"],
    token_exists_mask: Bool[Tensor, "... n"],
    token_asym_id: Int[Tensor, "... n"],
    token_entity_type: Int[Tensor, "... n"],
    token_valid_frames_mask: Bool[Tensor, "... n"],
    # lddt
    lddt_logits: Float[Tensor, "... a lddt_bins"],
    lddt_bin_centers: Float[Tensor, "lddt_bins"],
    # pae
    pae_logits: Float[Tensor, "... n n pae_bins"],
    pae_bin_centers: Float[Tensor, "pae_bins"],
    # clash
    clash_threshold: float = 1.1,
    max_clashes: int = 100,
    max_clash_ratio: float = 0.5,
) -> SampleRanking:
    """
    Compute ranking scores for a sample.
    In addition to the pTM/ipTM aggregate score, we also return chain
    and inter-chain level statistics for pTM and clashes.
    see documentation for SampleRanking for a complete description.
    """

    ptm_scores = ptm.get_scores(
        pae_logits=pae_logits,
        token_exists_mask=token_exists_mask,
        valid_frames_mask=token_valid_frames_mask,
        bin_centers=pae_bin_centers,
        token_asym_id=token_asym_id,
    )
    clash_scores = clashes.get_scores(
        atom_coords=atom_coords,
        atom_mask=atom_mask,
        atom_asym_id=torch.gather(
            token_asym_id,
            dim=-1,
            index=atom_token_index.long(),
        ),
        atom_entity_type=torch.gather(
            token_entity_type,
            dim=-1,
            index=atom_token_index.long(),
        ),
        max_clashes=max_clashes,
        max_clash_ratio=max_clash_ratio,
        clash_threshold=clash_threshold,
    )

    plddt_scores = plddt.get_scores(
        lddt_logits=lddt_logits,
        atom_mask=atom_mask,
        bin_centers=lddt_bin_centers,
        atom_asym_id=torch.gather(
            token_asym_id,
            dim=-1,
            index=atom_token_index.long(),
        ),
    )

    # aggregate score
    aggregate_score = (
        0.2 * ptm_scores.complex_ptm
        + 0.8 * ptm_scores.interface_ptm
        - 100 * clash_scores.has_inter_chain_clashes.float()
    )

    # Get chain masks and asym IDs
    chain_mask, asyms = rutils.get_chain_masks_and_asyms(
        asym_id=token_asym_id,
        mask=token_exists_mask,
    )

    # Number of chains
    c = asyms.numel()

    # Compute expected PAE
    pae_expected = compute_expected_pae(pae_logits, pae_bin_centers)  # Shape: [batch, n, n]

    # Initialize tensor to hold PAE scores between chains
    # Shape: [batch, c, c]
    pae_scores = torch.zeros(pae_expected.shape[0], c, c, device=pae_expected.device)

    for i in range(c):
        for j in range(c):
            # Masks for chain i and chain j
            chain_i_mask = chain_mask[:, i, :]  # Shape: [batch, n]
            chain_j_mask = chain_mask[:, j, :]  # Shape: [batch, n]

            # Compute pairwise mask for interactions between chain i and chain j
            pair_mask = chain_i_mask.unsqueeze(2) & chain_j_mask.unsqueeze(1)  # Shape: [batch, n, n]

            # Calculate mean PAE for the chain pair, ignoring NaNs
            mean_pae = torch.where(
                pair_mask,
                pae_expected,
                torch.full_like(pae_expected, float('nan'))
            ).nanmean(dim=(1, 2))  # Shape: [batch]

            pae_scores[:, i, j] = mean_pae  # Assign to pae_scores

    return SampleRanking(
        asym_ids=asyms,
        aggregate_score=aggregate_score,
        ptm_scores=ptm_scores,
        clash_scores=clash_scores,
        plddt_scores=plddt_scores,
        pae_scores=pae_scores,  # Include PAE scores in the ranking
    )


def get_scores(ranking_data: SampleRanking) -> dict[str, np.ndarray]:
    scores = {
        "aggregate_score": ranking_data.aggregate_score,
        "ptm": ranking_data.ptm_scores.complex_ptm,
        "iptm": ranking_data.ptm_scores.interface_ptm,
        "per_chain_ptm": ranking_data.ptm_scores.per_chain_ptm,
        "per_chain_pair_iptm": ranking_data.ptm_scores.per_chain_pair_iptm,
        "has_inter_chain_clashes": ranking_data.clash_scores.has_inter_chain_clashes,
        # TODO replace with just one tensor that contains both
        "chain_intra_clashes": ranking_data.clash_scores.chain_intra_clashes,
        "chain_chain_inter_clashes": ranking_data.clash_scores.chain_chain_inter_clashes,
        "pae": ranking_data.pae_scores,
        "per_chain_plddt": ranking_data.plddt_scores.per_chain_plddt,
        "complex_plddt": ranking_data.plddt_scores.complex_plddt,
    }
    converted_scores = {}
    for k, v in scores.items():
        if isinstance(v, torch.Tensor):
            converted_scores[k] = v.cpu().numpy()
        elif isinstance(v, np.ndarray):
            converted_scores[k] = v
        else:
            # If there are other types, handle them accordingly
            converted_scores[k] = v
    
    return converted_scores