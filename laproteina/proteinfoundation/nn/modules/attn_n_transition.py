import torch

from proteinfoundation.nn.modules.pair_bias_attn import MultiHeadBiasedAttentionADALN_MM
from proteinfoundation.nn.modules.seq_transition_af3 import TransitionADALN


class MultiheadAttnAndTransition(torch.nn.Module):
    """Layer that applies mha and transition to a sequence representation. Both layers are their adaptive versions
    which rely on conditining variables (see above).

    Args:
        dim_token: Token dimension in sequence representation.
        dim_pair: Dimension of pair representation.
        nheads: Number of attention heads.
        dim_cond: Dimension of conditioning variables.
        residual_mha: Whether to use a residual connection in the mha layer.
        residual_transition: Whether to use a residual connection in the transition layer.
        parallel_mha_transition: Whether to run mha and transition in parallel or sequentially.
        use_attn_pair_bias: Whether to use a pair represnetation to bias attention.
        use_qkln: Whether to use layer norm on keyus and queries for attention.
        dropout: droput use in the self-attention layer.
    """

    def __init__(
        self,
        dim_token,
        dim_pair,
        nheads,
        dim_cond,
        residual_mha,
        residual_transition,
        parallel_mha_transition,
        use_attn_pair_bias,
        use_qkln,
        dropout=0.0,
        expansion_factor=4,
        use_relpos_bias=False,
        relpos_max=256,
    ):
        super().__init__()
        self.parallel = parallel_mha_transition
        self.use_attn_pair_bias = use_attn_pair_bias
        self.use_relpos_bias = use_relpos_bias
        self.relpos_max = relpos_max

        # If parallel do not allow both layers to have a residual connection since it leads to adding x twice
        if self.parallel and residual_mha and residual_transition:
            residual_transition = False

        self.residual_mha = residual_mha
        self.residual_transition = residual_transition

        self.mhba = MultiHeadBiasedAttentionADALN_MM(
            dim_token=dim_token,
            dim_pair=dim_pair,
            nheads=nheads,
            dim_cond=dim_cond,
            use_qkln=use_qkln,
        )
        self.nheads = nheads
        if self.use_relpos_bias:
            self.relpos_bias_embed = torch.nn.Embedding(2 * relpos_max + 1, nheads)

        self.transition = TransitionADALN(
            dim=dim_token, dim_cond=dim_cond, expansion_factor=expansion_factor
        )

    def _compute_relpos_bias(self, residue_pdb_idx, mask):
        if residue_pdb_idx is None:
            b, n = mask.shape
            residue_pdb_idx = (
                torch.arange(1, n + 1, device=mask.device, dtype=torch.long)
                .unsqueeze(0)
                .repeat(b, 1)
            )
        else:
            residue_pdb_idx = residue_pdb_idx.long()
        relpos = residue_pdb_idx[:, :, None] - residue_pdb_idx[:, None, :]
        relpos = relpos.clamp(-self.relpos_max, self.relpos_max) + self.relpos_max
        relpos_bias = self.relpos_bias_embed(relpos).permute(0, 3, 1, 2)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        relpos_bias = relpos_bias * pair_mask[:, None, :, :]
        return relpos_bias

    def _apply_mha(self, x, pair_rep, cond, mask, residue_pdb_idx=None):
        relpos_bias = None
        if self.use_relpos_bias:
            relpos_bias = self._compute_relpos_bias(residue_pdb_idx, mask)
        x_attn = self.mhba(
            x, pair_rep, cond, mask, relpos_bias=relpos_bias
        )
        if self.residual_mha:
            x_attn = x_attn + x
        return x_attn * mask[..., None]

    def _apply_transition(self, x, cond, mask):
        x_tr = self.transition(x, cond, mask)
        if self.residual_transition:
            x_tr = x_tr + x
        return x_tr * mask[..., None]

    def forward(self, x, pair_rep, cond, mask, residue_pdb_idx=None):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: conditioning variables, shape [b, n, dim_cond]
            mask: binary mask, shape [b, n]
            pair_rep: Pair representation (if provided, if no bias will be ignored), shape [b, n, n, dim_pair] or None

        Returns:
            Updated sequence representation, shape [b, n, dim].
        """
        x = x * mask[..., None]
        if self.parallel:
            x = self._apply_mha(
                x, pair_rep, cond, mask, residue_pdb_idx=residue_pdb_idx
            ) + self._apply_transition(x, cond, mask)
        else:
            x = self._apply_mha(
                x, pair_rep, cond, mask, residue_pdb_idx=residue_pdb_idx
            )
            x = self._apply_transition(x, cond, mask)
        return x * mask[..., None]
