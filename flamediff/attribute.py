"""Attribution -- *why* did embeddings drift?

Decompose the clean-survivor drift into (1) global basis drift (table-wide rotation + mean shift,
removed by orthogonal Procrustes), (2) popularity churn (regress the aligned per-id drift on
log dcount/count), and (3) the idiosyncratic residual -- the ids whose representation genuinely
changed. Correlational, validated by injection (tests/test_attribute.py, scripts/attribution_demo).

Runs at fixture scale through the EmbeddingTable Protocol -- it re-gathers the diff's survivor rows
(the diff keeps only summaries), so it works on sharded / out-of-core tables too.
"""
from __future__ import annotations

import numpy as np

from flamediff import stats
from flamediff.types import Attribution, EmbeddingTable, EmbeddingTableDiff

_MIN_SURVIVORS = 8  # below this the alignment + regression aren't meaningful


def attribute_table(
    prev: EmbeddingTable, cur: EmbeddingTable, diff: EmbeddingTableDiff
) -> Attribution:
    """Attribute the clean-survivor drift of one table between two checkpoints."""
    surv = diff.surv_ids
    if surv.size < _MIN_SURVIVORS:
        return Attribution.empty(cur.name)

    align = stats.procrustes_align(prev.gather(surv).float(), cur.gather(surv).float())

    covariates = [np.log1p(np.maximum(np.asarray(diff.dcount, dtype=np.float64), 0.0))]
    counts = cur.counts(surv)
    if counts is not None:
        covariates.append(np.log1p(np.maximum(counts.astype(np.float64), 0.0)))
    idiosyncratic, popularity_r2 = stats.loglog_residual(
        align["aligned_delta_norm"], np.column_stack(covariates)
    )

    return Attribution(
        name=cur.name,
        n=int(surv.size),
        surv_ids=surv,
        idiosyncratic=idiosyncratic,
        delta_norm=diff.delta_norm,
        dcount=diff.dcount,
        frac_translation=align["frac_translation"],
        frac_rotation=align["frac_rotation"],
        frac_aligned_residual=align["frac_aligned"],
        mean_shift_norm=align["mean_shift_norm"],
        rotation_magnitude=align["rotation_magnitude"],
        popularity_r2=popularity_r2,
    )
