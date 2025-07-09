import gzip
import math
import pickle

import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Crippen
from collections import Counter
from .sacrorer import calculateScore
from utils import canonic_smiles, mapper, fingerprints, average_agg_tanimoto, get_mol

NP_model = pickle.load(gzip.open("data/NP_score.pkl.gz"))

# -----------------------------------------------------------------------------
# Generic small helpers that were previously static methods of a class
# -----------------------------------------------------------------------------


def _avoid_sanitization_error(op):
    """Safely execute *op* (typically an RDKit call) catching ValueError."""

    try:
        return op()
    except ValueError:
        return None


def remap(x, x_min, x_max):
    """Linearly remap *x* from [x_min, x_max] → [0, 1]."""

    return (x - x_min) / (x_max - x_min)


# -----------------------------------------------------------------------------
# Helper metric: *quality* (valid ∧ unique ∧ drug-like ∧ synthesizable)
# -----------------------------------------------------------------------------


def valid_scores(mols, n_jobs=1):
    rd_mols = mapper(n_jobs)(get_mol, mols)
    return np.array([mol is not None for mol in rd_mols], dtype=float)


def valid_total_score(mols, n_jobs=1):
    return valid_scores(mols, n_jobs).mean()


def unique_scores(mols, n_jobs=1):
    canonic = mapper(n_jobs)(canonic_smiles, mols)
    counts = Counter([s for s in canonic if s is not None])
    return np.array(
        [1.0 if s is not None and counts[s] == 1 else 0.0 for s in canonic],
        dtype=np.float32,
    )


def unique_total_score(mols, n_jobs=1, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        k: compute unique@k
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    canonic = set(mapper(n_jobs)(canonic_smiles, mols))
    if None in canonic and check_validity:
        raise ValueError("Invalid molecule passed to unique@k")
    return len(canonic) / len(mols)


def novel_scores(mols, data, n_jobs: int = 1):
    """Return 1 for molecules whose canonical SMILES is *not* in the reference
    dataset, else 0."""

    mols_smiles = mapper(n_jobs)(canonic_smiles, mols)
    data_smiles = set(data.smiles)
    return np.asarray(
        [1.0 if s is not None and s not in data_smiles else 0.0 for s in mols_smiles],
        dtype=np.float32,
    )


def novel_total_score(mols, data, n_jobs=1):
    mols_smiles = mapper(n_jobs)(canonic_smiles, mols)
    mols_smiles_set = set(mols_smiles) - {None}
    data_smiles = set(data.smiles)
    return len(mols_smiles_set - data_smiles) / len(mols_smiles_set)


def qed_scores(mols, n_jobs=1):
    rd_mols = mapper(n_jobs)(get_mol, mols)
    return np.array([QED.qed(mol) if mol is not None else 0.0 for mol in rd_mols])


def sa_scores(mols, n_jobs=1):
    rd_mols = mapper(n_jobs)(get_mol, mols)
    return np.array(
        [calculateScore(mol) if mol is not None else 10.0 for mol in rd_mols]
    )


def quality(mols, n_jobs: int = 1) -> float:
    """Return the fraction of molecules that are simultaneously

    1. chemically valid,
    2. unique within the *valid* subset,
    3. drug-like (QED ≥ 0.6), and
    4. synthesizable (SA ≤ 4).

    This follows the definition in *Noutahi et al., 2024* and the excerpt
    provided by the user.
    """
    validity = valid_scores(mols, n_jobs).astype(bool)
    uniqueness = unique_scores(mols, n_jobs).astype(bool)
    druglike = (qed_scores(mols, n_jobs) >= 0.6).astype(bool)
    synth = (sa_scores(mols, n_jobs) <= 4.0).astype(bool)

    return (validity & uniqueness & druglike & synth).mean()


def remove_invalid(gen, canonize=True, n_jobs=1):
    """
    Removes invalid molecules from the dataset
    """
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in mapper(n_jobs)(canonic_smiles, gen) if x is not None]


def natural_product_scores(mols):
    # calculating the score
    scores = [
        sum(
            NP_model.get(bit, 0)
            for bit in Chem.rdMolDescriptors.GetMorganFingerprint(
                mol, 2
            ).GetNonzeroElements()
        )
        / float(mol.GetNumAtoms())
        if mol is not None
        else None
        for mol in mols
    ]

    # preventing score explosion for exotic molecules
    scores = list(
        map(
            lambda score: score
            if score is None
            else (
                4 + math.log10(score - 4 + 1)
                if score > 4
                else (-4 - math.log10(-4 - score + 1) if score < -4 else score)
            ),
            scores,
        )
    )

    scores = np.array(list(map(lambda x: -4 if x is None else x, scores)))
    return scores


def water_octanol_partition_coefficient_scores(mols, norm=False):
    scores = [
        _avoid_sanitization_error(lambda: Crippen.MolLogP(mol))
        if mol is not None
        else None
        for mol in mols
    ]
    scores = np.array(list(map(lambda x: -3 if x is None else x, scores)))
    scores = (
        np.clip(remap(scores, -2.12178879609, 6.0429063424), 0.0, 1.0)
        if norm
        else scores
    )

    return scores


def internal_diversity_scores(
    mols, n_jobs=1, device="cpu", fp_type="morgan", mols_fps=None, p=1
):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if mols_fps is None:
        mols_fps = fingerprints(mols, fp_type=fp_type, n_jobs=n_jobs)
    return (
        1
        - (
            average_agg_tanimoto(mols_fps, mols_fps, agg="mean", device=device, p=p)
        ).mean()
    )


def diversity_scores(
    mols,
    data,
    n_data: int = 1000,
    n_jobs: int = 1,
    device: str = "cpu",
    fp_type: str = "morgan",
    mols_fps=None,
    data_fps=None,
    p: int = 1,
):
    """Compute *external* diversity between ``mols`` and a random subset of
    the reference ``data``.

    The metric is defined analogously to ``internal_diversity_scores`` but for
    two different sets *A* (generated molecules) and *B* (reference
    molecules):

    .. math::

        1 - \frac{1}{|A|} \sum_{x \in A} \sum_{y \in B}
                \text{Tanimoto}(x, y)

    A higher score therefore means the generated set is more diverse with
    respect to the reference set.
    """

    # ------------------------------------------------------------------
    # Prepare fingerprints for *reference* molecules
    # ------------------------------------------------------------------
    if data_fps is None:
        # Ensure we do not sample more molecules than available
        n_data = min(n_data, len(data.data))
        ref_mols = np.random.choice(data.data, n_data, replace=False)
        data_fps = fingerprints(ref_mols, fp_type=fp_type, n_jobs=n_jobs)

    # ------------------------------------------------------------------
    # Prepare fingerprints for *generated* molecules
    # ------------------------------------------------------------------
    if mols_fps is None:
        mols_fps = fingerprints(mols, fp_type=fp_type, n_jobs=n_jobs)

    # ------------------------------------------------------------------
    # Average Tanimoto similarity between the two sets
    # ------------------------------------------------------------------
    avg_sim = average_agg_tanimoto(
        data_fps, mols_fps, agg="mean", device=device, p=p
    ).mean()

    # Diversity is defined as 1 - similarity
    return 1 - avg_sim


# -----------------------------------------------------------------------------
# Novelty per-molecule helper (1 if SMILES not present in reference data)
# -----------------------------------------------------------------------------


def compute_metrics(mols, data, n_jobs: int = 1):
    """Return a dictionary of key generation metrics for ``mols``.

    The implementation mirrors common open-source toolkits (MOSES / GuacaMol)
    and uses *mapper(n_jobs)* for parallelism instead of creating explicit
    process pools.
    """

    # Keep the full list for *valid*; for the rest we can drop invalid entries
    metrics = {
        "quality": quality(mols, n_jobs) * 100,
        "valid": valid_total_score(mols, n_jobs) * 100,
    }

    # Filter out invalid molecules once to avoid redundant work downstream
    mols_valid = remove_invalid(mols, n_jobs=n_jobs)

    if len(mols_valid) == 0:
        metrics.update(
            {
                "unique": 0.0,
                "novel": 0.0,
                "qed": 0.0,
                "sa": 10.0,
                "int_div_p1": 0.0,
                "int_div_p2": 0.0,
                "diversity": 0.0,
                "solute": -3.0,
                "NP_score": -4.0,
            }
        )
    else:
        metrics.update(
            {
                "unique": unique_total_score(mols_valid, n_jobs) * 100,
                "novel": novel_total_score(mols_valid, data, n_jobs) * 100,
                "qed": qed_scores(mols_valid, n_jobs).mean(),
                "sa": sa_scores(mols_valid, n_jobs).mean(),
                "int_div_p1": internal_diversity_scores(mols_valid, n_jobs),
                "int_div_p2": internal_diversity_scores(
                    mols_valid, n_jobs, p=2, device="cpu"
                ),
                "diversity": diversity_scores(mols_valid, data, n_jobs=n_jobs),
                "solute": water_octanol_partition_coefficient_scores(mols, norm=False),
                "NP_score": natural_product_scores(mols),
            }
        )

    return metrics
