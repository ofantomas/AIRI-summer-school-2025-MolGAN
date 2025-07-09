from multiprocessing import Pool
from functools import partial


import numpy as np
import pandas as pd
import scipy
import rdkit
import torch
import torch.nn.functional as F
from pysmiles import read_smiles
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
from rdkit.Chem import AllChem, Draw
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import confusion_matrix
from util_dir.utils_io import random_string


def get_mol(smiles_or_mol):
    """
    Loads SMILES/molecule into RDKit's object
    """
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def mapper(n_jobs):
    """
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    """
    if n_jobs == 1:

        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map


def fingerprint(
    smiles_or_mol,
    fp_type="maccs",
    dtype=None,
    morgan__r=2,
    morgan__n=1024,
    *args,
    **kwargs,
):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits

    Parameters:
        smiles: SMILES string
        type: type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """
    fp_type = fp_type.lower()
    molecule = get_mol(smiles_or_mol, *args, **kwargs)
    if molecule is None:
        return None
    if fp_type == "maccs":
        keys = MACCSkeys.GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype="uint8")
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == "morgan":
        fingerprint = np.asarray(
            Morgan(molecule, morgan__r, nBits=morgan__n), dtype="uint8"
        )
    else:
        raise ValueError("Unknown fingerprint type {}".format(fp_type))
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint


def fingerprints(smiles_mols_array, n_jobs=1, already_unique=False, *args, **kwargs):
    """
    Computes fingerprints of smiles np.array/list/pd.Series with n_jobs workers
    e.g.fingerprints(smiles_mols_array, type='morgan', n_jobs=10)
    Inserts np.NaN to rows corresponding to incorrect smiles.
    IMPORTANT: if there is at least one np.NaN, the dtype would be float
    Parameters:
        smiles_mols_array: list/array/pd.Series of smiles or already computed
            RDKit molecules
        n_jobs: number of parralel workers to execute
        already_unique: flag for performance reasons, if smiles array is big
            and already unique. Its value is set to True if smiles_mols_array
            contain RDKit molecules already.
    """
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.values
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array, return_inverse=True)

    fps = mapper(n_jobs)(partial(fingerprint, *args, **kwargs), smiles_mols_array)

    length = 1
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps = [
        fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :]
        for fp in fps
    ]
    if scipy.sparse.issparse(first_fp):
        fps = scipy.sparse.vstack(fps).tocsr()
    else:
        fps = np.vstack(fps)
    if not already_unique:
        return fps[inv_index]
    return fps


def average_agg_tanimoto(
    stock_vecs, gen_vecs, batch_size=5000, agg="max", device="cpu", p=1
):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ["max", "mean"], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j : j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i : i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (
                (tp / (x_stock.sum(1, keepdim=True) + y_gen.sum(0, keepdim=True) - tp))
                .cpu()
                .numpy()
            )
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == "max":
                agg_tanimoto[i : i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i : i + y_gen.shape[1]], jac.max(0)
                )
            elif agg == "mean":
                agg_tanimoto[i : i + y_gen.shape[1]] += jac.sum(0)
                total[i : i + y_gen.shape[1]] += jac.shape[0]
    if agg == "mean":
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto) ** (1 / p)
    return np.mean(agg_tanimoto)


def classification_report(
    data,
    encoder,
    decoder,
    device: torch.device = torch.device("cpu"),
    sample: bool = False,
):
    """Compute and print per-class metrics for node and edge reconstruction using PyTorch models.

    Parameters
    ----------
    data : SparseMolecularDataset
        Dataset that provides `next_validation_batch()` and decoders for atoms/bonds.
    encoder : torch.nn.Module
        Trained encoder network that maps (A, F, X) -> latent z.
    decoder : torch.nn.Module
        Decoder (a.k.a. generator) that maps latent z -> (edge_logits, node_logits).
    device : torch.device, optional
        CUDA / CPU device where the networks live. Default: CPU.
    sample : bool, optional
        If True uses Gumbel-softmax (hard) sampling before argmax, otherwise takes argmax over raw logits.
    """
    encoder.eval()
    decoder.eval()

    # Grab one full validation batch (no batch_size argument returns the whole set)
    _, _, _, a, x, _, f, _, _ = data.next_validation_batch()

    # Convert numpy arrays -> torch tensors on the correct device
    a = torch.from_numpy(a).to(device).long()  # (B, V, V)
    x = torch.from_numpy(x).to(device).long()  # (B, V)
    f = torch.from_numpy(f).to(device).float()  # (B, V, F)

    # One-hot encode adjacency & node labels so that they match training inputs
    a_onehot = F.one_hot(a, num_classes=data.bond_num_types).float()  # (B, V, V, Eb)
    x_onehot = F.one_hot(x, num_classes=data.atom_num_types).float()  # (B, V, Ea)

    with torch.no_grad():
        # Encode -> sample latent -> Decode
        z, _, _ = encoder(a_onehot, f, x_onehot)
        edges_logits, nodes_logits = decoder(z)

        # edges_logits : (B, V, V, Eb)
        # nodes_logits : (B, V, Ea)

        if sample:
            # Gumbel-softmax sampling (hard) before argmax, similar to TF version
            edges_hat = F.gumbel_softmax(
                edges_logits.view(-1, edges_logits.size(-1)), hard=True
            )
            edges_hat = edges_hat.view_as(edges_logits)
            nodes_hat = F.gumbel_softmax(
                nodes_logits.view(-1, nodes_logits.size(-1)), hard=True
            )
            nodes_hat = nodes_hat.view_as(nodes_logits)
            e_pred = edges_hat.argmax(dim=-1).cpu().numpy()
            n_pred = nodes_hat.argmax(dim=-1).cpu().numpy()
        else:
            e_pred = edges_logits.argmax(dim=-1).cpu().numpy()
            n_pred = nodes_logits.argmax(dim=-1).cpu().numpy()

    # Ground-truth labels
    e_true = a.cpu().numpy()
    n_true = x.cpu().numpy()

    # ---------- Edge-type metrics ----------
    y_true_e = e_true.flatten()
    y_pred_e = e_pred.flatten()
    bond_target_names = [
        str(Chem.rdchem.BondType.values[int(e)]) for e in data.bond_decoder_m.values()
    ]

    print("######## Bond Classification Report ########\n")
    print(
        sk_classification_report(
            y_true_e,
            y_pred_e,
            labels=list(range(len(bond_target_names))),
            target_names=bond_target_names,
        )
    )
    print("######## Bond Confusion Matrix ########\n")
    print(
        confusion_matrix(y_true_e, y_pred_e, labels=list(range(len(bond_target_names))))
    )

    # ---------- Atom-type metrics ----------
    y_true_n = n_true.flatten()
    y_pred_n = n_pred.flatten()
    atom_target_names = [Chem.Atom(e).GetSymbol() for e in data.atom_decoder_m.values()]

    print("\n######## Atom Classification Report ########\n")
    print(
        sk_classification_report(
            y_true_n,
            y_pred_n,
            labels=list(range(len(atom_target_names))),
            target_names=atom_target_names,
        )
    )
    print("\n######## Atom Confusion Matrix ########\n")
    print(
        confusion_matrix(y_true_n, y_pred_n, labels=list(range(len(atom_target_names))))
    )


def reconstructions(
    data,
    encoder,
    decoder,
    device: torch.device = torch.device("cpu"),
    batch_dim: int = 10,
    sample: bool = False,
):
    """Return a flattened list [orig1, recon1, orig2, recon2, ...] for a training batch.

    Parameters
    ----------
    data : SparseMolecularDataset
        Dataset providing `next_train_batch` and `matrices2mol`.
    encoder : torch.nn.Module
        Trained encoder network that maps (A, F, X) -> latent z.
    decoder : torch.nn.Module
        Decoder (generator) mapping z → (edge_logits, node_logits).
    device : torch.device, optional
        Device to run inference on.
    batch_dim : int, optional
        Batch size for the reconstruction set.
    sample : bool, optional
        If True, apply hard Gumbel-Softmax before argmax.
    """
    # Get batch of real molecules
    m0, _, _, a, x, _, f, _, _ = data.next_train_batch(batch_dim)

    # Convert to tensors
    a_t = torch.from_numpy(a).to(device).long()
    x_t = torch.from_numpy(x).to(device).long()
    f_t = torch.from_numpy(f).to(device).float()

    a_onehot = F.one_hot(a_t, num_classes=data.bond_num_types).float()
    x_onehot = F.one_hot(x_t, num_classes=data.atom_num_types).float()

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z, _, _ = encoder(a_onehot, f_t, x_onehot)
        edges_logits, nodes_logits = decoder(z)

    # Post-processing
    if sample:
        edges_hat = F.gumbel_softmax(
            edges_logits.view(-1, edges_logits.size(-1)), hard=True
        ).view_as(edges_logits)
        nodes_hat = F.gumbel_softmax(
            nodes_logits.view(-1, nodes_logits.size(-1)), hard=True
        ).view_as(nodes_logits)
        e_pred = edges_hat.argmax(dim=-1).cpu().numpy()
        n_pred = nodes_hat.argmax(dim=-1).cpu().numpy()
    else:
        e_pred = edges_logits.argmax(dim=-1).cpu().numpy()
        n_pred = nodes_logits.argmax(dim=-1).cpu().numpy()

    generated_mols = [
        data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n_pred, e_pred)
    ]
    m1 = np.array([mol if mol is not None else Chem.RWMol() for mol in generated_mols])
    mols = np.vstack((m0, m1)).T.flatten()
    return mols


def samples(
    data,
    decoder,
    embeddings,
    device: torch.device = torch.device("cpu"),
    sample: bool = False,
):
    """Generate molecules from latent `embeddings` with a PyTorch decoder.

    Parameters
    ----------
    data : SparseMolecularDataset
        Provides `matrices2mol` helper.
    decoder : torch.nn.Module
        Generator/decoder returning (edge_logits, node_logits).
    embeddings : np.ndarray | torch.Tensor
        Latent vectors of shape (B, z_dim).
    device : torch.device, optional
        Execution device.
    sample : bool, optional
        If True, apply hard Gumbel-Softmax sampling before argmax.
    """
    # Prepare latent vectors
    if isinstance(embeddings, np.ndarray):
        z = torch.from_numpy(embeddings).to(device).float()
    else:
        z = embeddings.to(device)

    decoder.eval()
    with torch.no_grad():
        edges_logits, nodes_logits = decoder(z)

        if sample:
            edges_hat = F.gumbel_softmax(
                edges_logits.view(-1, edges_logits.size(-1)), hard=True
            ).view_as(edges_logits)
            nodes_hat = F.gumbel_softmax(
                nodes_logits.view(-1, nodes_logits.size(-1)), hard=True
            ).view_as(nodes_logits)
            e_pred = edges_hat.argmax(dim=-1).cpu().numpy()
            n_pred = nodes_hat.argmax(dim=-1).cpu().numpy()
        else:
            e_pred = edges_logits.argmax(dim=-1).cpu().numpy()
            n_pred = nodes_logits.argmax(dim=-1).cpu().numpy()

    mols = [data.matrices2mol(n_, e_, strict=True) for n_, e_ in zip(n_pred, e_pred)]
    return mols


def save_mol_img(mols, f_name="tmp.png", is_test=False):
    orig_f_name = f_name
    for a_mol in mols:
        try:
            if Chem.MolToSmiles(a_mol) is not None:
                print("Generating molecule")

                if is_test:
                    f_name = orig_f_name
                    f_split = f_name.split(".")
                    f_split[-1] = random_string() + "." + f_split[-1]
                    f_name = "".join(f_split)

                rdkit.Chem.Draw.MolToFile(a_mol, f_name)
                a_smi = Chem.MolToSmiles(a_mol)
                _ = read_smiles(a_smi)

                break

                # if not is_test:
                #     break
        except Exception as e:
            print(e)
            continue


def mols2grid_image(mols, molsPerRow):
    mols = [e if e is not None else Chem.RWMol() for e in mols]

    for mol in mols:
        AllChem.Compute2DCoords(mol)

    return Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, subImgSize=(150, 150))
