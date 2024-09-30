"""Run the PLIF analysis"""

import warnings
from typing import NamedTuple, Iterable, TypeAlias

import os
import dill as pkl
import pandas as pd
import prolif as plf
from rdkit import Chem
from IPython.display import display
from Bio import pairwise2
from MDAnalysis.lib.util import inverse_aa_codes

from .system_prep import SystemPrep
from .file_prep import get_files, FilePrep
from .settings import settings

THREE_TO_ONE = inverse_aa_codes.copy()
THREE_TO_ONE.update({"HIP": "H"})
IFPType: TypeAlias = dict[int, dict[tuple[plf.ResidueId, plf.ResidueId], dict]]


def run(
    ligfile: str,
    protfile: str,
    viz_residues: bool = False,
    verbose: bool = False,
) -> tuple[plf.Molecule, plf.Molecule, plf.Fingerprint]:
    if verbose:
        print(ligfile)
        print(protfile)

    # read ligand SDF
    ligand_mol = SystemPrep.ligand_from_file(ligfile)
    assert any(
        atom.GetAtomicNum() == 1 for atom in ligand_mol.GetAtoms()
    ), f"Missing explicit hydrogens: {ligfile}"

    # read protein pocket
    with open(protfile, "rb") as fh:
        protein_mol = plf.Molecule(pkl.load(fh))

    # display pocket residues for sanity checking
    if viz_residues:
        view = plf.display_residues(protein_mol, mols_per_row=5)
        display(view)

    # compute ifp
    fp = plf.Fingerprint(
        interactions=settings.interactions,
        parameters=settings.interaction_parameters,
        count=True,
    )
    fp.run_from_iterable([ligand_mol], protein_mol, progress=False)
    if verbose:
        df = fp.to_dataframe()
        display(df.T)
    return ligand_mol, protein_mol, fp


def get_plifs(
    posebuster_targets: Iterable[str], method: str, system_prep: SystemPrep
) -> dict[str, plf.Fingerprint]:
    # dict to collect plif dataframes
    plifs = {}

    # loop over posebuster targets and calculate plifs
    for target in posebuster_targets:
        try:
            # get protein and ligand files
            file_prep = get_files(target, method, system_prep)
            assert os.path.exists(file_prep.prepared_ligand_file), "ligand file missing"
            assert os.path.exists(
                file_prep.prepared_protein_file
            ), "protein file missing"
        except Exception:
            warnings.warn(f"Could not prepare system for {method}::{target}")
            continue

        try:
            # construct plifs for this target
            _, _, fp = run(
                file_prep.prepared_ligand_file, file_prep.prepared_protein_file
            )
        except Exception:
            warnings.warn(f"Could not calculate PLIFs for {method}::{target}")
            continue
        plifs[target] = fp
    return plifs


def get_common_map(
    seq_a: str, seq_b: str, res_ids_a: list[str], res_ids_b: list[str]
) -> tuple[dict[str, int], dict[str, int]]:
    """Maps 2 sequences to the numbering used in sequence A."""
    map_a = {}
    map_b = {}

    # pick alignment where dashes are as far as possible from the start
    # (avoids alignments with gaps in the middle)
    alignment = max(
        pairwise2.align.globalxs(seq_a, seq_b, -0.5, -0.1),
        key=lambda a: sum((a.seqA.find("-"), a.seqB.find("-"))),
    )

    running_idx_a = 0
    running_idx_b = 0
    for i, (align_a, align_b) in enumerate(zip(alignment.seqA, alignment.seqB)):
        if align_a == align_b:
            map_a[res_ids_a[running_idx_a]] = i
            map_b[res_ids_b[running_idx_b]] = i
        if align_a != "-":
            running_idx_a += 1
        if align_b != "-":
            running_idx_b += 1
    return map_a, map_b


def adjust_fingerprint_residues(fp: plf.Fingerprint, mapper: dict[str, int]) -> IFPType:
    """Replaces protein residues in the fingerprint based on the mapping."""

    def adjust_resid(res: plf.ResidueId) -> plf.ResidueId:
        try:
            number = mapper[f"{res.number}.{res.chain}"]
            chain = "Z"
        except KeyError:
            number = res.number
            chain = res.chain
        return plf.ResidueId(THREE_TO_ONE.get(res.name, "X"), number, chain)

    return {
        0: {(lres, adjust_resid(pres)): ifp for (lres, pres), ifp in fp.ifp[0].items()}
    }


def get_sequence(file_prep: FilePrep) -> tuple[str, list[str]]:
    """
    Extracts sequence from chains that are found in the residues from the pocket.
    This avoids mapping to the wrong chain in homo-oligomers.
    """
    protein_pocket = file_prep.load_protein()
    pocket_chain = {resid.chain for resid in protein_pocket.residues}

    if pocket_chain:

        def chain_predicate(resid: plf.ResidueId) -> bool:
            return resid.chain in pocket_chain

    else:

        def chain_predicate(resid: plf.ResidueId) -> bool:
            return True

    protein = Chem.MolFromPDBFile(
        file_prep.protein_file, sanitize=False, proximityBonding=False
    )
    residues = list(
        dict.fromkeys(
            resid
            for atom in protein.GetAtoms()
            if chain_predicate(resid := plf.ResidueId.from_atom(atom))
        )
    )
    sequence = "".join(THREE_TO_ONE.get(resid.name, "X") for resid in residues)
    res_ids = [f"{resid.number}.{resid.chain}" for resid in residues]
    return sequence, res_ids


def map_fingerprints(
    fp_a: plf.Fingerprint,
    fp_b: plf.Fingerprint,
    file_prep_a: FilePrep,
    file_prep_b: FilePrep,
) -> tuple[IFPType, IFPType]:
    """
    Adjusts the fingerprints protein residues to use the same numbering based on a
    sequence alignment (in case the number of residues is not the same between the
    different methods).
    """
    # get sequence and residue ids of the chain(s) corresponding to the pocket
    seq_a, res_ids_a = get_sequence(file_prep_a)
    seq_b, res_ids_b = get_sequence(file_prep_b)

    # map
    map_a, map_b = get_common_map(seq_a, seq_b, res_ids_a, res_ids_b)

    ifp_a = adjust_fingerprint_residues(fp_a, map_a)
    ifp_b = adjust_fingerprint_residues(fp_b, map_b)
    return ifp_a, ifp_b


class PlifResults(NamedTuple):
    count_recovery: float | None = None
    xtal_plif: str = ""
    plif: str = ""


def fp_to_str(ifp_df: pd.DataFrame) -> str:
    return "/".join(
        [
            f"{'_'.join(key)}={count}"
            for key, count in ifp_df.droplevel("ligand", axis=1)
            .to_dict("records")[0]
            .items()
        ]
    )


def get_plif_recovery_rates(
    xtal_file_prep: FilePrep,
    other_file_prep: FilePrep,
    fpxtal: plf.Fingerprint,
    fpother: plf.Fingerprint,
) -> PlifResults:
    # no interaction
    if not fpxtal.ifp[0]:
        return PlifResults()
    if not fpother.ifp[0]:
        return PlifResults(0, fp_to_str(fpxtal.to_dataframe()))

    xtalifp, otherifp = map_fingerprints(
        fpxtal, fpother, xtal_file_prep, other_file_prep
    )
    xtal_df = plf.to_dataframe(xtalifp, fpxtal.interactions, count=True)
    xtal_counts = xtal_df.droplevel("ligand", axis=1).to_dict("records")[0]
    total_xtal_count = sum(xtal_counts.values())
    xtal_plifs = fp_to_str(xtal_df)

    other_df = plf.to_dataframe(otherifp, fpother.interactions, count=True)
    other_counts = other_df.droplevel("ligand", axis=1).to_dict("records")[0]
    count_recovery = (
        sum([min(xtal_counts[k], other_counts.get(k, 0)) for k in xtal_counts])
        / total_xtal_count
    )
    other_plifs = fp_to_str(other_df)
    return PlifResults(count_recovery, xtal_plifs, other_plifs)
