"""Prepare the ligand and protein for ProLIF."""

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterator, Optional, Callable

import numpy as np
import pdbinf
import prolif as plf
from MDAnalysis.lib.util import inverse_aa_codes
from MDAnalysis.converters.RDKit import (
    PERIODIC_TABLE,
    MONATOMIC_CATION_CHARGES,
    _infer_bo_and_charges,
    _standardize_patterns,
)
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdForceFieldHelpers
from rdkit.ForceField import rdForceField
from scipy.spatial import cKDTree

from .settings import settings


@dataclass
class MDAnalysisInferer:
    """Assign bond orders and charges using MDAnalysis' RDKitConverter module.

    Notes:
        Ported from https://github.com/MDAnalysis/mdanalysis/pull/4305 until it is
        merged and released.
    """

    reorder: bool = True
    sanitize: bool = True

    def __call__(self, mol: Chem.Mol) -> Chem.Mol:
        try:
            _infer_bo_and_charges(mol)
            mol = _standardize_patterns(mol)
        except Exception:
            if self.sanitize:
                raise
        if self.reorder:
            order = np.argsort(
                [atom.GetUnsignedProp("_idx") for atom in mol.GetAtoms()]
            )
            mol = Chem.RenumberAtoms(mol, order.astype(int).tolist())
        return mol


@dataclass
class SystemPrep:
    """Prepares the pocket of the complex for ProLIF.

    Args:
        pocket_cutoff: To limit the chance of errors while preparing the structure, the
            output protein molecule is restricted to residues within this distance of
            the ligand molecule.
        sanitize: Whether to catch errors during sanitization of the prepared molecule.
        non_standard_method: Callable used to determine bond-orders for non-standard
            residues. ``None`` will strip any atoms belonging to residues with
            non-standard names.
        radical_replaces_charge: PDB names of atoms that will bear a radical electron
            instead of a charge in residue fragments (N and C in peptide bond, and OXT
            by default). Any atom that doesn't have a filled valence shell is assigned
            one or more negative charges if they are not in this list.
        strip_invalid: Whether to strip atoms of non-standard residues for which bond
            orders could not be calculated when specifying a ``non_standard_method``
            other than ``None``.
        optimize_hydrogens: Run a short MMFF minimisation with heavy atoms fixed to
            optimize the hydrogen bond network.
    """

    pocket_cutoff: float = 6.0
    sanitize: bool = True
    non_standard_method: Optional[Callable[[Chem.Mol], Chem.Mol]] = field(
        default_factory=MDAnalysisInferer
    )
    radical_replaces_charge: set[str] = field(default_factory=lambda: {"C", "N", "OXT"})
    strip_invalid: bool = True
    optimize_hydrogens: bool = True

    def _fix_non_standard(
        self,
        residues: defaultdict[plf.ResidueId, list[Chem.Atom]],
    ) -> Iterator[tuple[plf.ResidueId, list[Chem.Atom]]]:
        for resid, atoms in residues.items():
            if resid.name == "HIS":
                # only HIS and HID have templates, the rest will use the RDKitConverter
                # from MDAnalysis
                names = {
                    atom.GetPDBResidueInfo().GetName().strip(): atom for atom in atoms
                }
                rename_resid = None
                rename_atoms = None
                if len({"1HD2", "2HD2"}.intersection(names)) == 2:
                    # weird non-aromatic histidine with sp3 CD2 from yasara minimization
                    warnings.warn(f"Non-aromatic histidine {resid!s} detected, fixing.")
                    # delete extra H on CD2, reposition remaining one on ring plane
                    # with corrected distance, and rename atoms to correspond to
                    # their actual position in the ring system
                    if "2HD2" in names:
                        self._skipped.append(names["2HD2"])
                    if "HG" in names:
                        self._skipped.append(names["HG"])
                    if "HE2" in names:
                        rename_resid = "HID"

                    # modify 1HD2 position so that distance CD2-1HD2 = 1.09
                    conf = names["1HD2"].GetOwningMol().GetConformer()
                    coords = np.array(
                        [
                            conf.GetAtomPosition(names["1HD2"].GetIdx()),
                            conf.GetAtomPosition(names["2HD2"].GetIdx()),
                        ]
                    )
                    cd2xyz = np.array(conf.GetAtomPosition(names["CD2"].GetIdx()))
                    v = np.mean(coords, axis=0) - cd2xyz
                    xyz = cd2xyz + (1.09 * v / np.linalg.norm(v))
                    conf.SetAtomPosition(names["1HD2"].GetIdx(), xyz.tolist())

                    rename_atoms = {
                        "1HD2": " HD2",
                        "ND1": " NE2",
                        "NE2": " ND1",
                        "HD1": " HE2",
                        "HE2": " HD1",
                    }

                elif "HD1" in names:
                    if "HE2" in names:
                        resid.name = "HIP"
                        rename_resid = "HIP"
                    else:
                        resid.name = "HID"
                        rename_resid = "HID"

                # check for inversion of ND1 and NE2 by yasara...
                if rename_atoms is None:
                    conf = names["CG"].GetOwningMol().GetConformer()
                    cg_nd1_dist = conf.GetAtomPosition(names["CG"].GetIdx()).Distance(
                        conf.GetAtomPosition(names["ND1"].GetIdx())
                    )
                    if cg_nd1_dist >= 2:
                        rename_atoms = {
                            "ND1": " NE2",
                            "HD1": " HE2",
                            "NE2": " ND1",
                            "HE2": " HD1",
                        }

                if rename_resid:
                    for atom in atoms:
                        atom.GetPDBResidueInfo().SetResidueName(rename_resid)
                        atom.SetFormalCharge(0)

                if rename_atoms:
                    for atom_name, new_name in rename_atoms.items():
                        if atom := names.get(atom_name):
                            atom.GetPDBResidueInfo().SetName(new_name)

            yield resid, atoms

    def group_by_resid(
        self, mol: Chem.Mol, subset: Optional[set[plf.ResidueId]] = None
    ) -> dict[plf.ResidueId, list[Chem.Atom]]:
        residues: defaultdict[plf.ResidueId, list[Chem.Atom]] = defaultdict(list)
        for atom in mol.GetAtoms():
            resid = plf.ResidueId.from_atom(atom)
            if subset is None or resid in subset:
                residues[resid].append(atom)
        return dict(self._fix_non_standard(residues))

    @staticmethod
    def mol_from_subset(refmol: Chem.Mol, atoms: list[Chem.Atom]) -> Chem.Mol:
        mw = Chem.RWMol()
        refconf = refmol.GetConformer()
        conf = Chem.Conformer(len(atoms))
        for atom in atoms:
            ix = mw.AddAtom(atom)
            mw.GetAtomWithIdx(ix).SetUnsignedProp("_idx", atom.GetIdx())
            conf.SetAtomPosition(ix, refconf.GetAtomPosition(atom.GetIdx()))
        mol = mw.GetMol()
        mol.AddConformer(conf, assignId=True)
        return mol

    def _subset_from_residues(
        self, mol: Chem.Mol, residues: set[plf.ResidueId]
    ) -> Chem.Mol:
        residues_to_atoms = self.group_by_resid(mol, subset=residues)
        atoms_subset = []
        for resid in residues:
            try:
                atoms = residues_to_atoms[resid]
            except KeyError:
                # HIS residue in input file might have been renamed to HID/HIP
                if resid.name != "HIS":
                    raise
                for resname in ("HID", "HIP"):
                    try:
                        atoms = residues_to_atoms[
                            plf.ResidueId(resname, resid.number, resid.chain)
                        ]
                    except KeyError:
                        continue
                    break
                else:
                    raise
            atoms_subset.extend(atoms)
        skipped = {a.GetIdx() for a in self._skipped}
        atoms_subset = list(filter(lambda a: a.GetIdx() not in skipped, atoms_subset))
        self._skipped.clear()
        return self.mol_from_subset(mol, atoms_subset)

    def subset_around_ligand(
        self, prot_mol: Chem.Mol, ligand_mol: plf.Molecule
    ) -> Chem.Mol:
        tree = cKDTree(prot_mol.GetConformer().GetPositions())
        ix = tree.query_ball_point(ligand_mol.xyz, self.pocket_cutoff)
        ix = {i for lst in ix for i in lst}
        pocket_resids = {
            plf.ResidueId.from_atom(prot_mol.GetAtomWithIdx(i)) for i in ix
        }
        return self._subset_from_residues(prot_mol, pocket_resids)

    def infer_bond_orders_subset(
        self, mol: Chem.Mol, atoms: list[Chem.Atom]
    ) -> Chem.Mol:
        bond: Chem.Bond
        if self.non_standard_method is None:
            self._skipped.extend(atoms)
            return mol

        # make separate subset mol
        subset = self.mol_from_subset(mol, atoms)
        # assign bonds
        rdDetermineBonds.DetermineConnectivity(subset)
        # assign bond orders
        try:
            subset = self.non_standard_method(subset)
        except Exception:
            if not self.strip_invalid:
                raise
            warnings.warn(
                "Unable to determine bond orders for"
                f" {plf.ResidueId.from_atom(atoms[0])!s}, stripping it."
            )
            self._skipped.extend(atoms)
            return mol

        # sanity check: charged carbon atom
        if any(
            atom.GetAtomicNum() == 6 and atom.GetFormalCharge() != 0
            for atom in subset.GetAtoms()
        ):
            msg = (
                "Invalid bond orders detected for"
                f" {plf.ResidueId.from_atom(atoms[0])!s}:"
                f" SMILES={Chem.MolToSmiles(subset)!r}"
            )
            if not self.strip_invalid:
                raise ValueError(msg)
            warnings.warn(f"{msg}, stripping it.")
            self._skipped.extend(atoms)
            return mol

        # transfer to mol
        refmol = Chem.RWMol(mol)
        for bond in subset.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            refmol.AddBond(
                a1.GetUnsignedProp("_idx"),
                a2.GetUnsignedProp("_idx"),
                bond.GetBondType(),
            )
        return refmol.GetMol()

    def assign_bond_orders(
        self, mol: Chem.Mol, resid: plf.ResidueId, atoms_subset: list[Chem.Atom]
    ) -> Chem.Mol:
        # reuse HIS block for HIP
        resname = "HIS" if resid.name == "HIP" else resid.name
        try:
            # infer sidechain bonds from template
            block = pdbinf.STANDARD_AA_DOC[resname]
        except KeyError:
            # skip ions
            if len(atoms_subset) == 1:
                # strip lone hydrogens and other atoms that aren't delt with by rdkit
                if self.strip_invalid and (
                    (ion := atoms_subset[0]).GetAtomicNum() == 1
                    or PERIODIC_TABLE.GetDefaultValence(ion.GetAtomicNum()) == -1
                ):
                    warnings.warn(
                        f"Found lone {ion.GetSymbol()} atom for residue {resid!s},"
                        " stripping it."
                    )
                    self._skipped.append(ion)
                return mol
            # infer from topology with explicit Hs
            mol = self.infer_bond_orders_subset(mol, atoms_subset)
        else:
            span = [a.GetIdx() for a in atoms_subset]
            mol, _ = pdbinf._pdbinf.assign_intra_props(mol, span, block)
        mol.UpdatePropertyCache(False)
        return mol

    def assign_charges(self, mol: Chem.Mol) -> None:
        cysteines = {"CYS", "CYX"}

        for atom in mol.GetAtoms():
            atom.SetNoImplicit(True)
            mi = atom.GetPDBResidueInfo()
            pdb_name = mi.GetName().strip()

            if pdb_name in self.radical_replaces_charge:
                unpaired = (
                    PERIODIC_TABLE.GetDefaultValence(atom.GetAtomicNum())
                    - atom.GetTotalValence()
                )
                if unpaired < 0:
                    # with RFAA, N-term nitrogen is not capped and has 3 explicit H
                    # so require a formal charge
                    atom.SetFormalCharge(-unpaired)
                else:
                    atom.SetFormalCharge(0)
                    atom.SetNumRadicalElectrons(unpaired)

            elif pdb_name == "SG" and mi.GetResidueName() in cysteines:
                resid = plf.ResidueId.from_atom(atom)
                for na in atom.GetNeighbors():
                    if (
                        nr := plf.ResidueId.from_atom(na)
                    ) != resid and nr.name in cysteines:
                        # S involved in cysteine bridge shouldn't be charged
                        atom.SetFormalCharge(0)
                        atom.SetNumRadicalElectrons(1)

            elif pdb_name == "ND1" and mi.GetResidueName() == "HIP":
                atom.SetFormalCharge(1)

            else:
                if (
                    atom.GetDegree() == 0
                    and atom.GetAtomicNum() in MONATOMIC_CATION_CHARGES
                ):
                    chg = MONATOMIC_CATION_CHARGES[atom.GetAtomicNum()]
                else:
                    chg = atom.GetTotalValence() - PERIODIC_TABLE.GetDefaultValence(
                        atom.GetAtomicNum()
                    )
                atom.SetFormalCharge(chg)
                atom.SetNumRadicalElectrons(0)

            mol.UpdatePropertyCache(False)

    @classmethod
    def split_pdb(
        cls, complex_file: str, bond_order_infering: Callable[[Chem.Mol], Chem.Mol]
    ) -> tuple[Chem.Mol, Chem.Mol]:
        cpx = Chem.MolFromPDBFile(
            complex_file, removeHs=False, sanitize=False, proximityBonding=False
        )
        for atom in cpx.GetAtoms():
            atom.SetNoImplicit(True)
        # split fragments on residue names
        unknown, prot_frags = [], []
        for resname, frag in Chem.SplitMolByPDBResidues(cpx).items():
            if resname in inverse_aa_codes:
                prot_frags.append(frag)
            else:
                unknown.append(frag)
        # identify ligand
        if not unknown:
            raise ValueError("No ligand found in complex PDB file")
        ligand = max(unknown, key=lambda m: m.GetNumAtoms())
        # assign bonds and bond orders for ligand
        rdDetermineBonds.DetermineConnectivity(ligand)
        ligand = bond_order_infering(ligand)
        for atom in ligand.GetAtoms():
            atom.SetNumRadicalElectrons(0)
        Chem.AssignStereochemistryFrom3D(ligand)
        Chem.SanitizeMol(ligand)
        # recombine protein
        protein = prot_frags.pop()
        if prot_frags:
            for frag in prot_frags:
                protein = Chem.CombineMols(protein, frag)
        return ligand, protein

    def mmff_optimize_hydrogens(
        self,
        ligand_mol: plf.Molecule,
        pocket_mol: plf.Molecule,
    ):
        # create complex from pocket residues and ligand
        cpx = Chem.CombineMols(ligand_mol, pocket_mol)
        Chem.SanitizeMol(cpx)
        # parametrize ff
        props: rdForceField.MMFFMolProperties = (
            rdForceFieldHelpers.MMFFGetMoleculeProperties(cpx, "MMFF94s")
        )
        ff: rdForceField.ForceField = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            cpx, props, ignoreInterfragInteractions=False
        )
        # constrain position of heavy atoms and certain hydrogens
        num_ligand_atoms = ligand_mol.GetNumAtoms()
        for atom in cpx.GetAtoms():
            # constrain heavy atom
            if atom.GetAtomicNum() != 1:
                ff.AddFixedPoint(atom.GetIdx())
            # constrain hydrogen bound to backbone nitrogen fragment
            elif (
                atom.GetNeighbors()[0].GetPDBResidueInfo().GetName().strip() == "N"
                and atom.GetIdx() >= num_ligand_atoms
            ):
                ff.AddFixedPoint(atom.GetIdx())
        # minimize
        ff.Initialize()
        ff.Minimize(maxIts=settings.max_minimization_iterations)
        # update coordinates
        cpx_conf = cpx.GetConformer()
        lig_conf = ligand_mol.GetConformer()
        prot_conf = pocket_mol.GetConformer()
        lig_idx = 0
        prot_idx = 0
        for atom in cpx.GetAtoms():
            idx = atom.GetIdx()
            xyz = cpx_conf.GetAtomPosition(idx)
            if idx < num_ligand_atoms:
                lig_conf.SetAtomPosition(lig_idx, xyz)
                lig_idx += 1
            else:
                prot_conf.SetAtomPosition(prot_idx, xyz)
                prot_idx += 1

    def pocket_from_pdb(
        self, protein_file: str, ligand_mol: plf.Molecule
    ) -> plf.Molecule:
        protein_mol = Chem.MolFromPDBFile(
            protein_file, removeHs=False, proximityBonding=False
        )
        return self.pocket_from_mol(protein_mol, ligand_mol)

    def pocket_from_mol(
        self, protein_mol: Chem.Mol, ligand_mol: plf.Molecule
    ) -> plf.Molecule:
        self._skipped = []
        pocket_mol = self.subset_around_ligand(protein_mol, ligand_mol)
        assert (
            pocket_mol.GetNumAtoms() > 0
        ), f"No atoms within {self.pocket_cutoff} of ligand!"

        residues = self.group_by_resid(pocket_mol)
        for resid, atoms_subset in residues.items():
            pocket_mol = self.assign_bond_orders(pocket_mol, resid, atoms_subset)

        if self._skipped:
            with Chem.RWMol(pocket_mol) as mw:
                for atom in self._skipped:
                    mw.RemoveAtom(atom.GetIdx())
            pocket_mol = mw.GetMol()
            pocket_mol.UpdatePropertyCache(False)

        self.assign_charges(pocket_mol)
        Chem.SanitizeMol(pocket_mol, catchErrors=not self.sanitize)
        return plf.Molecule(pocket_mol)

    class mol2_supplier(plf.mol2_supplier):
        """Bypass cleanupSubstructures to avoid errors with incorrect atom types"""

        def block_to_mol(self, block):
            mol = Chem.MolFromMol2Block(
                "".join(block), removeHs=False, cleanupSubstructures=False
            )
            Chem.SanitizeMol(mol)
            return plf.Molecule.from_rdkit(mol, **self._kwargs)

    @classmethod
    def ligand_from_file(
        cls, ligand_file: str, add_hydrogens: bool = False
    ) -> plf.Molecule:
        supplier = (
            cls.mol2_supplier if ligand_file.endswith(".mol2") else plf.sdf_supplier
        )
        mol = supplier(ligand_file)[0]
        if add_hydrogens:
            resid: plf.ResidueId = mol[0].resid
            mol = Chem.AddHs(mol, addCoords=True)
            # bug with addResidueInfo in some cases so add manually
            for atom in mol.GetAtoms():
                if atom.GetPDBResidueInfo() is None:
                    mi = Chem.AtomPDBResidueInfo(
                        f" {atom.GetSymbol():<3.3}",
                        residueName=resid.name,
                        residueNumber=resid.number,
                        chainId=resid.chain or "",
                    )
                    atom.SetMonomerInfo(mi)
        return plf.Molecule(mol)

    def prepare(
        self,
        ligand_file: str,
        protein_file: str,
    ) -> tuple[plf.Molecule, plf.Molecule]:
        ligand = self.ligand_from_file(ligand_file, add_hydrogens=True)
        protein = self.pocket_from_pdb(protein_file, ligand)
        if self.optimize_hydrogens:
            self.mmff_optimize_hydrogens(protein, ligand)
        return ligand, protein
