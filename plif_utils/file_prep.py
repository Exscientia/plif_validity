"""Fetch and prepare files before the analysis."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar

import os
import dill as pkl
import prolif as plf
from rdkit import Chem

from .system_prep import SystemPrep, MDAnalysisInferer
from .settings import settings, DATA_DIR


@dataclass
class FilePrep:
    system_prep_cls: ClassVar[type[SystemPrep]] = SystemPrep

    protein_file: str
    prepared_protein_file: str
    ligand_file: str
    prepared_ligand_file: str

    @property
    def prepared_files_exist(self) -> bool:
        return os.path.exists(self.prepared_ligand_file) and os.path.exists(
            self.prepared_protein_file
        )

    @staticmethod
    def ensure_parent_directories(path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def load_ligand(self) -> plf.Molecule:
        return self.system_prep_cls.ligand_from_file(self.prepared_ligand_file)

    def load_protein(self) -> plf.Molecule:
        with open(self.prepared_protein_file, "rb") as fh:
            return plf.Molecule(pkl.load(fh))

    def write_ligand(self, ligand: Chem.Mol) -> None:
        with Chem.SDWriter(self.prepared_ligand_file) as w:
            w.write(ligand)

    def write_protein(self, protein: plf.Molecule) -> None:
        with open(self.prepared_protein_file, "wb") as fh:
            pkl.dump(protein, fh)

    def split_complex(
        self,
        complex_file: str,
        bond_order_infering: Callable[[Chem.Mol], Chem.Mol] = MDAnalysisInferer(
            reorder=False, sanitize=True
        ),
    ) -> None:
        if not (os.path.exists(self.ligand_file) and os.path.exists(self.protein_file)):
            ligand, protein = self.system_prep_cls.split_pdb(
                complex_file, bond_order_infering=bond_order_infering
            )
            self.ensure_parent_directories(self.protein_file)
            Chem.MolToPDBFile(protein, self.protein_file, flavor=2)
            self.ensure_parent_directories(self.ligand_file)
            with Chem.SDWriter(self.ligand_file) as w:
                w.write(ligand)

    def pocket_pkl_to_pdb(self, overwrite: bool = False) -> str:
        protein = self.load_protein()
        output = Path(self.prepared_protein_file).with_suffix(".pdb")
        if overwrite or not output.exists():
            Chem.MolToPDBFile(protein, output.as_posix())
        return output.as_posix()

    def prepare(self, system_prep: SystemPrep) -> None:
        if not self.prepared_files_exist:
            ligand, protein = system_prep.prepare(self.ligand_file, self.protein_file)
            self.ensure_parent_directories(self.prepared_ligand_file)
            self.write_ligand(ligand)
            self.ensure_parent_directories(self.prepared_protein_file)
            self.write_protein(protein)


def assert_path_exists(path: Path) -> None:
    if not path.exists():
        raise OSError(f"File error: Bad input file {path!s}")


def get_files(
    target: str, method: str, system_prep: SystemPrep | None = None
) -> FilePrep:
    """Returns a `FilePrep` object containing"""
    prepared_suffix = settings.prepared_files_suffix

    if method == "xtal":
        folder = DATA_DIR / "posebusters_benchmark_set" / target
        protein_file = folder / f"{target}_spruced.pdb"
        assert_path_exists(protein_file)
        ligand_file = folder / f"{target}_ligand.sdf"
        assert_path_exists(ligand_file)
        file_prep = FilePrep(
            protein_file=protein_file.as_posix(),
            ligand_file=ligand_file.as_posix(),
            prepared_protein_file=(folder / f"protein{prepared_suffix}.pkl").as_posix(),
            prepared_ligand_file=(folder / f"ligand{prepared_suffix}.sdf").as_posix(),
        )

    elif method in {"gold", "fred", "hybrid"}:
        protein_file = (
            DATA_DIR / "posebusters_benchmark_set" / target / f"{target}_spruced.pdb"
        )
        assert_path_exists(protein_file)
        ligand_file = DATA_DIR / method / target / f"docked_{target}_selected.sdf"
        assert_path_exists(ligand_file)
        file_prep = FilePrep(
            protein_file=protein_file.as_posix(),
            ligand_file=ligand_file.as_posix(),
            prepared_protein_file=(
                DATA_DIR / f"{method}{prepared_suffix}" / target / "protein.pkl"
            ).as_posix(),
            prepared_ligand_file=(
                DATA_DIR / f"{method}{prepared_suffix}" / target / "ligand.sdf"
            ).as_posix(),
        )

    elif method == "diffdock":
        protein_file = (
            DATA_DIR / "posebusters_benchmark_set" / target / f"{target}_spruced.pdb"
        )
        assert_path_exists(protein_file)
        ligand_file = DATA_DIR / method / target / "rank1.sdf"
        assert_path_exists(ligand_file)
        file_prep = FilePrep(
            protein_file=protein_file.as_posix(),
            ligand_file=ligand_file.as_posix(),
            prepared_protein_file=(
                DATA_DIR / f"{method}{prepared_suffix}" / target / "protein.pkl"
            ).as_posix(),
            prepared_ligand_file=(
                DATA_DIR / f"{method}{prepared_suffix}" / target / "ligand.sdf"
            ).as_posix(),
        )

    elif method == "rfaa":
        folder = DATA_DIR / method / target
        assert_path_exists(folder)
        protein_file = folder / f"{target}_postprocessed_protein_min.pdb"
        if not protein_file.exists():
            protein_file = folder / f"{target}_postprocessed_protein.pdb"
        assert_path_exists(protein_file)
        ligand_file = folder / f"{target}_postprocessed_ligand_min.mol2"
        if not ligand_file.exists():
            ligand_file = folder / f"{target}_postprocessed_ligand.mol2"
        assert_path_exists(ligand_file)
        file_prep = FilePrep(
            protein_file=protein_file.as_posix(),
            ligand_file=ligand_file.as_posix(),
            prepared_protein_file=(
                DATA_DIR / f"{method}{prepared_suffix}" / target / "protein.pkl"
            ).as_posix(),
            prepared_ligand_file=(
                DATA_DIR / f"{method}{prepared_suffix}" / target / "ligand.sdf"
            ).as_posix(),
        )

    elif method == "umol":
        folder = DATA_DIR / method / target
        complex_file = folder / f"{target}_relaxed_complex.pdb"
        assert_path_exists(complex_file)
        # created by split_complex method
        ligand_file = folder / f"{target}_relaxed_complex_ligand.sdf"
        protein_file = folder / f"{target}_relaxed_complex_protein.pdb"
        file_prep = FilePrep(
            ligand_file=ligand_file.as_posix(),
            protein_file=protein_file.as_posix(),
            prepared_protein_file=(
                DATA_DIR / f"{method}{prepared_suffix}" / target / "protein.pkl"
            ).as_posix(),
            prepared_ligand_file=(
                DATA_DIR / f"{method}{prepared_suffix}" / target / "ligand.sdf"
            ).as_posix(),
        )
        file_prep.split_complex(complex_file.as_posix())

    if system_prep:
        file_prep.prepare(system_prep)
    return file_prep
