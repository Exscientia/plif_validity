# PLIF Validity

This repository contains the code and notebook to reproduce the PLIF analysis
experiments in the paper
[*"Assessing interaction recovery of predicted protein-ligand poses"*](https://arxiv.org/abs/2409.20227) [arXiv:2409.20227].

## Structure

### Code

The code to calculate the PLIF recovery rates can be found under the `plif_utils` folder:

- `system_prep`: for preparing the ligand and protein (binding pocket) objects that will
  be used for generating the PLIFs,
- `file_prep`: for loading files and running the above system preparation code,
- `analysis`: for generating the PLIFs and calculating the PLIF recovery rate from the
  prepared files,
- `settings`: for controlling some optional settings (interaction types investigated,
  number of minimisation steps, suffix of output files...etc.).

### Data

The `data` folder only contains the identifiers from the PoseBusters benchmark study. We
also provide the corresponding protein PDB files preprocessed with Spruce and ligand
SDF, these can be downloaded on [Zenodo](https://doi.org/10.5281/zenodo.13851241).

After running the docking experiments, place the resulting poses in a
`data/${docking_method}/${posebusters_id}/` folder. More details on the expected file
structure can be found in the `plif_utils.file_prep.get_files` function.

## Setting up the environment

Install Python 3.11 in your virtual environment of choice and run the following command:
```
pip install -r requirements.txt
```

You will also need Jupyter notebook installed if you wish to run the notebook directly.

## Running the analysis code

Open the `notebooks/plif_analysis.ipynb` file with Jupyter notebook and run all cells.

## Citing us

```
@misc{errington2024assessinginteractionrecoverypredicted,
  title         = {Assessing interaction recovery of predicted protein-ligand poses},
  author        = {David Errington and Constantin Schneider and Cédric Bouysset and Frédéric A. Dreyer},
  year          = {2024},
  url           = {https://arxiv.org/abs/2409.20227},
  eprint        = {2409.20227},
  archiveprefix = {arXiv},
  primaryclass  = {q-bio.BM}
}
```