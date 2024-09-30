from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = (Path(__file__).parents[1] / "data").resolve()


@dataclass
class Settings:
    """Some common settings for the PLIF analysis.

    Attributes:
        interactions: List of interactions used in the analysis.
        interaction_parameters: Parametrization for the interactions SMARTS patterns or
            thresholds.
        max_minimization_iterations: Max number of iterations performed for the hydrogen
            network optimization step.
        prepared_files_suffix: Suffix used for the prepared output files.
    """

    interactions: list[str] = field(
        default_factory=lambda: [
            "HBDonor",
            "HBAcceptor",
            "PiStacking",
            "Anionic",
            "Cationic",
            "CationPi",
            "PiCation",
            "XBAcceptor",
            "XBDonor",
        ]
    )
    interaction_parameters: dict[str, dict] = field(
        default_factory=lambda: {
            "HBAcceptor": {
                "distance": 3.7,
                # modified nitrogen pattern (replaced valence specs with charge
                # to account for broken peptide bond during pocket extraction)
                # otherwise HBonds with backbone nitrogen wouldn't be detected
                "donor": "[$([O,S,#7;+0]),$([N+1])]-[H]",
            },
            "HBDonor": {"distance": 3.7},
            "CationPi": {"distance": 5.5},
            "PiCation": {"distance": 5.5},
            "Anionic": {"distance": 5},
            "Cationic": {"distance": 5},
        }
    )
    max_minimization_iterations: int = 200
    prepared_files_suffix: str = "_v1"


settings = Settings()
