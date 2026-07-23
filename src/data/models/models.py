"""
===========================================================================
Models Data Classes
Description: Data classes for kinetic parameter records
===========================================================================

Workflow:
1. Defines the `KineticRecord` dataclass to contain parsed kinetic-parameter records.

Author: ThermoKP Team
License: MIT
"""
from dataclasses import dataclass
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
#  Data Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class KineticRecord:
    """Container for a single parsed kinetic-parameter record.

    Attributes
    ----------
    entry_id : int
        A unique integer ID assigned during extraction.
    source_db : str
        The name of the source database ("BRENDA" or "SABIO-RK").
    ec_number : str
        The Enzyme Commission number (e.g. "1.1.1.27").
    uniprot_id : str or None
        The primary UniProt accession string.
    measured_substrate : str
        The primary substrate involved in the reaction (for which Km is reported).
    co_substrates : str
        Other substrates in the reaction equation, comma-separated.
    kcat : float or None
        Turnover number in 1/s.
    km : float or None
        Michaelis constant in mM.
    temperature : float or None
        Assay temperature in °C.
    ph : float or None
        Assay pH.
    mutation : str or None
        A single conservative point-mutation code (e.g. "W95L") applied to
        the wild-type UniProt sequence for this specific assay, or None for
        wild-type. Only ever set for entries where exactly one, non-Pro/Gly,
        non-charge-reversing substitution could be parsed from the source
        commentary; anything more ambiguous is dropped upstream rather than
        risking a mismatched mutation code.
    """
    entry_id: int
    source_db: str
    ec_number: str
    uniprot_id: Optional[str]
    measured_substrate: str
    co_substrates: str
    kcat: Optional[float]
    km: Optional[float]
    temperature: Optional[float]
    ph: Optional[float]
    mutation: Optional[str] = None
