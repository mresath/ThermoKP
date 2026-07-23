"""
===========================================================================
Ligand Cleaner
Description: Shared Ligand Filtering & Standardization Logic
===========================================================================

Workflow:
1. Provides the `canonicalize_and_filter_ligands` function used by both BRENDA and SABIO-RK parsers to guarantee dataset uniformity across substrates.

Author: ThermoKP Team
License: MIT
"""

import re
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

# Standardize messy or shorthand cofactor nomenclature to PubChem/RDKit-friendly strings
COFACTOR_STANDARDIZATION: dict[str, str] = {
    "nad": "nad+",
    "nadh2": "nadh",
    "nadp": "nadp+",
    "nadph2": "nadph",
    "atp": "atp",
    "adp": "adp",
    "amp": "amp",
    "fad": "fad",
    "fadh2": "fadh2"
}

# Unified regex for catching all generic/non-specific substrates and PPI anomalies.
anomaly_pattern = re.compile(
    r'^(a|an|ferric)\s+'  # Catches generic descriptors like "a sugar", "an alcohol"; "ferric" is a keyword in some entries for Fe3+ ions which we can't use with AlphaFold structures
    r'|protein|peptide|reductase|ferredoxin|cytochrome|globulin|albumin|\benzyme\b|thioredoxin|flavodoxin|factor-1|casein|gelatin|histone|adrenodoxin|putidaredoxin|igfbp-5|kininogen|plasminogen|complement component|factor ix|factor x|factor-l-asparagine|factor-l-proline|insulin'
    r'|cellulose|starch|glycogen|chitin|pectin|dextran|amylose|amylopectin|peptidoglycan|xylan|saccharide|glucuronan|levan|kappa-carrageen|chitosan'
    r'|\b(dna|rna|rnan)\b|trna|mrna|rrna|snrna|oligonucleotide|polynucleotide|\b(purine|pyrimidine|nucleotide|nucleoside)\b'
    r'|lipid|fatty acid|triglyceride|steroid|membrane|diacylglycerol|sphingomyelin'
    r'|\b(acceptor|donor|rx|more|dye|quinone|ion|cation|anion|rooh)\b'
    r'|nad\(p\)|(?:\b|-)ndp(?:\b|-)|(?:\b|-)n(?:\b|-)|acyl-coa|n-acyl|colominic acid'
    r'|\[.*?\]|\([^)]+\)n'
)

# ═══════════════════════════════════════════════════════════════════════════
#  Ligand Filtering
# ═══════════════════════════════════════════════════════════════════════════

def canonicalize_and_filter_ligands(substrate: str, co_substrates: list[str]) -> tuple[str, list[str]] | None:
    """
    Standardize ligands and filter out unparseable generic chemical classes.
    
    Parameters
    ----------
    substrate : str
        The measured substrate name.
    co_substrates : list[str]
        A list of co-substrate names.
        
    Returns
    -------
    tuple[str, list[str]] | None
        The standardized (substrate, co_substrates) tuple if valid, else None.
    """
    def _clean_ligand(ligand: str) -> Optional[str]:
        """
        Cleans a single ligand string by stripping stoichiometry, filtering 
        anomalies, and mapping cofactors.

        Parameters
        ----------
        ligand : str
            The raw ligand name.

        Returns
        -------
        Optional[str]
            Cleaned ligand name, or None if invalid.
        """
        # Step A: Strip stoichiometry (e.g., "1/2 o2", "2 nadh")
        ligand = re.sub(r'^(\d+/\d+|\d+)\s+', '', ligand).strip()
        lig_lower = ligand.lower()
        
        # Step B: Catch all PPIs, anomalies, and generic chemical classes via unified regex
        if anomaly_pattern.search(lig_lower):
            return None
            
        # Step C: Apply Cofactor Mapping
        return COFACTOR_STANDARDIZATION.get(lig_lower, lig_lower)

    # Process measured substrate
    clean_sub = _clean_ligand(substrate)
    if clean_sub is None:
        return None
        
    # Process co-substrates
    clean_cos = []
    for c in co_substrates:
        clean_c = _clean_ligand(c)
        if clean_c is None:
            return None
        clean_cos.append(clean_c)
        
    return clean_sub, clean_cos
