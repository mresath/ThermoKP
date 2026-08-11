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

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════

# Standardize messy or shorthand cofactor nomenclature to PubChem/RDKit-friendly strings
COFACTOR_STANDARDIZATION: dict[str, str] = {
    "nad": "nad+",
    "nad+": "nad+",
    "nadh": "nadh",
    "nadh2": "nadh",
    "nadp": "nadp+",
    "nadp+": "nadp+",
    "nadph": "nadph",
    "nadph2": "nadph",
    "nad(p)": "nad+",
    "nad(p)+": "nad+",
    "nad(p)h": "nadh",
    "ndp": "adp",
    "ntp": "atp",
    "dntp": "atp",
    "dndp": "adp",
    "atp": "atp",
    "gtp": "gtp",
    "ctp": "ctp",
    "utp": "utp",
    "ttp": "ttp",
    "adp": "adp",
    "gdp": "gdp",
    "cdp": "cdp",
    "udp": "udp",
    "tdp": "tdp",
    "amp": "amp",
    "gmp": "gmp",
    "cmp": "cmp",
    "ump": "ump",
    "tmp": "tmp",
    "fad": "fad",
    "fadh2": "fadh2",
    "adpribose": "adp-d-ribose",
    "cdpribose": "cdp-d-ribose",
    "gdpribose": "gdp-d-ribose",
    "udpribose": "udp-d-ribose",
    "tdpribose": "tdp-d-ribose",
    "cyclic-nadph": "nadph",
    "beta-deamino-nadh": "nadh",
    "beta-ngdh": "nadh",
    "dehydro-tmp": "tmp",
    "d-ddcmp": "dcmp",
    "sl-atp": "atp",
    "sl-adp": "adp",
    "r-sh": "methanethiol"
}

# Regex for catching anomalies.
anomaly_pattern = re.compile(
    r'protein|peptide|reductase|ferredoxin|globulin|albumin|\benzyme\b|thioredoxin|flavodoxin|factor-1|casein|gelatin|histone|adrenodoxin|putidaredoxin|igfbp-5|kininogen|plasminogen|complement component|factor ix|factor x|factor-l-asparagine|factor-l-proline|insulin|hemoglobin|actin|ubiquitin|fibrinogen|gastrin|glutaredoxin|amyloid|collagen|myosin|cytochrome|hormone|angiotensinogen|apelin|complex|domain|azurin|rubredoxin|prothrombin|fetuin|cask|samp2|aggrecan|heparin|bckdh' # Full-on proteins
    r'|\b(acceptor|donor|rx|more|dye|ion|cation|anion|rooh)\b' # Unparseable generics
    r'|\[.*?\]' # Unparseable annotations
)

# Map generic polymer names to specific trimer or simple proxy names
# ═══════════════════════════════════════════════════════════════════════════
#  Ligand Filtering
# ═══════════════════════════════════════════════════════════════════════════

def canonicalize_and_filter_ligands(substrate: str, co_substrates: list[str]) -> tuple[str, list[str]] | None:
    """
    Standardize ligands and filter out unparseable generic chemical classes,
    and apply dynamic proxy rules (e.g. polymers -> trimers).
    
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
    def _clean_ligand(ligand: str) -> list[str] | None:
        """
        Cleans a single ligand string by stripping stoichiometry, filtering 
        anomalies, and mapping cofactors and proxies.

        Returns
        -------
        list[str] | None
            A list of cleaned ligand names (since one input like "ferric X" 
            might decompose into ["X", "Fe3+"]), or None if invalid.
        """
        # Strip stoichiometry (e.g., "1/2 o2", "2 nadh", "2n h+", "n h2o")
        ligand = re.sub(r'^(\d+/\d+|\d+[nN]?|[nN])\s+', '', ligand).strip()
        lig_lower = ligand.lower()
        
        # Strip "reduced" prefix
        lig_lower = re.sub(r'^reduced\s+', '', lig_lower)
        
        # Dynamically map cofactors inside complex names (e.g. ndp-glucose -> adp-glucose)
        # Build dynamic regex using sorted keys (longest first to prevent nad matching before nadh2)
        keys = sorted(COFACTOR_STANDARDIZATION.keys(), key=len, reverse=True)
        escaped_keys = [re.escape(k) for k in keys]
        cofactor_pattern = re.compile(rf"(?<![a-zA-Z0-9_])({'|'.join(escaped_keys)})(?![a-zA-Z0-9_])", re.IGNORECASE)
        
        lig_lower = cofactor_pattern.sub(lambda m: COFACTOR_STANDARDIZATION.get(m.group(0).lower(), m.group(0)), lig_lower)
        
        results = []
        
        # Handle "ferric X" -> X + Fe3+
        if lig_lower.startswith("ferric "):
            lig_lower = lig_lower[7:].strip()
            if not lig_lower:
                lig_lower = "fe3+"
            else:
                results.append("fe3+")
                
        # Catch all PPIs, anomalies, and remaining undesired generic chemical classes via unified regex
        if anomaly_pattern.search(lig_lower):
            return None
            
        # Apply Cofactor Mapping
        mapped = COFACTOR_STANDARDIZATION.get(lig_lower, lig_lower)
        results.insert(0, mapped)
        
        return results

    # Process measured substrate
    clean_subs = _clean_ligand(substrate)
    if clean_subs is None or not clean_subs:
        return None
        
    primary_sub = clean_subs[0]
    extra_co = clean_subs[1:]
        
    # Process co-substrates
    clean_cos = []
    for c in co_substrates:
        clean_c_list = _clean_ligand(c)
        if clean_c_list is None:
            return None
        clean_cos.extend(clean_c_list)
        
    return primary_sub, extra_co + clean_cos
