"""Component A downloads: only the members needed to build each EBSD-vs-strain-map pair (range reads).
Sizes are logged to results/arm2/download_sizes.txt."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm2_fetch import BENCH, fetch_members, fetch_url, zenodo_url  # noqa: E402

RAW = BENCH / "raw"

JOBS = [
    # (record, zip, members, outdir)
    ("21218524", "Zr4_0.1dpa_R_T.zip", [
        "Zr4_0.1dpa_R_T/TD/irr_003.TXT", "Zr4_0.1dpa_R_T/TD/nirr_003.TXT",
        "Zr4_0.1dpa_R_T/RD/irr_002.TXT", "Zr4_0.1dpa_R_T/RD/nirr_002.TXT",
        "Zr4_0.1dpa_R_T/TD/i_8scale.bmp", "Zr4_0.1dpa_R_T/TD/ni_8scale.bmp",
        "Zr4_0.1dpa_R_T/RD/i_scale8.bmp", "Zr4_0.1dpa_R_T/RD/ni_scale8.bmp",
    ], "21218524"),
    ("21218524", "Zr4_0.1dpa_R_T.zip", ["Zr4_0.1dpa_R_T/TD/Zr4_T_irr.crc", "Zr4_0.1dpa_R_T/TD/Zr4_T_irr.cpr",
                                        "Zr4_0.1dpa_R_T/TD/Zr4_T_nirr.crc", "Zr4_0.1dpa_R_T/TD/Zr4_T_nirr.cpr"], "21218524/TD"),
    ("21218524", "Zr4_0.1dpa_R_T.zip", ["Zr4_0.1dpa_R_T/RD/Zr4_R_irr.crc", "Zr4_0.1dpa_R_T/RD/Zr4_R_irr.cpr",
                                        "Zr4_0.1dpa_R_T/RD/Zr4_R_nirr.crc", "Zr4_0.1dpa_R_T/RD/Zr4_R_nirr.cpr"], "21218524/RD"),
    ("14245480", "Defect DefDAP.zip", [
        "Defect DefDAP/SLM_Ti64/Pore/B00002.txt", "Defect DefDAP/SLM_Ti64/Pore/ebsd.crc",
        "Defect DefDAP/SLM_Ti64/Pore/ebsd.cpr", "Defect DefDAP/SLM_Ti64/Pore/pattern.png"], "14245480/Pore"),
    ("14245480", "Defect DefDAP.zip", [
        "Defect DefDAP/SLM_Ti64/Bulk/B00035.txt", "Defect DefDAP/SLM_Ti64/Bulk/ebsd.crc",
        "Defect DefDAP/SLM_Ti64/Bulk/ebsd.cpr"], "14245480/Bulk"),
    ("16633511", "Case study 1 Slip activation in a titanium alloy during elastic-plastic transition.zip", [
        "Case study 1 Slip activation in a titanium alloy during elastic-plastic transition/DIC_DaVis_Export/B00012.txt",
        "Case study 1 Slip activation in a titanium alloy during elastic-plastic transition/DIC_DaVis_Export/pattern.png",
        "Case study 1 Slip activation in a titanium alloy during elastic-plastic transition/EBSD/Sample 2.crc",
        "Case study 1 Slip activation in a titanium alloy during elastic-plastic transition/EBSD/Sample 2.cpr"], "16633511"),
    ("13755208", "HRDIC and EBSD data.zip", [
        "HRDIC and EBSD data with Jupyter Notebook/Displacement data/Integrated/B00011.txt",
        "HRDIC and EBSD data with Jupyter Notebook/Displacement data/Mask/mask.tif",
        "HRDIC and EBSD data with Jupyter Notebook/EBSD data/EBSD data.crc",
        "HRDIC and EBSD data with Jupyter Notebook/EBSD data/EBSD data.cpr"], "13755208"),
]
ZEN_FILES = [  # (record, key) direct Zenodo files
    ("14532401", "Fig.3(E)CG-HEA_2.25%_HRDIC.txt"), ("14532401", "Fig.3(F)FG-HEA_2.25%_HRDIC.txt"),
    ("14532401", "Fig.3CG-HEA_HRDIC.crc"), ("14532401", "Fig.3CG-HEA_HRDIC.cpr"),
    ("14532401", "Fig.3FG-HEA_HRDIC.crc"), ("14532401", "Fig.3FG-HEA_HRDIC.cpr"),
]


def main() -> None:
    for rec, zipname, members, out in JOBS:
        fetch_members(rec, zipname, members, RAW / out)
    for rec, key in ZEN_FILES:
        fetch_url(zenodo_url(rec, key), RAW / rec / key)
    print("DONE A")


if __name__ == "__main__":
    main()
