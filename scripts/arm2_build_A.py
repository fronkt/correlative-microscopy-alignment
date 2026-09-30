"""Component A: DefDAP HR-DIC vs EBSD pairs (Zenodo CC BY 4.0).

Pair = EBSD band-contrast image  vs  DIC max-shear strain map (grid resolution, cropped as in the authors'
notebook so the notebook's homologous points apply 1:1; no resampling of the strain map). This mirrors
AmalgaMatch's SlipPartitioning task (EBSD IQ/CI/PRIAS vs SE / exx strain maps).
Homologous points are the authors' hand-picked lists copied from their notebooks (source of GT).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from skimage import io as skio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm2_common import BENCH, finalize_pair, sha256, to_u8, to_u16  # noqa: E402
from arm2_defdap_io import max_shear, read_crc, read_dic  # noqa: E402

RAW = BENCH / "raw"
LIC = "CC BY 4.0"


def crop(a, c):
    l, r, t, b = c
    return a[t:a.shape[0] - b, l:a.shape[1] - r]


SPECS = [
    dict(id="defdap_8383311_Zr4hydride", rec="8383311", cluster="8383311:Zr4-hydride", sub="Zr4-hydrides",
         dic=RAW / "8383311/SetA_003.TXT", ebsd=RAW / "8383311/EBSD", crop=(100, 100, 100, 100), rot=0,
         um_px=0.0048828, gt_model="piecewiseAffine (authors)", step_note="SetA_003 (3rd/last step; notebook plots DicMaps[2])",
         ebsd_pts=[(133, 186), (897, 655), (710, 228), (178, 638), (233, 425), (1031, 155), (807, 209), (778, 290), (711, 430), (346, 734),
                   (757, 745), (111, 94), (106, 88), (111, 747), (76, 766), (609, 739), (1017, 759), (1046, 122), (1054, 115), (319, 95),
                   (1020, 790), (490, 194), (514, 524)],
         dic_pts=[(39, 151), (1061, 848), (796, 188), (127, 851), (181, 520), (1200, 60), (916, 160), (878, 284), (804, 507), (353, 991),
                  (885, 985), (5, 6), (0, 0), (45, 1009), (0, 1053), (693, 986), (1225, 996), (1216, 7), (1229, 0), (273, 2),
                  (1229, 1053), (501, 148), (555, 653)],
         url="https://zenodo.org/records/8383311", member="Archive.zip: data/{SetA_003.TXT,EBSD.crc,EBSD.cpr,Notebook.ipynb}"),
    dict(id="defdap_21218524_Zr4_TD_irr", rec="21218524", cluster="21218524:Zr4-TD-irr", sub="Zr4-irradiated-TD-irr",
         dic=RAW / "21218524/irr_003.TXT", ebsd=RAW / "21218524/TD/Zr4_T_irr", crop=(35, 37, 14, 16), rot=2,
         um_px=10 / 2048, gt_model="default DefDAP link (not set in notebook)", step_note="irr_003 (last of 3 steps)",
         ebsd_pts=[(700, 776), (665, 848), (154, 856), (761, 530), (856, 837), (937, 865), (851, 352), (186, 661), (255, 366), (491, 392), (504, 493)],
         dic_pts=[(1542, 1581), (1448, 1809), (68, 1790), (1714, 814), (1962, 1790), (2175, 1890), (1965, 267), (164, 1179), (358, 265), (990, 358), (1030, 680)],
         url="https://zenodo.org/records/21218524", member="Zr4_0.1dpa_R_T.zip: TD/irr_003.TXT, TD/Zr4_T_irr.{crc,cpr}, Notebook.ipynb"),
    dict(id="defdap_21218524_Zr4_TD_nirr", rec="21218524", cluster="21218524:Zr4-TD-nirr", sub="Zr4-irradiated-TD-nirr",
         dic=RAW / "21218524/nirr_003.TXT", ebsd=RAW / "21218524/TD/Zr4_T_nirr", crop=(35, 37, 14, 16), rot=2,
         um_px=10 / 2048, gt_model="default DefDAP link (not set in notebook)", step_note="nirr_003 (last of 3 steps)",
         ebsd_pts=[(336, 619), (525, 449), (295, 158), (982, 424), (275, 712), (432, 478), (885, 682), (784, 180)],
         dic_pts=[(474, 1509), (988, 988), (381, 50), (2208, 946), (308, 1793), (734, 1068), (1937, 1754), (1682, 148)],
         url="https://zenodo.org/records/21218524", member="Zr4_0.1dpa_R_T.zip: TD/nirr_003.TXT, TD/Zr4_T_nirr.{crc,cpr}, Notebook.ipynb"),
    dict(id="defdap_21218524_Zr4_RD_irr", rec="21218524", cluster="21218524:Zr4-RD-irr", sub="Zr4-irradiated-RD-irr",
         dic=RAW / "21218524/irr_002.TXT", ebsd=RAW / "21218524/RD/Zr4_R_irr", crop=(37, 36, 21, 22), rot=0,
         um_px=10 / 2048, gt_model="default DefDAP link (not set in notebook)", step_note="irr_002 (last of 2 steps)",
         ebsd_pts=[(382, 810), (377, 865), (206, 769), (169, 901), (388, 393), (218, 323), (917, 461), (586, 281),
                   (593, 385), (932, 315), (265, 721), (401, 576), (462, 597), (568, 687), (906, 689), (685, 903), (938, 885)],
         dic_pts=[(659, 1563), (643, 1723), (187, 1447), (91, 1829), (681, 357), (229, 151), (2091, 559), (1213, 29),
                  (1227, 339), (2138, 137), (343, 1302), (712, 885), (875, 944), (1157, 1207), (2058, 1217), (1458, 1835), (2137, 1783)],
         url="https://zenodo.org/records/21218524", member="Zr4_0.1dpa_R_T.zip: RD/irr_002.TXT, RD/Zr4_R_irr.{crc,cpr}, Notebook.ipynb"),
    dict(id="defdap_21218524_Zr4_RD_nirr", rec="21218524", cluster="21218524:Zr4-RD-nirr", sub="Zr4-irradiated-RD-nirr",
         dic=RAW / "21218524/nirr_002.TXT", ebsd=RAW / "21218524/RD/Zr4_R_nirr", crop=(40, 35, 70, 5), rot=0,
         um_px=10 / 2048, gt_model="default DefDAP link (not set in notebook)", step_note="nirr_002 (last of 2 steps)",
         ebsd_pts=[(951, 779), (769, 881), (727, 948), (229, 862), (470, 841), (773, 592), (702, 375), (373, 676),
                   (342, 742), (553, 556), (479, 497), (385, 418), (272, 404), (164, 651), (934, 611), (881, 515)],
         dic_pts=[(2196, 1304), (1708, 1603), (1594, 1800), (269, 1560), (912, 1492), (1726, 760), (1540, 120), (654, 1008),
                  (574, 1202), (1138, 655), (940, 479), (698, 244), (396, 203), (104, 932), (2157, 815), (2014, 531)],
         url="https://zenodo.org/records/21218524", member="Zr4_0.1dpa_R_T.zip: RD/nirr_002.TXT, RD/Zr4_R_nirr.{crc,cpr}, Notebook.ipynb"),
    dict(id="defdap_14245480_Ti64_pore", rec="14245480", cluster="14245480:Ti64-pore", sub="SLM-Ti64-pore",
         dic=RAW / "14245480/Pore/B00002.txt", ebsd=RAW / "14245480/Pore/ebsd", crop=(140, 150, 90, 115), rot=0,
         um_px=30 / 2048, gt_model="affine (authors)", step_note="B00002 (dicMaps[1], plotted in notebook)",
         ebsd_pts=[(74, 71), (159, 60), (204, 31), (238, 41), (81, 176), (50, 147), (136, 220), (293, 235)],
         dic_pts=[(108, 128), (327, 65), (456, 17), (538, 39), (144, 483), (59, 380), (276, 553), (674, 577)],
         url="https://zenodo.org/records/14245480", member="Defect DefDAP.zip: SLM_Ti64/Pore/{B00002.txt,ebsd.crc,ebsd.cpr}, Ti64_defect paper.ipynb"),
    dict(id="defdap_14245480_Ti64_bulk", rec="14245480", cluster="14245480:Ti64-bulk", sub="SLM-Ti64-bulk",
         dic=RAW / "14245480/Bulk/B00035.txt", ebsd=RAW / "14245480/Bulk/ebsd", crop=(210, 350, 230, 200), rot=0,
         um_px=30 / 2048, gt_model="piecewiseAffine (authors)", step_note="B00035 (dicMapsBulk[1], plotted in notebook)",
         # the notebook lists 10 points; the last 4 are extrapolated corner anchors (one EBSD x = -20, outside the map): excluded
         ebsd_pts=[(742, 647), (154, 697), (421, 1394), (525, 51), (1476, 310), (1691, 971)],
         dic_pts=[(862, 729), (170, 782), (477, 1616), (617, 6), (1738, 336), (1974, 1068)],
         url="https://zenodo.org/records/14245480", member="Defect DefDAP.zip: SLM_Ti64/Bulk/{B00035.txt,ebsd.crc,ebsd.cpr}, Ti64_defect paper.ipynb"),
    dict(id="defdap_16633511_Ti_CS1", rec="16633511", cluster="16633511:Ti-CS1", sub="Ti-alloy-CaseStudy1",
         dic=RAW / "16633511/B00012.txt", ebsd=RAW / "16633511/Sample 2", crop=(100, 100, 100, 100), rot=0,
         um_px=30 / 2048, gt_model="affine (authors)", step_note="B00012 (last of 12 steps)",
         ebsd_pts=[(238, 120), (324, 102), (375, 147), (440, 230), (400, 344), (429, 440), (310, 371),
                   (166, 399), (207, 334), (216, 295), (155, 308), (102, 282), (104, 206), (306, 288)],
         dic_pts=[(806, 239), (1198, 147), (1445, 376), (1769, 790), (1622, 1341), (1782, 1814), (1215, 1473),
                  (563, 1617), (728, 1295), (758, 1102), (483, 1161), (231, 1032), (212, 659), (1169, 1066)],
         url="https://zenodo.org/records/16633511", member="Case study 1 ... .zip: DIC_DaVis_Export/B00012.txt, EBSD/Sample 2.{crc,cpr}, DefDAP_analysis_notebook.ipynb"),
    dict(id="defdap_13755208_Ni_superalloy", rec="13755208", cluster="13755208:Ni-superalloy", sub="Ni-superalloy",
         dic=RAW / "13755208/B00011.txt", ebsd=RAW / "13755208/EBSD data", crop=(50, 100, 20, 80), rot=-1,
         um_px=50 / 4096, gt_model="affine (authors)", step_note="Integrated/B00011 (Step=11 in notebook)",
         mask=RAW / "13755208/mask.tif",
         ebsd_pts=[(623, 47), (1081, 877), (1342, 1418), (733, 1279), (1063, 221), (1370, 600), (612, 392), (698, 565), (778, 1495)],
         dic_pts=[(40, 77), (742, 1202), (1152, 1942), (273, 1778), (685, 282), (1161, 791), (47, 560), (177, 784), (356, 2077)],
         url="https://zenodo.org/records/13755208", member="HRDIC and EBSD data.zip: Displacement data/Integrated/B00011.txt, EBSD data/EBSD data.{crc,cpr}, Plot the strain maps (with EBSD data).ipynb"),
]


SPECS += [
    dict(id="defdap_14532401_HEA_CG", rec="14532401", cluster="14532401:CoCrFeNi-CG", sub="CoCrFeNi-HEA-CG",
         dic=RAW / "14532401/Fig.3(E)CG-HEA_2.25%_HRDIC.txt", ebsd=RAW / "14532401/Fig.3CG-HEA_HRDIC", crop=(263, 1470, 1137, 554), rot=0,
         um_px=30 / 2048, gt_model="affine (authors; projective/poly2 also tried)",
         step_note="2.25% file (notebook B00003 assumed = 3rd step; points are step-independent)",
         ebsd_pts=[(250.3, 428.1), (449.6, 507.5), (332.3, 362.1), (272, 516.9), (325.2, 508.4)],
         dic_pts=[(114, 218), (719, 456), (354, 29), (186, 496), (339, 473)],
         url="https://zenodo.org/records/14532401", member="Fig.3(E)CG-HEA_2.25%_HRDIC.txt, Fig.3CG-HEA_HRDIC.{crc,cpr}, Fig.3Notebook-checkpoint.ipynb (HEA2 block)"),
    dict(id="defdap_14532401_HEA_FG", rec="14532401", cluster="14532401:CoCrFeNi-FG", sub="CoCrFeNi-HEA-FG",
         dic=RAW / "14532401/Fig.3(F)FG-HEA_2.25%_HRDIC.txt", ebsd=RAW / "14532401/Fig.3FG-HEA_HRDIC", crop=(1322, 535, 516, 1306), rot=0,
         um_px=30 / 2048, gt_model="affine (authors; projective/poly2 also tried)",
         step_note="2.25% file (notebook B00003 assumed = 3rd step; points are step-independent)",
         ebsd_pts=[(1322.9, 242.3), (1330.9, 421.9), (1280.3, 573.6), (1369.2, 717.9), (1571.9, 313.6), (1814.8, 346.7)],
         dic_pts=[(69.7, 37.9), (79.6, 258.7), (25.7, 448.0), (130.9, 618.5), (362.1, 130.5), (644.4, 167.3)],
         url="https://zenodo.org/records/14532401", member="Fig.3(F)FG-HEA_2.25%_HRDIC.txt, Fig.3FG-HEA_HRDIC.{crc,cpr}, Fig.3Notebook-checkpoint.ipynb (HEA3 block)"),
]


def build(spec: dict, force_rot=None) -> dict:
    dic = read_dic(spec["dic"])
    ms = max_shear(dic["u"], dic["v"], dic["step_px"])
    if spec.get("mask"):
        m = skio.imread(spec["mask"])
        if m.shape == ms.shape:
            ms = ms.copy()
            ms[m < 115] = 0
        else:
            print("  mask shape", m.shape, "!= grid", ms.shape, "-> mask NOT applied")
    ms_c = crop(ms, spec["crop"])
    e = read_crc(spec["ebsd"])
    bc = e["bc"]
    rot = spec["rot"]
    if rot == 2:
        bc = np.rot90(bc, 2)
    elif rot == -1:
        bc = np.rot90(bc, axes=(1, 0))
    bc = np.ascontiguousarray(bc)
    dp = np.array(spec["dic_pts"], float)
    ep = np.array(spec["ebsd_pts"], float)
    info = dict(dic_grid=(dic["ny"], dic["nx"]), step_px=dic["step_px"], cropped=ms_c.shape, ebsd=bc.shape)
    # bounds check: notebook points must lie inside the cropped DIC map / (rotated) EBSD map
    inside_d = ((dp[:, 0] >= 0) & (dp[:, 0] <= ms_c.shape[1]) & (dp[:, 1] >= 0) & (dp[:, 1] <= ms_c.shape[0])).all()
    inside_e = ((ep[:, 0] >= 0) & (ep[:, 0] <= bc.shape[1]) & (ep[:, 1] >= 0) & (ep[:, 1] <= bc.shape[0])).all()
    info.update(inside_dic=bool(inside_d), inside_ebsd=bool(inside_e))
    print(spec["id"], info)
    vmax = float(np.percentile(ms_c[np.isfinite(ms_c)], 99.5))
    strain_u16 = to_u16(ms_c, vmax)
    bc_u8 = to_u8(bc)
    px_dic_um = spec["um_px"] * dic["step_px"]
    px_ebsd_um = e["_meta"]["step_um"]
    fov_dic = px_dic_um * ms_c.shape[1]
    fov_ebsd = px_ebsd_um * bc.shape[1]
    raw_files = {p.name: sha256(p) for p in [spec["dic"], Path(str(spec["ebsd"]) + ".crc"), Path(str(spec["ebsd"]) + ".cpr")]}
    rec = finalize_pair(
        pair_id=f"{spec['id']}#0", component="A", cluster=spec["cluster"], group="SlipPartitioning", subclass=spec["sub"],
        img_a=bc_u8, img_b=strain_u16, xy_a=ep, xy_b=dp, name_a="EBSD band contrast", name_b="DIC max shear strain",
        px_a_nm=px_ebsd_um * 1e3, px_b_nm=px_dic_um * 1e3, mod_a="EBSD band contrast (8-bit, 1-99 pct stretch)",
        mod_b=f"HR-DIC max shear strain (16-bit, clip at p99.5={vmax:.4f})", licence=LIC,
        source_url=spec["url"], files=dict(archive_members=spec["member"], raw_sha256=raw_files, step=spec["step_note"]),
        notes=f"GT model in authors' notebook: {spec['gt_model']}; crop(l,r,t,b)={spec['crop']}; ebsd_rot={spec['rot']}; {info}",
        fov_a=fov_ebsd, fov_b=fov_dic)
    rec["_info"] = info
    return rec


if __name__ == "__main__":
    only = sys.argv[1:]
    for s in SPECS:
        if only and s["id"] not in only:
            continue
        needed = [s["dic"], Path(str(s["ebsd"]) + ".crc"), Path(str(s["ebsd"]) + ".cpr")]
        miss = [p for p in needed if not p.exists()]
        if miss:
            print("SKIP (missing)", s["id"], [p.name for p in miss])
            continue
        build(s)
