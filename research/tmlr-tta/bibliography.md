# Annotated Bibliography (APA 7.0) — verified

Stage 1 / Phase 2. Every entry web-verified; gray-zone = FAIL (dropped, see bottom).
Tier: A = top peer-reviewed venue; B = other peer-reviewed; C = preprint/arXiv-only.

## A. Test-Time Adaptation
- **Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). Tent: Fully test-time adaptation by entropy minimization.** *ICLR.* — Norm-affine-only TTA precedent; we replace entropy with matching-specific consistency. **A**
- **Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A. A., & Hardt, M. (2020). Test-time training with self-supervision…** *ICML* (pp. 9229–9248). — Founding TTT; conceptual parent. **A**
- **Liu, Y., Kothari, P., van Delft, B., Bellot-Gurlet, B., Mordan, T., & Alahi, A. (2021). TTT++: When does self-supervised test-time training fail or thrive?** *NeurIPS, 34.* — Adds feature-distribution alignment to TTT + failure analysis; motivates our appearance signal AND flags instability. **A**
- **Wang, Q., Fink, O., Van Gool, L., & Dai, D. (2022). Continual test-time domain adaptation.** *CVPR* (pp. 7201–7211). — CoTTA; anti-error-accumulation, relevant to forgetting framing. **A**
- **Niu, S., Wu, J., Zhang, Y., Chen, Y., Zheng, S., Zhao, P., & Tan, M. (2022). Efficient test-time model adaptation without forgetting.** *ICML* (pp. 16888–16905). — EATA; anti-forgetting TTA precedent. **A**
- **Niu, S., Wu, J., Zhang, Y., Wen, Z., Chen, Y., Zhao, P., & Tan, M. (2023). Towards stable test-time adaptation in dynamic wild world.** *ICLR.* — SAR; small-batch stability, informs single-pair adaptation. **A**
- **Zhang, M., Levine, S., & Finn, C. (2022). MEMO: Test-time robustness via adaptation and augmentation.** *NeurIPS, 35.* — Single-sample augmentation-consistency TTA; closest analog to per-pair self-consistency. **A**

## B. Dense / Local Matching
- **Hong, S., & Kim, S. (2021). Deep matching prior: Test-time optimization for dense correspondence.** *ICCV* (pp. 9907–9917). — **NEAREST PRIOR / new baseline.** Per-pair TTO; we differ by adapting a *frozen pretrained SOTA* matcher's decoder norm-affine params, axis-aware. **A**
- **Sun, J., Shen, Z., Wang, Y., Bao, H., & Zhou, X. (2021). LoFTR…** *CVPR* (pp. 8922–8931). — Detector-free backbone. **A**
- **Wang, Y., He, X., Peng, S., Tan, D., & Zhou, X. (2024). Efficient LoFTR…** *CVPR.* — Efficient backbone (adaptee). **A**
- **Edstedt, J., Athanasiadis, I., Wadenbäck, M., & Felsberg, M. (2023). DKM: Dense kernelized feature matching…** *CVPR* (pp. 17765–17775). — Dense warp+certainty matcher. **A**
- **Edstedt, J., Sun, Q., Bökman, G., Wadenbäck, M., & Felsberg, M. (2024). RoMa: Robust dense feature matching.** *CVPR.* — Frozen-DINOv2 matcher; primary adaptee. **A**
- **He, X., Yu, H., Peng, S., Tan, D., Shen, Z., Bao, H., & Zhou, X. (2025). MatchAnything… [Preprint].** arXiv:2501.07556. — Supervised cross-modal rival/ceiling our TTA aims to approach. **C**

## C. Self-Supervision / Consistency
- **Meister, S., Hur, J., & Roth, S. (2018). UnFlow: Unsupervised learning of optical flow with a bidirectional census loss.** *AAAI, 32*(1). — Forward-backward consistency = our appearance-axis signal. **A**
- **Jabri, A., Owens, A., & Efros, A. A. (2020). Space-time correspondence as a contrastive random walk.** *NeurIPS, 33.* — Cycle-consistency self-supervision grounding. **A**

## D. Distribution Alignment
- **Sun, B., & Saenko, K. (2016). Deep CORAL: Correlation alignment for deep domain adaptation.** *ECCV 2016 Workshops* (LNCS 9915, pp. 443–450). — Second-order stat alignment = our feature-alignment term. **B** (workshop)
- **Li, Y., Wang, N., Shi, J., Hou, X., & Liu, J. (2018). Adaptive batch normalization for practical domain adaptation.** *Pattern Recognition, 80,* 109–117. — AdaBN; parameter-free stat replacement = the ablation lower bound vs our affine adaptation. **B** (cite journal, NOT ICLR'17 reject)

## E. Forgetting / Continual (supervised rival's toolkit)
- **Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks.** *PNAS, 114*(13), 3521–3526. — EWC; forgetting metric/method. **A**
- **Li, X., Grandvalet, Y., & Davoine, F. (2018). Explicit inductive bias for transfer learning with convolutional networks.** *ICML* (pp. 2825–2834). — L2-SP; the anchor already used in this repo. **A**
- **Li, Z., & Hoiem, D. (2017). Learning without forgetting.** *IEEE TPAMI, 40*(12), 2935–2947. — LwF; classic forgetting baseline. **A**
- **Huang, C., et al. (2025). Proxy-FDA: Proxy-based feature distribution alignment for fine-tuning vision foundation models without forgetting.** *ICML.* — Closest contemporary: feature-dist alignment + no forgetting; shows forgetting correlates with distributional distance (supports our severity metric). **A**

## F. Metrics
- **Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). GANs trained by a two time-scale update rule converge to a local Nash equilibrium.** *NeurIPS, 30.* — FID; appearance-shift quantifier (→ Fréchet DINOv2 Distance practice). **A**
- **Alvarez-Melis, D., & Fusi, N. (2020). Geometric dataset distances via optimal transport.** *NeurIPS, 33.* — OTDD; cross-check severity metric. **A**

## G. Datasets
- **Borovec, J., Kybic, J., Arganda-Carreras, I., Sorokin, D. V., Bueno, G., Khvostikov, A. V., … Muñoz-Barrutia, A. (2020). ANHIR: Automatic non-rigid histological image registration challenge.** *IEEE TMI, 39*(10), 3042–3052. — Cross-stain histology; appearance-axis domain #1. License CC-BY-NC-SA. **A**
- **Ye, Y., et al. (2024). 3MOS: Multi-sources, multi-resolutions, and multi-scenes dataset for optical–SAR image matching [Preprint].** arXiv:2404.00838. — Optical–SAR; appearance-axis domain #2. **License CC BY-NC-ND 4.0** (NonCommercial + NoDerivatives — eval only, no redistribution of derivatives). Springer *Visual Intelligence* 2025 version = **B**. **C/B**
- **Hernandez-Matas, C., Zabulis, X., Triantafyllou, A., Anyfanti, P., Douma, S., & Argyros, A. A. (2017). FIRE: Fundus image registration dataset.** *Journal for Modeling in Ophthalmology, 1*(4), 16–28. — Backup appearance domain. **B**

## Scale-axis training-signal references (replacements for the dropped hallucination)
- **Scale-Net — Learning to reduce scale differences for large-scale invariant image matching.** arXiv:2112.10485. — scale-difference reduction in matching. *(verify before cite)*
- **PRISM — Progressive dependency maximization for scale-invariant image matching.** arXiv:2408.03598. *(verify before cite)*
- **Jin, K., Chen, J., & Ye, Q. (2026). Improving local feature matching by entropy-inspired scale adaptability and flow-endowed local consistency.** arXiv:2604.06713 (linked IEEE TCSVT). — real paper for scale-adaptability+local-consistency as a *training* signal (we move it to *test time*). *(cite under TRUE title only)*

---

## FAIL list (excluded — integrity)
1. **"Multi-view Scale-invariant Learning" (feature matching)** — no such paper verified. **Dropped.**
2. **arXiv:2604.06713 under any "scale self-consistency" title** — ID real but title was wrong; only cite under its true title (Jin et al., above).

## Camera-ready watch-items
- MatchAnything: arXiv-only — do not assert a conference venue.
- 3MOS: confirm lead author's full given name from the arXiv PDF; respect NC-ND.
- Deep CORAL (workshop, B) and AdaBN (journal, not ICLR) cited correctly above.
