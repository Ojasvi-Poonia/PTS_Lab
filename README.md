# PTS_Lab

sudo arp-scan -I eth0 192.168.20.0/24

nmap -sn 192.168.20.0/24

nmap -sV -p- 192.168.20.47

notion - https://www.notion.so/36063af63ccd80a5b48fedd16d36c334?source=copy_link
docs - https://docs.google.com/document/d/1kzw0iMyQz-id1cuwZu4RQJuLY4Qt0sAOwHN8UK2RC6k/edit?tab=t.0

drive - https://drive.google.com/drive/folders/1B5Vw9tpbZ2i18H_5ln_M47jHsdQo_aXv

main main docs - https://docs.google.com/document/d/161vCeIs2hqVdIj8Vo-BdSYyGXnPljSYKLJx28sf42ek/edit?usp=sharing
main docs - https://docs.google.com/document/d/1m-GSSwZD-Bq6ne9kVTj_ku2I2HtdxSHWveuw0KXh-hU/edit?usp=sharing

web - https://dontpad.com/ptslabhack

Burnercat/pentesting

# CLINE PROMPT — Test-Time Registers for Open-Vocabulary Detection (OWLv2)

Copy everything below this line into Cline as the task prompt.

---

## ROLE

You are implementing a research experiment end-to-end. Work in phases, stop at each checkpoint, and report before continuing. The experiment tests whether **test-time registers** — a training-free feature-space intervention for Vision Transformers — improve a frozen OWLv2 open-vocabulary detector.

## RESEARCH CONTEXT

- ViTs pretrained like CLIP/DINOv2 develop **high-norm outlier tokens**: a sparse set of "register neurons" in MLP layers dumps large activations onto a few (usually background) patch tokens. Those patches lose local information and pollute attention (Darcet et al. 2024).
- Jiang et al. (NeurIPS 2025, arXiv 2506.08010, code: https://github.com/nickjiang2378/test-time-registers) showed a training-free fix: append an extra zero-initialized token, and at each identified register neuron **copy the maximum patch activation into that extra token and zero it at all patch positions**, then continue the forward pass. Patch tokens come out clean; downstream tasks improve. Demonstrated on CLIP and DINOv2 — never on a dense per-patch detector.
- OWLv2 (`google/owlv2-base-patch16-ensemble`) is a CLIP-lineage ViT-B/16 whose detection head reads **every patch token** (class embedding, box, objectness, and a learned per-patch shift b_p and scale w_p per patch). If artifact patches exist, they sit directly in the detection readout — plausibly causing spurious boxes or corrupted per-patch features.
- Open question this experiment answers first: **does detection fine-tuning suppress these artifacts?** Nobody has measured this. Either outcome is a result:
  - Artifacts absent/neutralized → finding: "dense detection training removes register artifacts" (+ mechanism via b_p/w_p, below).
  - Artifacts present → apply the intervention and measure detection gains.

## HYPOTHESES (pre-registered — report outcomes positive or negative)

- **H0 (existence):** OWLv2's vision tower exhibits high-norm outlier patch tokens (bimodal norm distribution in mid/late layers, as in Darcet et al.), on ≥ a nontrivial fraction of images.
- **H1 (harm):** false-positive detections co-locate with outlier patches more than chance (spatial correlation between outlier maps and unmatched high-score detections).
- **H2 (accuracy):** with n test-time registers, paired ΔmAP ≥ +0.3 (COCO subset) or ΔAP_r ≥ +1.0 (LVIS subset), Wilcoxon-significant.
- **H3 (reliability):** selective-prediction pooled AUROC (correct-vs-wrong ranking by raw score) improves by ≥ +1.0 point — the same 1-point gate used in the prior OWLv2 study.
- **H4 (mechanism):** outlier patches carry extreme learned per-patch shift/scale (b_p, w_p) — i.e., OWLv2 may have *learned* to mute its own artifacts at the score level. Test by correlating patch outlier-ness with b_p, w_p, and objectness.

Note the built-in logic: if H0 fails, H2/H3 are moot and the paper becomes the H0+H4 measurement. If H0 holds but H2 fails while H4 holds, that is a coherent mechanistic story (artifacts exist but the learned calibration already neutralizes them — "miscalibrated but well-ranked" extended to features). Report whichever branch the data takes.

## HONESTY RULES (non-negotiable)

- Never fabricate or fill in a number; every reported value traces to a run artifact on disk.
- Report failures (OOM, hook errors, download issues) and the fallback taken; never silently skip.
- 5-image smoke test before every full run. All comparisons paired on identical images, identical post-processing, seed 42 everywhere.
- Neuron identification uses **held-out images only** — never the evaluation subsets.

## HARDWARE & ENVIRONMENT

- Target: Apple M1, 8 GB RAM, PyTorch MPS, fp32 (fp16 on MPS is flaky). Batch 1, sequential, cache aggressively. Must also run with `--device cuda` unchanged.
- Expected encode cost ~1 s/image on M1; each full config re-encode of 200 images ≈ 4–5 min. This is affordable — do not over-engineer.
- Python 3.10+, venv. `requirements.txt` (pin after verify): `torch`, `torchvision`, `transformers>=4.44`, `pycocotools`, `lvis`, `numpy`, `scipy`, `pandas`, `matplotlib`, `Pillow`, `tqdm`, `requests`.
- Clone https://github.com/nickjiang2378/test-time-registers into `third_party/` and **port** its neuron-identification and intervention logic to the HF OWLv2 vision tower; do not reinvent the algorithm. Read their README/algorithm first and summarize it in `notes/ttr_algorithm.md` before writing code.

## MODEL & FORWARD-PASS ANATOMY (get this right)

- `Owlv2ForObjectDetection` from HF transformers, checkpoint `google/owlv2-base-patch16-ensemble`. 960×960 input → 60×60 = 3,600 patch tokens + 1 CLS.
- The detection head (`image_embedder` path): takes vision-tower `last_hidden_state`, merges the CLS token into patch tokens (broadcast multiply), layernorm, reshapes patches into the 60×60 grid, then class head (per-patch class embedding + learned per-patch shift/scale used as logit shift/scale), box head, objectness head. **Any extra appended token must be sliced off before this merge/reshape or the grid breaks.** Inspect the installed transformers source for the exact tensor flow before patching; record the file/line references in `notes/owlv2_anatomy.md`.
- Extraction hooks needed:
  1. Per-layer patch-token norms (all encoder layers) — for the H0 measurement.
  2. MLP hidden activations (post-activation, per neuron) — for register-neuron identification.
  3. Final per-patch quantities: class embedding, b_p, w_p, objectness, boxes — for eval and H4.

## INTERVENTION IMPLEMENTATION (`src/ttr_owlv2.py`)

- **Register tokens:** append n zero-initialized tokens (n ∈ {1, 2, 4}) to the sequence AFTER positional embeddings are added (registers get no positional embedding), so sequence = [CLS, 3600 patches, n registers]. Full attention, no mask changes.
- **Register-neuron edit (ported from the official repo):** at each identified register neuron, during the forward pass, copy the max activation across patch positions into the register token slot(s) and zero the activation at all patch positions. Implement with forward hooks or a patched MLP forward — whichever matches the reference implementation most faithfully.
- **Slicing:** drop the n register tokens from `last_hidden_state` before the detection head consumes it.
- **Configs to support:** `baseline` (no intervention — must be bit-identical to stock HF forward; assert this), `ttr_n{1,2,4}`, and `neuron_zero` ablation (zero the register neurons WITHOUT providing a register token — the reference paper says outlier mass must land in some token; this ablation tests that in detection).
- **Fidelity checks (gate before any eval):** on 10 held-out images, (a) patch-token outlier norms reduced ≥ 90% under ttr configs, (b) no NaNs/Infs, (c) baseline config reproduces stock HF outputs exactly, (d) runtime overhead per image measured and reported.
- Optional flag `--prior_cache path`: if a prior cached-encoding dump from the earlier OWLv2 study is supplied, verify the baseline re-encode matches it within fp16 rounding (~1e-4) and report the max deviation.

## DATA

- **COCO:** 200-image val2017 subset, seed 42 (mirrors the prior OWLv2 study; support `--image_ids path.json` to inject the exact earlier id list). Download only subset images via `coco_url`. Splits: 50-image fast, 5-image smoke.
- **Held-out neuron-ID set:** 50 val2017 images, seed 43, **disjoint from the 200 eval ids** (assert disjointness in code).
- **LVIS:** 400-image rare-stratified subset, seed 42 (≥ 250 images with ≥ 1 rare category), `--image_ids` supported; images via each record's `coco_url` (LVIS val draws from COCO train2017 AND val2017 — do not assume val2017 only). LVIS class prompt names: `_`→space, strip parentheticals, resolve collisions by keeping parentheses; persist and unit-test the 1,203-entry map.
- Text queries: encode all class names once per dataset with OWLv2's text tower, template "a photo of a {name}"; cache.

## EVALUATION PROTOCOL (match the prior study exactly)

- **COCO:** per-class top-30 selection, pycocotools mAP (AP@[.50:.95]) on the 200 subset.
- **LVIS:** global top-300/image, official federated AP; report AP, AP_r, AP_c, AP_f.
- **Selective prediction (H3):** per image emit top-100 detections by raw score; label correct/wrong by greedy IoU ≥ 0.5 matching to GT; pooled AUROC of raw score separating correct from wrong. Compute per config.
- Paired per-image deltas + Wilcoxon signed-rank for every config vs baseline. Subset absolute numbers are not literature-comparable; paired deltas are the result.

## ANALYSES

1. **H0 — artifact census (`src/measure_artifacts.py`):** per-layer patch-norm distributions over the eval subset; outlier definition following Darcet et al. (bimodality-based cutoff — implement their criterion from the reference paper/repo, document the exact rule used); fraction of outlier tokens per image, per layer; spatial heatmaps for 10 example images saved to `results/viz/`.
2. **H1 — harm localization:** from the baseline run, take unmatched (wrong) detections with score above the median; compute overlap between their box centers/patches and the outlier map; compare against a permutation-shuffled null (1,000 shuffles, seed 42) for a p-value.
3. **H4 — learned self-calibration:** scatter/correlate per-patch outlier-norm vs b_p, w_p, objectness on the eval subset; report Spearman correlations. If outliers carry extreme negative shift or tiny scale, OWLv2 learned to mute them — state this explicitly in the report and connect it to whether H2 succeeded or failed.
4. **Main table:** rows = {baseline, ttr_n1, ttr_n2, ttr_n4, neuron_zero}; columns = {COCO mAP, ΔmAP, COCO AUROC, ΔAUROC, LVIS AP, AP_r, ΔAP_r, LVIS AUROC, Δ}; plus per-image encode time overhead.
5. **Qualitative:** side-by-side attention/norm maps and detections (baseline vs ttr) for 10 images including cases where predictions changed.

## PHASES & CHECKPOINTS (stop and report after each)

- **Phase 0 — Setup:** venv, requirements, model loads, one stock forward; clone reference repo; write `notes/ttr_algorithm.md` and `notes/owlv2_anatomy.md`. Deliver `env_report.txt`.
- **Phase 1 — Data:** COCO subset + held-out set (assert disjoint), text-query cache. LVIS deferred to Phase 6.
- **Phase 2 — H0 measurement:** artifact census on the 200-image COCO subset. **Decision gate:** if outliers are essentially absent (report the numbers against the documented cutoff), stop intervention work, run H4 anyway, and write the measurement-paper report (`REPORT_no_artifacts.md`). Otherwise continue.
- **Phase 3 — Neuron ID:** port identification from the reference repo, run on the 50 held-out images; report neuron count, sparsity, layer distribution; before/after norm maps on held-out images.
- **Phase 4 — Intervention + fidelity:** implement configs, pass all fidelity checks on held-out images. Do not proceed until baseline bit-match and ≥90% outlier-norm reduction are demonstrated.
- **Phase 5 — COCO eval:** re-encode 200 images per config (report wall time), run mAP + AUROC + H1 + H4 analyses, paired stats. Evaluate H2/H3 on COCO.
- **Phase 6 — LVIS eval:** build LVIS subset, re-encode per config, federated AP + AUROC, rare-class focus. (Run regardless of the COCO H2 outcome — rare classes and 1,203-way scoring may respond differently.)
- **Phase 7 — Report:** `results/REPORT.md` with every hypothesis marked SUPPORTED / NOT SUPPORTED and exact numbers; include the branch logic (which story the data chose) and the main table, plots, and viz.

## REPO LAYOUT

```
ttr-owlv2/
  README.md  requirements.txt  env_report.txt
  notes/   (ttr_algorithm.md, owlv2_anatomy.md)
  third_party/test-time-registers/
  configs/default.yaml           # n registers, neuron set path, device, subsets, thresholds
  data/    (annotations/, subsets/, images/)
  cache/   (text_queries_*.npy, encodes/{config}/{image_id}.npz)
  src/     (download_data.py, build_subsets.py, measure_artifacts.py, neuron_id.py,
            ttr_owlv2.py, run_encode.py, run_detect_eval.py, selective_prediction.py,
            analysis.py)
  results/ (runs/*.json, tables/, plots/, viz/, REPORT.md)
```

Every run writes a self-describing JSON (config snapshot, code-state hash, timestamps) so any reported number traces to a run file.

## WHAT NOT TO DO

- No training, fine-tuning, or weight updates anywhere; the intervention is activation editing only.
- Never identify neurons or tune n, cutoffs, or any design choice on the evaluation subsets.
- Do not change selection rules, thresholds, or matching between configs.
- Do not report any metric without its run artifact.

