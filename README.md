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
# CLINE PROMPT — ActiveDINO: Image-Conditional Vocabulary Pruning for Early-Fusion Open-Vocabulary Detectors
# Stage A: COCO (pipeline validation + accuracy probe) → Stage B: LVIS (efficiency + headline)

Copy everything below this line into Cline as the task prompt.

---

## ROLE

You are implementing a research experiment end-to-end. Work in phases, stop at each checkpoint, and report results before continuing. You are building **ActiveDINO**: a training-free cascade where a cheap image-level ranker (CLIP) prunes the class vocabulary per image, and Grounding DINO (an early-fusion open-vocabulary detector) then detects only the retained classes.

We run **COCO first** (small vocabulary, cheap, in-distribution) to validate every component and measure the accuracy mechanism, then scale to **LVIS** where the efficiency claim lives.

## RESEARCH CONTEXT (motivates every design choice)

Early-fusion detectors like Grounding DINO pass the whole class vocabulary through cross-modal attention on every forward pass. Two consequences:

1. **Cost scales with vocabulary.** LVIS has 1,203 classes; the text encoder (BERT) has a ~256-token limit, so LVIS needs the vocabulary chunked into ~30–40 prompts = a full detector forward pass per chunk, per image. Pruning to top-K classes cuts forward passes near-linearly. **COCO's 80 classes fit in ~1–2 chunks, so COCO cannot demonstrate meaningful speedup — do not claim one there.**
2. **Accuracy degrades with vocabulary.** Prior aerial-imagery work showed shrinking a Grounding DINO prompt from 80 classes to only the ~3 ground-truth classes per image improved F1 ~15× — cross-modal attention diffuses across similar categories. So pruning is potentially an **accuracy** lever ("attention decongestion"), and an 80-class COCO prompt is already large enough to probe this.

This mirrors ActiveSAM (arXiv 2606.16996), which did image-conditional class pruning for SAM 3 **segmentation** (+1.4 mIoU, up to 5.5× faster, training-free). Nobody has done it for **detection**.

## HYPOTHESES (pre-registered — report outcomes whether positive or negative)

Stage A (COCO, in-distribution — a saturation probe, so effects may be small):
- **A1 (pruner):** CLIP shortlist recall@20 ≥ 95% over (image, positive-class) pairs on the COCO subset.
- **A2 (accuracy):** For some K in {5,10,20,40}, pruned mAP − baseline mAP ≥ +0.5 (mechanism: attention decongestion on present classes + false-positive suppression on absent classes). If ΔmAP lands in (−0.3, +0.5), record it as neutral — Stage B proceeds regardless, because the efficiency claim does not depend on A2.

Stage B (LVIS, rare-stratified):
- **B1 (pruner):** recall@100 ≥ 95% overall and ≥ 90% on rare categories.
- **B2 (efficiency):** at K=100, ≥ 3× fewer detector forward passes than full-vocabulary baseline with AP drop ≤ 0.5.
- **B3 (accuracy):** some K achieves ΔAP ≥ +1.0 vs baseline.

## HONESTY RULES (non-negotiable)

- Never fabricate, estimate, or "fill in" a number. Every number in the report must trace to a run artifact on disk.
- If something fails (OOM, unsupported MPS op, broken download), report the failure and the fallback taken. Never silently skip.
- Run the 5-image smoke test for every stage before any full run.
- All comparisons are **paired on identical images** with identical post-processing. Fix all seeds (42).

## HARDWARE & ENVIRONMENT

- Target: Apple M1, 8 GB RAM, PyTorch MPS. Must also run unchanged with `--device cuda` (Colab T4); auto-detect device with an override flag.
- fp32 on MPS (fp16 there is flaky). Batch size 1. Sequential images; cache aggressively so nothing is computed twice.
- Python 3.10+, venv. `requirements.txt` (pin after verifying): `torch`, `torchvision`, `transformers>=4.44`, `pycocotools`, `lvis`, `numpy`, `scipy`, `pandas`, `matplotlib`, `Pillow`, `tqdm`, `requests`.
- Verify install by loading both models and running one forward pass each on a test image before writing pipeline code.

## MODELS (exact checkpoints, all frozen — no training anywhere)

- Detector: `IDEA-Research/grounding-dino-tiny` via `AutoProcessor` + `AutoModelForZeroShotObjectDetection` (config flag for `grounding-dino-base` later; do not default to base on 8 GB).
- Pruner: `openai/clip-vit-base-patch32` (`CLIPModel` + `CLIPProcessor`); config flag for `google/siglip-base-patch16-224` as an ablation.

## SHARED COMPONENTS (dataset-agnostic — build once, use in both stages)

### Detector wrapper (`src/detector_gdino.py`)
- Prompt format: lowercase class names joined `"name1. name2. name3."` (period-separated).
- **Token-aware chunking:** build chunks with the model's own tokenizer so each prompt ≤ 250 tokens incl. specials. Log realized chunk count and classes/chunk. Also expose `--classes_per_chunk` to force small chunks (needed for the prompt-composition experiment).
- Low thresholds so AP evaluation sees the full score distribution: `box_threshold=0.05`, `text_threshold=0.05` (configurable). No aggressive pre-filtering.
- Post-processing via `processor.post_process_grounded_object_detection`; its signature differs across transformers versions (`threshold` vs `box_threshold`/`text_threshold`) — inspect the installed version and adapt.
- **Label→category mapping:** returned labels are decoded phrases, possibly partial/merged. Exact match against the chunk's prompt names first; fall back to `difflib` similarity ≥ 0.8; below that log to `results/unmatched_labels.log` and drop. Unit-test with multi-word names ("traffic light", "dining table", "flip-flop"). On each smoke split, render 5 images with boxes+labels to `results/viz/` and eyeball before proceeding.
- **Merging chunks:** concatenate all chunks' detections per image, keep global top-N by score (N per dataset protocol below). No extra cross-chunk NMS — identical protocol for baseline and pruned runs. Log per-chunk score stats per image for the drift analysis.
- Output per run: results JSON in the dataset's official format (`{"image_id","category_id","bbox":[x,y,w,h],"score"}`) + per-image sidecar with timing and chunk metadata. Cache text-side work per unique chunk string.

### Pruner (`src/pruner_clip.py`)
- Encode all class prompt-names once with templates `["a photo of a {}", "a photo of the {}"]`, mean, L2-normalize → `cache/clip_text_{dataset}.npy`.
- Per image: one CLIP image embedding (`cache/clip_img/{image_id}.npy`), cosine sim vs all classes, save the **full ranking** to `cache/rankings/{dataset}/{image_id}.json`. All K values slice this one ranking — pruner never reruns per K.
- **Recall@K:** fraction of (image, positive-category) pairs where the category is in the image's top-K. COCO: overall. LVIS: overall + rare/common/frequent split.
- Time the pruner (image encode + matmul); it counts inside end-to-end pruned latency.
- Accept `--pruner external path.json` for a drop-in per-image ranking from another source (e.g., cached OWLv2 max-scores) as an ablation.

### Runs (same trio in both stages, always paired on the same subset)
1. **Baseline:** full vocabulary, chunked.
2. **Pruned:** per image, top-K classes from the cached ranking (K sweep per stage).
3. **Oracle-classes:** per image, only ground-truth positive categories — upper bound on perfect pruning, isolates maximum attention decongestion.

### Evaluation (`src/evaluate.py`)
- **Decomposition note (encode in REPORT.md):** pruning moves AP via (a) missed positives when the pruner drops a present class (recall@K cost) and (b) suppressed false positives on images where the class is absent. Report ΔAP alongside recall@K; compare pruned vs oracle to bound mechanism (a).
- **Score-drift analysis:** for GT boxes matched (IoU ≥ 0.5) in both baseline and pruned/oracle runs, compare the matched detection's score across conditions; plot score-delta vs K. Rising scores on present classes as the prompt shrinks = direct evidence of attention decongestion.
- Paired per-image deltas + Wilcoxon signed-rank (scipy). State that subset numbers aren't comparable to full-split literature; paired deltas are the result.

### Latency (`src/profile_latency.py`)
- Per image: pruner time, text-encoding, detector forward per chunk, post-processing, end-to-end. 3 warmup images excluded; synchronize device around timers; `time.perf_counter`; record hardware string.

## STAGE A — COCO (build + validate + accuracy probe)

### Data
- `instances_val2017.json` (from the official `annotations_trainval2017.zip`).
- **200-image subset of val2017, seed 42** — this mirrors the prior OWLv2 study for continuity. Support `--image_ids path.json` to inject an externally supplied id list for exact reuse of the earlier subset. Download only the subset's images via each record's `coco_url` → `data/images/coco/{image_id}.jpg`; verify count/integrity.
- Fast-iteration split: first 50 ids; smoke split: first 5.
- Class map: the 80 COCO names, lowercased, as-is (several are already multi-word). Persist `cache/class_map_coco.json`; unit-test 80 unique entries.

### Stage-A specifics
- First, tokenize the full 80-name prompt and **log how many chunks it needs (expect 1–2)**. Put this number in the report — it is why COCO cannot show speedup.
- K sweep: {5, 10, 20, 40}. Oracle uses GT classes (typically ~3/image).
- Eval: pycocotools mAP (AP@[.50:.95]), also AP50; standard 100 dets/image cap.
- Latency measured anyway (cheap), reported as "no meaningful speedup expected at 80 classes" context.

### Stage-A phases (stop and report after each)
- **Phase 0 — Setup:** venv, requirements, both models load + one forward each. Deliver `env_report.txt`.
- **Phase 1 — COCO data:** annotations, class map + test, subset built, images downloaded.
- **Phase 2 — Pruner on COCO:** rankings for 200 images; recall@K table for K∈{5,10,20,40}. **Gate A1 evaluated.** If recall@20 < 90%, stop and report before any detector runs.
- **Phase 3 — Detector smoke:** 5 images end-to-end (baseline + K=20), visualizations rendered, unmatched-label rate < 5%.
- **Phase 4 — 50-image split:** baseline, K sweep, oracle → metric + latency tables. Sanity: zero-shot GDINO-tiny mAP on COCO should be plausibly nonzero double digits on a small subset; if ~0, debug label mapping/thresholds before scaling.
- **Phase 5 — 200-image split:** full paired runs → **Gate A2 evaluated**, score-drift plots, Wilcoxon.
- **Phase 6 — Stage-A report:** `results/REPORT_coco.md`: A1/A2 verdicts with exact numbers, decongestion evidence (score-drift), oracle gap, and an explicit "what COCO can and cannot show" paragraph.
- **Phase 6b (cheap, recommended) — Prompt-composition measurement:** fixed image + fixed present class, co-prompt size {5,10,20,40,80} with random co-classes (seed 42, 5 draws each); plot the matched GT box's score vs co-prompt size. This quantifies context-dependent scoring in early-fusion detectors and is the measurement section of the eventual paper.

**Decision rule:** proceed to Stage B once the pipeline is verified (Phases 0–5 complete), regardless of the A2 verdict — the efficiency headline lives in Stage B. Only stop if A1 fails badly (<90%) or the pipeline itself is broken.

## STAGE B — LVIS (rare-stratified; efficiency + headline)

### Data
- `lvis_v1_val.json` (official LVIS release).
- **400-image subset, seed 42, rare-stratified:** ≥ 250 images contain ≥ 1 rare-frequency category. Save id list; support `--image_ids`. Download subset images via each record's `coco_url` — **LVIS val includes images from both COCO train2017 and val2017; do not assume val2017 only.** 50-image fast split; 5-image smoke split.
- Class map pitfall: LVIS `name` looks like `flip-flop_(sandal)`. Build `prompt_name`: `_`→space, strip parenthetical qualifiers; if stripping collides two categories, keep parentheses for those only. Persist `cache/class_map_lvis.json`; unit-test 1,203 unique entries; spot-check 10.

### Stage-B specifics
- Baseline: all 1,203 classes, token-aware chunks (~30–40 forward passes/image). This is the expensive run — 50-image split first; before the 400-image run, project ETA and flag if > 12 h on M1 (recommend `--device cuda` on Colab).
- K sweep: {25, 50, 100, 200, 400}.
- Eval: official LVIS API, federated AP, 300 dets/image; report AP, AP_r, AP_c, AP_f.
- Latency: end-to-end and #forward-passes vs K (baseline = K=1203), speedup factors, stacked cost breakdown.

### Stage-B phases
- **Phase 7 — LVIS data:** annotations, class map + tests, subsets, images.
- **Phase 8 — Pruner on LVIS:** rankings for 400 images; recall@K overall + per frequency bin. **Gate B1.** If recall@100 < 90% overall, stop and report before detector runs.
- **Phase 9 — Smoke + 50-image split:** baseline, K sweep, oracle → tables.
- **Phase 10 — 400-image split:** winners + baseline (ETA check first).
- **Phase 11 — Final analysis (`analysis.py`):** AP/AP_r vs K with baseline and oracle lines; latency & forward-passes vs K; recall@K; score-drift; Wilcoxon; `results/REPORT.md` with every hypothesis marked SUPPORTED / NOT SUPPORTED and exact numbers.

## REPO LAYOUT

```
activedino/
  README.md  requirements.txt  env_report.txt
  configs/default.yaml            # dataset, K sweep, thresholds, chunk params, device, paths
  data/    (annotations/, subsets/, images/coco/, images/lvis/)
  cache/   (class_map_*.json, clip_text_*.npy, clip_img/, rankings/, text_chunks/)
  src/     (download_data.py, build_subsets.py, pruner_clip.py, detector_gdino.py,
            run_baseline.py, run_pruned.py, run_oracle.py, evaluate.py,
            profile_latency.py, analysis.py)
  results/ (runs/*.json, viz/, plots/, tables/, unmatched_labels.log,
            REPORT_coco.md, REPORT.md)
```

Every run writes a self-describing JSON (config snapshot + code-state hash + timestamps) so any number in a report traces to a run file.

## WHAT NOT TO DO

- Do not fine-tune, prompt-tune, or update any weights.
- Do not change thresholds, chunking, or post-processing between baseline and pruned runs.
- Do not claim a speedup from Stage A — say explicitly that COCO's vocabulary is too small to test efficiency.
- Do not report any metric without the run artifact that produced it.
