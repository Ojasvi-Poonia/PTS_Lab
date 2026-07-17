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

Copy everything below this line into Cline as the task prompt.

---

## ROLE

You are implementing a research experiment end-to-end. Work in phases, stop at each checkpoint, and report results before continuing. You are building **ActiveDINO**: a training-free cascade where a cheap image-level ranker (CLIP) prunes the class vocabulary per image, and Grounding DINO (an early-fusion open-vocabulary detector) then detects only the retained classes.

## RESEARCH CONTEXT (read carefully — this motivates every design choice)

Early-fusion detectors like Grounding DINO pass the entire class vocabulary through cross-modal attention on every forward pass. Two consequences:

1. **Cost scales with vocabulary.** LVIS has 1,203 classes; the text encoder (BERT) has a ~256-token limit, so evaluating LVIS requires chunking the vocabulary into ~30–40 prompts and running a **full detector forward pass per chunk, per image**. Pruning the vocabulary to top-K classes cuts forward passes near-linearly.
2. **Accuracy degrades with vocabulary.** Prior work (aerial-imagery evaluation of Grounding DINO) showed that shrinking the prompt from 80 classes to only the ~3 ground-truth classes per image improved F1 ~15×, because cross-modal attention diffuses across semantically similar categories. So pruning is not just a speed lever — it may be an **accuracy** lever ("attention decongestion").

This mirrors ActiveSAM (arXiv 2606.16996), which did image-conditional class pruning for SAM 3 **segmentation** (+1.4 mIoU, up to 5.5× faster, training-free). Nobody has done it for **detection**. That is the paper.

### Hypotheses (pre-registered — report outcomes whether positive or negative)

- **H1 (pruner quality):** A frozen CLIP image–text ranker achieves shortlist recall@100 ≥ 95% overall and ≥ 90% on LVIS-rare categories.
- **H2 (efficiency):** At K=100, pruned inference uses ≥ 3× fewer detector forward passes than the full-vocabulary baseline with AP drop ≤ 0.5.
- **H3 (accuracy):** At some K in the sweep, pruned AP **exceeds** the full-vocabulary baseline by ≥ +1.0 AP (attention decongestion + false-positive suppression on absent classes).

### Honesty rules (non-negotiable)

- Never fabricate, estimate, or "fill in" a number. Every number in the report must come from an actual run whose logs exist on disk.
- If something fails (OOM, MPS op unsupported, download broken), report the failure and the fallback you took. Do not silently skip.
- Run the 5-image smoke test for every stage before any full run.
- All comparisons are **paired on identical images** with identical post-processing. Fix all seeds (42).

## HARDWARE & ENVIRONMENT

- Target: Apple M1, 8 GB RAM, PyTorch MPS. Everything must also run with `--device cuda` (Colab T4) unchanged; detect device automatically with an override flag.
- On MPS use fp32 (fp16 on MPS is flaky for these models). Batch size 1. Process images sequentially; cache aggressively so nothing is computed twice.
- Python 3.10+, venv. `requirements.txt` (pin versions after verifying):
  - `torch`, `torchvision`
  - `transformers>=4.44` (Grounding DINO + CLIP)
  - `lvis` (official LVIS API), `pycocotools`
  - `numpy`, `scipy` (Wilcoxon), `pandas`, `matplotlib`, `Pillow`, `tqdm`, `requests`
- Verify install by loading both models and running one forward pass on a test image before writing any pipeline code.

## MODELS (exact checkpoints)

- Detector: `IDEA-Research/grounding-dino-tiny` via `AutoProcessor` + `AutoModelForZeroShotObjectDetection`. (Config flag to swap in `grounding-dino-base` later; do NOT default to base on 8 GB.)
- Pruner: `openai/clip-vit-base-patch32` (`CLIPModel` + `CLIPProcessor`). Config flag to swap in `google/siglip-base-patch16-224` as an ablation.
- Everything frozen. No training, no gradient updates anywhere.

## DATA

- **LVIS v1 val** annotations: download `lvis_v1_val.json` (from the official LVIS site / `dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip`).
- Build a **400-image subset, seed 42, rare-stratified**: at least 250 images must contain ≥ 1 rare-frequency category. Save the image-id list to `data/subset_lvis_400.json`. Also support `--image_ids path.json` so an externally supplied id list can be dropped in for exact reproduction of prior work.
- Download only the subset's images using each image record's `coco_url` (LVIS val includes images from both COCO train2017 and val2017 — do not assume val2017 only). Store under `data/images/{image_id}.jpg`. Verify count and integrity.
- Also create `data/subset_lvis_50.json` (seed 42, same stratification) as the fast-iteration split, and use the first 5 ids of it as the smoke-test split.

### Class-name handling (this is a known pitfall — get it right)

- LVIS category `name` looks like `flip-flop_(sandal)`. Build `prompt_name`: replace `_` with space, strip parenthetical qualifiers → `flip-flop`. If stripping causes two categories to collide, keep the parenthetical text for those categories only.
- Persist a bidirectional map `prompt_name <-> category_id` to `cache/class_map.json`. Every stage uses this one map. Unit-test it: 1,203 entries, unique prompt names, spot-check 10 known categories.

## GROUNDING DINO INFERENCE WRAPPER (`src/detector_gdino.py`)

- Prompt format: lowercase class names joined as `"name1. name2. name3."` (period-separated — this is the format Grounding DINO expects).
- **Token-aware chunking:** build chunks with the model's own tokenizer so each prompt stays ≤ 250 tokens including specials (BERT limit ~256). Log the realized chunk count and mean classes/chunk. Chunk size must therefore be adaptive, not a fixed class count. Also expose `--classes_per_chunk` to force smaller chunks for the ablation in Phase 7.
- Thresholds: keep them LOW so AP evaluation sees the full score distribution — `box_threshold=0.05`, `text_threshold=0.05` (configurable). Do not pre-filter aggressively before eval.
- Post-processing: use `processor.post_process_grounded_object_detection`. Its signature differs across transformers versions (`threshold` vs `box_threshold`/`text_threshold`); inspect the installed version and adapt.
- **Label→category mapping:** returned labels are decoded phrases and can be partial or merged. Map each returned label to a `category_id` via exact match against the chunk's prompt names first; fall back to `difflib` closest match with similarity ≥ 0.8; below that, log to `results/unmatched_labels.log` and drop. Unit-test with multi-word and hyphenated names. On the smoke split, render 5 images with drawn boxes+labels to `results/viz/` and eyeball them before proceeding.
- **Merging chunks:** concatenate detections from all chunks for an image, keep global **top-300 by score** (LVIS protocol cap). No cross-chunk NMS beyond the model's own per-forward behavior — protocol must be identical for baseline and pruned runs. Log per-chunk score distribution stats (mean/max) per image to enable the Phase 7 drift analysis.
- Output format per run: LVIS results JSON — list of `{"image_id", "category_id", "bbox": [x,y,w,h], "score"}` — plus a per-image sidecar with timing and chunk metadata.
- Cache text-side encodings per unique chunk string where the API allows; never re-tokenize the same chunk twice.

## PRUNER (`src/pruner_clip.py`)

- Encode all 1,203 `prompt_name`s once with CLIP's text encoder using templates `["a photo of a {}", "a photo of the {}"]`, mean the embeddings, L2-normalize, cache to `cache/clip_text_lvis.npy`.
- Per image: one CLIP image embedding (cache to `cache/clip_img/{image_id}.npy`), cosine similarity against all classes, output the full 1,203-class ranking to `cache/rankings/{image_id}.json`. All top-K experiments slice this one ranking — the pruner never reruns per K.
- **Recall@K metric:** over all (image, positive-category) pairs in the subset, the fraction where the category ranks in the image's top-K. Report overall and split by LVIS frequency bin (rare / common / frequent), for K ∈ {25, 50, 100, 200, 400}. This is the H1 table and it gates everything downstream.
- Time the pruner (image encode + matmul) — it must be reported as part of end-to-end pruned latency.

## RUNS (all paired on the same subset, same post-processing)

1. **Baseline** (`run_baseline.py`): full 1,203-class vocabulary, chunked. This is the expensive run — start it on the 50-image split; run 400 only after everything else works (or on Colab).
2. **Pruned** (`run_pruned.py --K {25,50,100,200,400}`): per image, prompt only the top-K classes from the cached ranking, chunked the same token-aware way.
3. **Oracle-classes** (`run_oracle.py`): per image, prompt only the ground-truth positive categories. Upper bound on what perfect pruning buys; quantifies maximum attention decongestion.
4. Optional flag `--pruner owlv2_scores path.json` to accept an externally supplied per-image class ranking (e.g., from cached OWLv2 max-scores) as a drop-in pruner ablation.

## EVALUATION (`src/evaluate.py`)

- Official LVIS API, federated AP, 300 dets/image: report AP, AP_r, AP_c, AP_f per run.
- **Interpretation note to encode in the report:** under federated evaluation, pruning affects AP through two mechanisms — (a) missed positives when the pruner drops a present class (recall@K cost), and (b) suppressed false positives on images where a class is absent/negative (pruned classes emit no boxes there). Decompose: report ΔAP alongside recall@K, and compare pruned vs oracle to bound mechanism (a).
- **Score-drift analysis:** for GT boxes matched (IoU ≥ 0.5) in both baseline and pruned/oracle runs, compare the matched detection's score across conditions. If scores on present classes rise as the prompt shrinks, that is direct evidence of attention decongestion — plot score-delta distribution vs K.
- Per-image AP-proxy deltas for paired significance: Wilcoxon signed-rank across images (scipy), report p-values. State clearly that subset numbers are not comparable to full-split literature; only paired deltas matter.

## LATENCY PROFILING (`src/profile_latency.py`)

- On the 50-image split, measure per image: pruner time, text-encoding time, detector forward time per chunk, post-processing time, end-to-end. 3 warmup images excluded; synchronize the device (`torch.mps.synchronize()` / `torch.cuda.synchronize()`) around timers; use `time.perf_counter`.
- Report: end-to-end latency and #forward-passes vs K (baseline as K=1203), speedup factors, and a stacked breakdown mirroring a "prunable vs fixed cost" table. Record hardware string in the output.

## PHASES & CHECKPOINTS (stop and report after each)

- **Phase 0 — Setup:** venv, requirements, both models load, one forward each on a sample image. Deliver: `env_report.txt` (versions, device).
- **Phase 1 — Data:** annotations, class map + unit test, subsets built, subset images downloaded and verified. Deliver: counts, rare-category coverage stats.
- **Phase 2 — Pruner:** rankings cached for the 400 subset, recall@K table (overall + per frequency bin). **Gate H1 evaluated here.** If recall@100 < 90% overall, stop and report — pruned-run AP will be bounded by this and we need to see it before burning compute.
- **Phase 3 — Detector smoke:** 5-image end-to-end (baseline + K=100), visualizations rendered, label mapping verified, unmatched-label rate < 5%.
- **Phase 4 — 50-image split:** baseline, K sweep, oracle. Deliver: full metric + latency tables. Sanity: baseline AP should be plausible for a tiny zero-shot checkpoint on a small subset (single digits to low tens) — if it is ~0, debug the label mapping / thresholds before scaling.
- **Phase 5 — 400-image split:** repeat winners + baseline (flag long ETA first; recommend Colab if projected > 12 h on M1).
- **Phase 6 — Analysis:** `analysis.py` produces: AP/AP_r vs K curve with baseline and oracle lines; latency & forward-pass vs K; recall@K; score-drift plots; Wilcoxon tests; `results/REPORT.md` summarizing each hypothesis as SUPPORTED / NOT SUPPORTED with the exact numbers.
- **Phase 7 (stretch) — Prompt-composition measurement:** fixed image + fixed present class, vary co-prompt (chunk) size {5, 10, 20, 40, 80 classes} with random co-classes (seed 42, 5 draws); measure the matched GT box's score vs co-prompt size. This quantifies context-dependent scoring in early-fusion detectors — the "measurement section" of the eventual paper.

## REPO LAYOUT

```
activedino/
  README.md  requirements.txt  env_report.txt
  configs/default.yaml            # K sweep, thresholds, chunk params, device, paths
  data/    (annotations, subsets, images/)
  cache/   (class_map.json, clip_text_lvis.npy, clip_img/, rankings/, text_chunks/)
  src/     (download_data.py, build_subsets.py, pruner_clip.py, detector_gdino.py,
            run_baseline.py, run_pruned.py, run_oracle.py, evaluate.py,
            profile_latency.py, analysis.py)
  results/ (runs/*.json, viz/, plots/, tables/, unmatched_labels.log, REPORT.md)
```

Every run writes a self-describing JSON (config snapshot + git-style hash of code state + timestamps) so any table in REPORT.md can be traced to a run file.

## WHAT NOT TO DO

- Do not fine-tune, prompt-tune, or update any weights.
- Do not change thresholds, chunking, or post-processing between baseline and pruned runs.
- Do not evaluate on images used to make any design decision beyond the declared subsets.
- Do not report any metric without the run artifact that produced it.
