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

Context & Objective
Build a modular Python Proof of Concept (POC) for a training-free open-vocabulary object detection method using Hugging Face's OWLv2. The goal is to implement "Lever 1" from our strategic memo: fusing text-derived query embeddings with image-derived embeddings (visual exemplars) at inference time. The backbone must remain completely frozen.
Architecture & Interface Requirements
 Language: Python 3.10+
 Core Libraries: ⁠transformers⁠, ⁠torch⁠, ⁠Pillow⁠, ⁠matplotlib⁠
 Model: Use ⁠google/owlv2-base-patch16-ensemble⁠.
 Modular Setup: Create a clear, easily editable "Configuration Block" at the very top of the script. This block must define dummy variables for:
 ⁠MAIN_IMAGE_PATH⁠ (the target image to run detection on)
 ⁠TEXT_QUERIES⁠ (a list of text strings)
 ⁠EXEMPLAR_PATHS⁠ (a dictionary mapping each text query to a list of file paths for ground-truth exemplar image crops)
 ⁠FUSION_ALPHA⁠ (a float between 0.0 and 1.0 to weight the fusion)
 Note: Use placeholders or fetch a sample COCO image for the initial run, but ensure I can easily swap these out with local file paths later.
Implementation Steps
1. Base Extraction (Text & Visual)
 Load the OWLv2 model and processor. Ensure inference is run in ⁠torch.no_grad()⁠ to guarantee the model remains frozen.
 Text Embeddings: Process the ⁠TEXT_QUERIES⁠ to extract the baseline text query embeddings.
 Visual Embeddings: Process the images in ⁠EXEMPLAR_PATHS⁠ using the image-guided path to extract visual embeddings. Ensure you handle NMS correctly (image-guided path defaults to ⁠nms_threshold=0.3⁠). If multiple crops exist for a single class, average their embeddings into a single representation.
2. Implement Fusion Strategies
Write modular functions for the two fusion candidates:
 Strategy A (Embedding-level): Calculate a convex combination of the text embedding and the averaged visual exemplar embedding using the ⁠FUSION_ALPHA⁠ weight. Compute the final bounding box logits using this fused embedding.
 Strategy B (Score-level): Perform late fusion. Calculate the detection logits for the text embeddings and the visual embeddings separately, then combine the logit streams using a configurable max or mean operation.
3. Output & Evaluation
 Implement a visualization function using ⁠matplotlib⁠ to draw the bounding boxes on the main image.
 The script should generate a side-by-side visual plot or clear console output comparing three states:
1. Text-only baseline
2. Strategy A results
3. Strategy B results
 This output must allow me to easily measure the "fusion oracle" ceiling (i.e., whether the ground-truth exemplars improve upon the text-only baseline).
Strict Constraints
 Do not introduce any training loops or parameter updates.
 Do not hardcode image paths deep within the logic; everything must be driven by the top-level Configuration Block.
