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



Goal: my Android pointing app (camera on, point at an appliance, say on/off → toast) works but isn't smooth or realtime. Fix the performance. This is a speed problem, not an accuracy problem. Keep RF-DETR as the detector (optimize it, don't replace it). Don't use NNAPI — RF-DETR's ops aren't supported there.

First, before changing anything: read the camera, detection, hand-tracking, and overlay code and tell me in a few lines how the pipeline is currently wired and where you think the time is going. Then implement the changes below in phases, one phase at a time, and pause after each so I can test.

Phase 1 — Decouple detection from rendering:
Move RF-DETR inference onto its own background coroutine. It should write results into a shared, timestamped list of cached boxes. The camera preview and the pointing overlay must render every frame on their own and only read that cached list — they must never wait for detection to finish. The per-frame work is only: get the hand ray, smooth it, test the ray against the cached boxes, pick the nearest hit, draw. This is what makes pointing feel instant.

Phase 2 — Make detection event-driven:
Right now it runs on a fixed cadence. Instead, run detection when the camera actually moves (use IMU movement or a cheap frame-difference), and let it idle when the scene is still. Also run a single detection on the current frame the moment a voice command comes in if the ray isn't already on a cached box. Cap it around 5–10 per second when active.

Phase 3 — Speed up the model:
Use ONNX Runtime with the XNNPACK execution provider (CPU, multi-threaded, threads set to the number of performance cores). Log which provider actually loaded so we can confirm it's not silently falling back. Re-export RF-DETR with a fixed input size instead of dynamic. Lower the detector input resolution to 320 or 416. Run onnx-simplifier on the model and apply INT8 quantization. Do these model-side steps first — they usually get inference under budget on their own.

Phase 4 — Remove stutter:
Reuse image buffers instead of allocating a new bitmap or array every frame (per-frame allocations cause GC pauses that look like jank). Only convert a frame to the model's input tensor when it's actually being sent to the detector, not for every preview frame.

Keep the existing voice-to-toast logic; it should act on whichever appliance is currently selected. Add per-stage timing logs (camera, detection, hand, intersection, draw) so we can see the milliseconds.

Success looks like: overlay stays at 30fps+ with no stall while detection runs, switching between two visible devices is instant, and one detection inference is under ~150ms.
