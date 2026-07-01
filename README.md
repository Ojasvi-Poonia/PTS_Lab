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



The app now crashes after the performance changes. Here's the crash log: [paste the FATAL EXCEPTION / backtrace section from `adb logcat`].

Before changing anything, tell me which phase's change caused this based on the stack trace. Then:

1. Confirm the shared cached-boxes list is thread-safe — the background detection coroutine writes it while the render thread reads it. Use a concurrent/immutable structure or a lock/synchronized swap so reads and writes can't collide. This is the most likely cause.
2. Verify the detector input tensor shape matches the re-exported fixed model size (320/416) — check the preprocessing resizes to exactly that, or it's a shape-mismatch crash.
3. If it's a native crash on model load/inference, temporarily disable INT8 quantization and run the FP32/FP16 model to isolate whether quantization is the problem. Also confirm the XNNPACK EP is actually present in the ORT build and log an error (don't crash) if session creation fails.
4. Check the reused image buffers aren't shared between the camera-write and detector-read across threads — give the detector its own copy or double-buffer.

Fix only the phase that caused the crash first, then let me test before touching the others.
