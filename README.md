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

After the performance refactor, the app runs but shows NOTHING — no hand detection and no object detection, and the debug results bar is empty. Separately, if left idle it closes itself. It worked before the refactor.

Both detectors being dead at once means it's not the model — it's frame flow or result wiring. Investigate in this order and tell me the cause before fixing:

1. CameraX: confirm imageProxy.close() (or the equivalent frame release) is still called for EVERY analyzed frame. If frames aren't released, the analyzer stalls after one frame and everything freezes. This is the top suspect.
2. Confirm the overlay and the debug results bar read the SAME shared cached-boxes object the detection coroutine writes to — not the old source that's no longer populated.
3. Confirm the hand tracker still receives frames. If frame→tensor conversion was gated to detector-only, the hand pipeline may be starved; make sure hand tracking runs every frame independently.
4. Add a log where the detection gate decides to fire. If it never fires when the phone is held still, the motion threshold is too high or inverted — lower it and make detection also run at least at a slow baseline, not only on motion.
5. Idle crash: wrap the background detection/motion coroutines in a SupervisorJob with an exception handler that logs instead of crashing. Here's the crash log: [paste the FATAL EXCEPTION block from adb logcat].

Fix the frame-flow issue first (#1–#3) so detection shows again, then the idle crash. Let me test after each.
