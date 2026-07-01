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

The app closes by itself (especially when idle) but you say the log shows no crash. That means it's likely NOT crashing — it's either being killed or finishing cleanly. Help me find which. 

1. Tell me every place in the code that could end the app: any call to finish(), finishAffinity(), System.exit(), activity recreation, or the camera/detection being torn down in onPause/onStop without being restored in onResume. List them.
2. Add a log line in onPause, onStop, onDestroy, and onResume of the main activity so we can see, in logcat, exactly which lifecycle callback fires right before it closes.
3. Check whether detection or any blocking/locking call is running on the main thread — an ANR would close the app without a FATAL EXCEPTION. 
4. Confirm the detection coroutine is properly cancelled in onPause/onStop and the CameraX use cases are unbound and rebound across the lifecycle, so a screen-timeout doesn't kill the app.

Then reproduce: I'll watch logcat while it closes and tell you which lifecycle log printed last.
