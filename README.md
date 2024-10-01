# thorlabs_CS165MU1
Python device adaptor: Thorlabs CS165MU1 Zelux 1.6 MP Monochrome CMOS Camera.
## Quick start:
- Install the "ThorCam" GUI (from Thorlabs) and check the camera. It should be 
straightforward run live mode and save an image etc.
- Download this repository and run "thorlabs_CS165MU1.py" with essential the .dll
files in the same folder (copies included here for convenience).

**Note:** Using the PC (e.g. web browsing) while recording to memory can trigger "LOST FRAME TIMEOUT!" if the PC can't keep up with the camera. So it's best to leave the computer alone if running an important acquisition.

![social_preview](https://github.com/amsikking/thorlabs_CS165MU1/blob/main/social_preview.png)

## Details:
- Installing ThorCam should also put the critical SDK folder "Scientific 
Camera Interfaces" into the location:
"C:\Program Files\Thorlabs\Scientific Imaging\Scientific Camera Support"
- The C API used to generate this adaptor is called "Thorlabs_Camera_C_API_Reference.chm"
- The essential .dll files are in "C:\Users\localuser01\Desktop\Scientific Camera Interfaces\SDK\Native Toolkit\dlls".
("thorlabs_tsi_camera_sdk.dll", "thorlabs_tsi_usb_hotplug_monitor.dll" and "thorlabs_tsi_zelux_camera_device.dll")
- Some useful Python examples (like "tl_camera.py") are in "C:\Users\localuser01\Desktop\Scientific Camera Interfaces\SDK\Python Toolkit"
