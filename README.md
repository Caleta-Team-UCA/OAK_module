# OAK Module
OAK module that groups all the scripts and functions necessary for neonatal infant monitoring.

## Environment configuration
Execute with conda installed:
```
conda env create -f env.yml
conda activate oak
pip install -r requirements.txt
```

If new libraries are installed, it is necessary to execute the following command:
```
pip-compile --extra=dev
```

## Structure

![Classes diagram](img/classes_diagram.png?raw=true)

### Scripts

With the OAK camera plugged into your device, and the `oak` environment active (see [Environment configuration](#environment-configuration)) run the following command from the root folder:

```
python demo_video.py
```

By default, the script will load the session stored in the [demo folder](demo). This session last 20 seconds and it consist on three videos, each recorded by a different camera of the OAK device.
A window will appear on screen, showing the video from the center camera. When the neonate is detected, the script draws a bounding box around their face, and other surrounding the body.
The script also plots three graphs next to the image. Each graph shows a different score on real-time:
- Stress score (between 0 and 1)
- Limb stretch scores (between 0 and 1)
- Breath

Instead of feeding a video to the script, you can record on real-time:

```
python demo_cam.py
```

To stop the record, press "q" then "Ctrl + C".

## Raspberry installation
First install all libraries:
```
python3.7 -m pip install -r requirements.txt
python3.7 -m pip install -e .
```

Now install important cv2 dependencies:
```
sudo apt upgrade
sudo apt-get install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev libatlas-base-dev libjasper-dev  libqtgui4  libqt4-test
```