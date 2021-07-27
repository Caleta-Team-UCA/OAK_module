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

![Classes diagram](classes_diagram.png?raw=true)

### Scripts

With the OAK camera plugged into your device, run the following command from the root folder:

```
python oak/run_pipeline.py
```

The camera will start recording, and you will see on screen two windows. The first shows the image on real-time, with a bounding box surrounding the face and other around the body. The second window depicts the depth map of the very same image.

Instead of recording on real-time, you can feed a video to the script:

```
python oak/run_pipeline.py --video-path PATH
```

In this case, only one window is shown: the image with the bounding boxes. No depth map can be computed from a pre-recorded video.

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