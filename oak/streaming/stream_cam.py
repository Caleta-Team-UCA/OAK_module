import queue
import subprocess as sp
import threading

import cv2
import depthai as dai

fps = 30
width = 1920
height = 1080
frame_queue = queue.Queue()
rtmp_url = "rtsp://vai.uca.es:1935/mystream"
command = [
    "ffmpeg",
    "-y",
    "-f",
    "rawvideo",
    "-vcodec",
    "rawvideo",
    "-pix_fmt",
    "bgr24",
    "-s",
    "{}x{}".format(width, height),
    "-i",
    "-",
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-preset",
    "ultrafast",
    "-f",
    "rtsp",
    rtmp_url,
]
p = None


def push_frame():
    global command, frame_queue, p

    while True:
        if len(command) > 0:
            p = sp.Popen(command, stdin=sp.PIPE)
            break

    while True:
        if frame_queue.empty() != True:
            frame = frame_queue.get()
            p.stdin.write(frame.tostring())


def gen_frames():  # generate frame by frame from camera
    while True:
        # Create pipeline
        pipeline = dai.Pipeline()

        # Define source and output
        camRgb = pipeline.createColorCamera()
        xoutVideo = pipeline.createXLinkOut()

        xoutVideo.setStreamName("video")

        # Properties
        camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setPreviewSize(width, height)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        camRgb.setVideoSize(width, height)
        ve2 = pipeline.createVideoEncoder()

        ve2.setDefaultProfilePreset(
            width, height, fps, dai.VideoEncoderProperties.Profile.MJPEG
        )
        camRgb.video.link(ve2.input)

        # camRgb.setVideoSize(width,height)

        xoutVideo.input.setBlocking(False)
        xoutVideo.input.setQueueSize(1)

        # Linking
        camRgb.video.link(xoutVideo.input)

        with dai.Device(pipeline) as device:
            # Output queue will be used to get the rgb frames from the output defined above
            qRgb = device.getOutputQueue(name="video", maxSize=1, blocking=False)

            while True:
                inRgb = (
                    qRgb.get()
                )  # blocking call, will wait until a new data has arrived
                frame = inRgb.getCvFrame()  # read the camera frame
                # cv2.imshow("bgr", frame)
                ret, buffer = cv2.imencode(".jpg", frame)

                if not ret:
                    print("Opening camera is failed")

                    pass
                else:
                    frame_queue.put(frame)
                # yield (b'--frame\r\n'
                #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


if __name__ == "__main__":
    try:
        threads = [
            threading.Thread(target=gen_frames, args=()),
            threading.Thread(target=push_frame, args=()),
        ]
        [thread.setDaemon(True) for thread in threads]
        [thread.start() for thread in threads]
    except:
        pass
