import depthai as dai
import cv2
from time import monotonic
import numpy as np
from time import sleep

body_path_model = "models/mobilenet-ssd_openvino_2021.2_8shave.blob"
face_path_model = "models/face-detection-openvino_2021.2_4shave.blob"
video_path = "videos/21-center-2.mp4"


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


def process_frame(frame: np.array, width: int, height: int) -> dai.ImgFrame:
    # Generate ImgFrame to use as input of the Pipeline
    img = dai.ImgFrame()
    img.setData(to_planar(frame, (width, height)))
    img.setTimestamp(monotonic())
    img.setWidth(width)
    img.setHeight(height)

    return img


if __name__ == "__main__":
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_1)

    cam = pipeline.createColorCamera()
    cam.setPreviewSize(300, 300)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    # Define output stream
    cam_xout = pipeline.createXLinkOut()
    cam_xout.setStreamName("cam_out")
    cam.preview.link(cam_xout.input)
    # Define control stream
    control_in = pipeline.createXLinkIn()
    control_in.setStreamName("input")
    control_in.out.link(cam.inputControl)

    in_frame = pipeline.createXLinkIn()
    in_frame.setStreamName("input")

    # Create Mobilenet and Face nodes
    nn_body = pipeline.createMobileNetDetectionNetwork()
    nn_body.setConfidenceThreshold(0.7)
    nn_body.setBlobPath(body_path_model)
    nn_body.setNumInferenceThreads(2)
    nn_body.input.setBlocking(False)
    nn_body.input.setQueueSize(1)

    nn_face = pipeline.createMobileNetDetectionNetwork()
    nn_face.setBlobPath(face_path_model)
    nn_body.setConfidenceThreshold(0.7)
    nn_body.setNumInferenceThreads(2)
    nn_body.input.setBlocking(False)
    nn_body.input.setQueueSize(1)

    # Links
    # Inputs
    cam.preview.link(nn_body.input)
    cam.preview.link(nn_face.input)

    # outputs
    body_out_frame = pipeline.createXLinkOut()
    body_out_frame.setStreamName("body_out")
    nn_body.out.link(body_out_frame.input)

    face_out_frame = pipeline.createXLinkOut()
    face_out_frame.setStreamName("face_out")
    nn_face.out.link(face_out_frame.input)

    # Initilize pipeline
    device = dai.Device(pipeline)

    # Queues
    cam_out = device.getOutputQueue("cam_out", 1, True)

    body_out_q = device.getOutputQueue(name="body_out", maxSize=1, blocking=True)
    face_out_q = device.getOutputQueue(name="face_out", maxSize=1, blocking=True)
    cv2.namedWindow("rgb", cv2.WINDOW_NORMAL)
    while True:
        frame = (
            np.array(cam_out.get().getData())
            .reshape((3, 300, 300))
            .transpose(1, 2, 0)
            .astype(np.uint8)
        )
        print(body_out_q.tryGet(), face_out_q.tryGet())

        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord("q"):
            break
