import depthai as dai
import cv2
from time import monotonic
import numpy as np

body_path_model = "models/mobilenet-ssd_openvino_2021.2_8shave.blob"
face_path_model = "models/face-detection-openvino_2021.2_4shave.blob"
video_path = "videos/22-center-2.mp4"


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

    body_in_frame = pipeline.createXLinkIn()
    body_in_frame.setStreamName("body_input")

    face_in_frame = pipeline.createXLinkIn()
    face_in_frame.setStreamName("face_input")

    display_in_frame = pipeline.createXLinkIn()
    display_in_frame.setStreamName("display_input")

    # Create Mobilenet and Face nodes
    nn_body = pipeline.createMobileNetDetectionNetwork()
    nn_body.setConfidenceThreshold(0.7)
    nn_body.setBlobPath(body_path_model)
    nn_body.setNumInferenceThreads(2)
    nn_body.input.setBlocking(False)

    nn_face = pipeline.createMobileNetDetectionNetwork()
    nn_face.setBlobPath(face_path_model)

    # Links
    # Inputs
    body_in_frame.out.link(nn_body.input)
    face_in_frame.out.link(nn_face.input)

    # outputs
    body_out_frame = pipeline.createXLinkOut()
    body_out_frame.setStreamName("body_out")
    nn_body.out.link(body_out_frame.input)

    face_out_frame = pipeline.createXLinkOut()
    face_out_frame.setStreamName("face_out")
    nn_face.out.link(face_out_frame.input)

    display_out_frame = pipeline.createXLinkOut()
    display_out_frame.setStreamName("display_out")
    display_in_frame.out.link(display_out_frame.input)

    # Initilize pipeline
    device = dai.Device(pipeline)
    device.setLogLevel(dai.LogLevel.TRACE)

    # Queues
    body_in_q = device.getInputQueue(name="body_input", maxSize=4, blocking=False)
    face_in_q = device.getInputQueue(name="face_input", maxSize=4, blocking=False)
    display_in_q = device.getInputQueue(name="display_input", maxSize=4, blocking=False)

    body_out_q = device.getOutputQueue(name="body_out", maxSize=4, blocking=False)
    face_out_q = device.getOutputQueue(name="face_out", maxSize=4, blocking=False)
    display_out_q = device.getOutputQueue(name="display_out", maxSize=4, blocking=False)

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        read_correctly, frame = cap.read()
        if not read_correctly:
            break

        img = dai.ImgFrame()
        img.setData(to_planar(frame, (300, 300)))
        img.setTimestamp(monotonic())
        img.setWidth(300)
        img.setHeight(300)
        body_in_q.send(img)
        # face_in_q.send(img)
        display_in_q.send(img)

        # print(body_out_q.tryGet(), face_out_q.tryGet(), display_out_q.tryGet())

        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord("q"):
            break
