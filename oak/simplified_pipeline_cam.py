from time import sleep

import cv2
import depthai as dai
import numpy as np
import typer

from oak.utils.params import LIST_LABELS
from oak.utils.opencv import frame_norm


def main(
    body_path_model="models/mobilenet-ssd_openvino_2021.2_8shave.blob",
    face_path_model="models/face-detection-openvino_2021.2_4shave.blob",
):
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_1)

    cam = pipeline.createColorCamera()
    cam.setPreviewSize(300, 300)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
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
    nn_face.setConfidenceThreshold(0.7)
    nn_face.setNumInferenceThreads(2)
    nn_face.input.setBlocking(False)
    nn_face.input.setQueueSize(1)

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
        cam_frame = cam_out.tryGet()
        if cam_frame:
            frame = (
                np.array(cam_frame.getData())
                .reshape((3, 300, 300))
                .transpose(1, 2, 0)
                .astype(np.uint8)
            )

            body_detections = body_out_q.tryGet()
            if body_detections:
                for d in body_detections.detections:
                    label = LIST_LABELS[d.label]
                    if label == "person":
                        bbox = frame_norm(frame, (d.xmin, d.ymin, d.xmax, d.ymax))
                        print(bbox, frame.shape)

                        # Esto lo hago  por un bug que tiene en Windows
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frame = np.array(frame)
                        # ===========

                        cv2.rectangle(
                            frame,
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            (255, 0, 0),
                            2,
                        )

            face_detections = face_out_q.tryGet()
            if face_detections:
                if len(face_detections.detections) > 0:
                    d = face_detections.detections[0]

                    bbox = frame_norm(frame, (d.xmin, d.ymin, d.xmax, d.ymax))
                    print(bbox, frame.shape)

                    # Esto lo hago  por un bug que tiene en Windows
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = np.array(frame)
                    # ===========

                    cv2.rectangle(
                        frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2
                    )

            cv2.imshow("rgb", frame)

        # Simulo algún tipo de procesamiento
        sleep(0.05)

        if cv2.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    typer.run(main)
