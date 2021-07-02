import depthai as dai
import cv2
from time import monotonic
import numpy as np
from time import sleep

body_path_model = "models/mobilenet-ssd_openvino_2021.2_8shave.blob"
face_path_model = "models/face-detection-openvino_2021.2_4shave.blob"
stress_path_model = "models/mobilenet_stress_classifier_2021.2.blob"
video_path = "videos/21-center-1.mp4"

LIST_LABELS = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",  # index 15
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
    "face",  # index 21, added by us
]


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


def frame_norm(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Normalizes the frame"""
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


if __name__ == "__main__":
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_1)

    in_frame = pipeline.createXLinkIn()
    in_frame.setStreamName("input")

    stress_in_frame = pipeline.createXLinkIn()
    stress_in_frame.setStreamName("stress_input")

    # Create Mobilenet and Face nodes
    nn_body = pipeline.createMobileNetDetectionNetwork()
    nn_body.setConfidenceThreshold(0.7)
    nn_body.setBlobPath(body_path_model)
    nn_body.setNumInferenceThreads(2)
    nn_body.input.setBlocking(False)
    nn_body.input.setQueueSize(1)

    nn_face = pipeline.createMobileNetDetectionNetwork()
    nn_face.setConfidenceThreshold(0.7)
    nn_face.setBlobPath(face_path_model)
    nn_face.setNumInferenceThreads(2)
    nn_face.input.setBlocking(False)
    nn_face.input.setQueueSize(1)

    # Create Stress Classifier Neural Network
    nn_stress = pipeline.createNeuralNetwork()
    nn_stress.setBlobPath(stress_path_model)
    nn_stress.setNumInferenceThreads(2)
    nn_stress.input.setBlocking(False)
    nn_stress.input.setQueueSize(1)

    # Links
    # Inputs
    in_frame.out.link(nn_body.input)
    in_frame.out.link(nn_face.input)

    stress_in_frame.out.link(nn_stress.input)

    # outputs
    body_out_frame = pipeline.createXLinkOut()
    body_out_frame.setStreamName("body_out")
    nn_body.out.link(body_out_frame.input)

    face_out_frame = pipeline.createXLinkOut()
    face_out_frame.setStreamName("face_out")
    nn_face.out.link(face_out_frame.input)

    stress_out_frame = pipeline.createXLinkOut()
    stress_out_frame.setStreamName("stress_out")
    nn_stress.out.link(stress_out_frame.input)

    # Initilize pipeline
    device = dai.Device(pipeline)

    # Queues
    in_q = device.getInputQueue(name="input", maxSize=1, blocking=False)
    stress_in_q = device.getInputQueue(name="stress_input", maxSize=1, blocking=False)

    body_out_q = device.getOutputQueue(name="body_out", maxSize=1, blocking=True)
    face_out_q = device.getOutputQueue(name="face_out", maxSize=1, blocking=True)
    stress_out_q = device.getOutputQueue(name="stress_out", maxSize=1, blocking=True)

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
        in_q.send(img)

        body_detections = body_out_q.tryGet()
        if body_detections:
            for d in body_detections.detections:
                label = LIST_LABELS[d.label]
                if label == "person":
                    bbox = frame_norm(frame, (d.xmin, d.ymin, d.xmax, d.ymax))

                    # Esto lo hago  por un bug que tiene en Windows
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = np.array(frame)
                    # ===========

                    frame = cv2.rectangle(
                        frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2
                    )

        face_detections = face_out_q.tryGet()
        if face_detections:
            if len(face_detections.detections) > 0:
                d = face_detections.detections[0]

                bbox = frame_norm(frame, (d.xmin, d.ymin, d.xmax, d.ymax))

                # Esto lo hago  por un bug que tiene en Windows
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = np.array(frame)
                # ===========

                face_frame = frame[bbox[1] : bbox[3], bbox[0] : bbox[2], :]
                frame = cv2.rectangle(
                    frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2
                )

                cv2.imshow("rgb2", face_frame)

                nn_data = dai.NNData()
                nn_data.setLayer("prob", to_planar(face_frame, (224, 224)))

                stress_in_q.send(nn_data)

                stress_nn_out = stress_out_q.tryGet()
                if stress_nn_out:
                    stress_labels = ["non_stress", "stress"]
                    stress_value = np.array(stress_nn_out.getFirstLayerFp16())[0]
                    label = stress_labels[stress_value > 0.5]

                    frame = cv2.putText(
                        frame,
                        f"{label} {stress_value:.2}",
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (36, 255, 12),
                        2,
                    )

        # Aquí simulamos que se estuviera haciendo
        # algún tipo de procesamiento con los datos
        sleep(0.05)

        cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord("q"):
            break
