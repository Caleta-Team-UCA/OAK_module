from abc import abstractmethod
from collections import namedtuple
from typing import Optional

import depthai as dai
import numpy as np

from oak.utils import process_frame

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


class OAKparent(dai.Pipeline):
    device: dai.Device = None
    in_q: dai.DataInputQueue = None
    body_out_q: dai.DataOutputQueue = None
    face_out_q: dai.DataOutputQueue = None
    display_out_q: dai.DataOutputQueue = None
    stress_in_q: dai.DataInputQueue = None
    stress_out_q: dai.DataOutputQueue = None

    def __init__(
        self,
        path_model_body: str = "models/mobilenet-ssd_openvino_2021.2_8shave.blob",
        path_model_face: str = "models/face-detection-openvino_2021.2_4shave.blob",
        path_model_stress: Optional[str] = "models/stress_classifier_2021.2.blob",
    ):
        """Create and configure the parameters of the pipeline.

        Parameters
        ----------
        path_model_body : str
            Path to body detection ".blob" model, by default
            "models/mobilenet-ssd_openvino_2021.2_8shave.blob"
        path_model_face : str
            Path to face detection ".blob" model, by default
            "models/face-detection-openvino_2021.2_4shave.blob"
        path_model_stress : Optional[str], optional
            Path to stress classification ".blob" model, by default
            "models/stress_classifier_2021.2.blob"
        """
        super(OAKparent, self).__init__()
        self.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_1)

        self._create_in_stream()
        self._create_body_face(path_model_body, path_model_face)

        if path_model_stress is not None:
            self._create_stress_in_stream()
            self._create_stress(path_model_stress)
            self.stress_bool = True
        else:
            self.stress_bool = False

        self._link_display()

    # ========= PRIVATE =========
    def _create_body_face(self, body_path_model: str, face_path_model: str):
        """Initializes neural networks used for detecting bodies and faces
        Parameters
        ----------
        body_path_model : str
            Path to the body blob model
        face_path_model : str
            Path to the face blob model
        """
        # BODY NN
        # Define a neural network that will make predictions based on the source frames
        self.nn_body = self.createMobileNetDetectionNetwork()
        self.nn_body.setConfidenceThreshold(0.7)
        self.nn_body.setBlobPath(body_path_model)
        self.nn_body.setNumInferenceThreads(2)
        self.nn_body.input.setBlocking(False)

        # FACE IN
        # Define a neural network that will make predictions based on the source frames
        self.nn_face = self.createMobileNetDetectionNetwork()
        self.nn_face.setBlobPath(face_path_model)

        # LINKS
        self._link_body_face()

    def _create_stress(self, stress_path_model: str):
        """Initializes neural network used for classifying stress.
        Parameters
        ----------
        stress_path_model : str
            Path to the stress blob model
        """
        # Define a neural network that will make predictions based on the source frames
        self.nn_stress = self.createNeuralNetwork()
        self.nn_stress.setBlobPath(stress_path_model)

        # LINKS
        self._link_stress()

    @abstractmethod
    def _create_in_stream(self):
        """Create input Pipeline stream."""
        pass

    @abstractmethod
    def _create_stress_in_stream(self):
        "Create stress input."
        pass

    @abstractmethod
    def _link_body_face(self):
        """Assigns input and output streams to body and face Neural Networks"""
        pass

    @abstractmethod
    def _link_stress(self):
        """Assigns input and output streams to stress Neural Networks"""
        pass

    @abstractmethod
    def _link_display(self):
        """Assigns input and output streams to frame display"""
        pass

    # ========= PUBLIC =========
    @abstractmethod
    def start(self):
        """Initialize and define Pipeline and I/O streams."""
        pass

    def get_display(self) -> np.ndarray:
        """Get frame to display.

        Returns
        -------
        any
            Frame used as input of the Pipeline.
            TODO: define type
        """
        result = self.display_out_q.tryGet()

        if result is not None:
            result = result.getFrame()

        return result

    def get_stress(self, frame) -> float:
        """Get output of stress.

        Returns
        -------
        float
            Stress represented by a number between 0 and 1.
        """

        img = process_frame(frame, 224, 224)
        self.stress_in_q.send(img)
        nn_data = self.stress_out_q.tryGet()
        numpy_nn_data = np.array(nn_data.getFirstLayerFp16())
        return numpy_nn_data[0]

    def get_face(self) -> dai.RawImgDetections:
        """Get face detection.

        Returns
        -------
        dai.RawImgDetections
            Face detection.
        """
        face_data = self.face_out_q.tryGet()

        detection = None

        if face_data is not None:
            detections = face_data.detections

            if len(detections) > 0:
                detection = detections[0]
                detection.label = 21

        # TODO: Convertir detection a un np.array o algo así más manejable,
        # no estoy seguro de como se hace
        return detection

    def get_body(self) -> np.array:
        """Get body detection.

        Returns
        -------
        dai.RawImgDetections
            Body detection.
        """
        body_data = self.body_out_q.tryGet()
        new_detections = []

        if body_data is not None:
            detections = body_data.detections

            for detection in detections:
                label = LIST_LABELS[detection.label]
                if label == "person":
                    new_detections.append(detection)

        if len(new_detections) > 0:
            d = new_detections[0]
            result = (d.xmax, d.xmin, d.ymax, d.ymin)
        else:
            result = None

        return result

    @abstractmethod
    def get(self) -> namedtuple:
        """Get all the results that output the entire Pipeline.

        Returns
        -------
        NamedTuple
            Named Tuple containing all the data processed by the device.
        """
        pass
