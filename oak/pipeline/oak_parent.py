from abc import abstractmethod
from collections import namedtuple
from typing import Any, List, Optional, Tuple, Union

import cv2
import depthai as dai
import numpy as np
from oak.utils.opencv import frame_norm, to_planar
from oak.utils.params import LIST_LABELS

PipelineOut = namedtuple(
    "PipelineOut",
    "display face_detection body_detection stress depth calculator_results roi_breath",
)


class OAKParent(dai.Pipeline):
    # Breath roi corners
    breath_roi_corners: Tuple[float] = (0.5, 0.3, 0.52, 0.32)

    input_name: str = "input"
    stress_input_name: str = "stress_input"

    face_output_name: str = "face_out"
    body_output_name: str = "body_out"
    stress_output_name: str = "stress_out"

    stress_bool: bool = False

    depth_resolution: Optional[Tuple[int]] = None

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
        super(OAKParent, self).__init__()
        self.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_1)

        self._create_in_stream()
        self._create_detection_network(path_model_body, self.body_output_name)
        self._create_detection_network(path_model_face, self.face_output_name)

        if path_model_stress is not None:
            self._create_stress(path_model_stress, self.stress_output_name)
            self.stress_bool = True

    # ========= PRIVATE =========
    def _create_detection_network(self, model_path: str, name: str):
        """Initializes neural networks used for detecting entities from frames.

        Parameters
        ----------
        model_path : str
            Path to the body blob model
        """
        # Define a neural network that will make predictions based on the source frames
        nn = self.createMobileNetDetectionNetwork()
        nn.setConfidenceThreshold(0.5)
        nn.setBlobPath(model_path)
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)

        self._link_input(nn)
        self._link_output(nn, name)

    def _create_stress(self, stress_path_model: str, name: str):
        """Initializes neural network used for classifying stress.
        Parameters
        ----------
        stress_path_model : str
            Path to the stress blob model
        """
        # Define a neural network that will make predictions based on the source frames
        nn_stress = self.createNeuralNetwork()
        nn_stress.setBlobPath(stress_path_model)
        nn_stress.setNumInferenceThreads(2)
        nn_stress.input.setBlocking(False)
        nn_stress.input.setQueueSize(1)

        # LINKS
        self._create_and_link_stress_input(nn_stress)
        self._link_output(nn_stress, name)

    def _create_and_link_stress_input(self, nn_stress: dai.NeuralNetwork):
        """Assigns input and output streams to stress Neural Networks"""
        stress_in_frame = self.createXLinkIn()
        stress_in_frame.setStreamName(self.stress_input_name)
        stress_in_frame.out.link(nn_stress.input)

    def _link_output(self, nn: dai.MobileNetDetectionNetwork, name: str):
        """Assigns an output stream to given Neural Network
        Parameters
        ----------
        nn : depthai.MobileNetDetectionNetwork
            Neural Network
        name : str
            Label of the output stream
        """
        nn_out = self.createXLinkOut()
        nn_out.setStreamName(name)
        nn.out.link(nn_out.input)

    @abstractmethod
    def _create_in_stream(self):
        """Create input Pipeline stream."""
        pass

    @abstractmethod
    def _link_input(self, nn: dai.MobileNetDetectionNetwork):
        """Assigns an input stream to given Neural Network
        Parameters
        ----------
        nn : depthai.MobileNetDetectionNetwork
            Neural Network
        """
        pass

    @staticmethod
    def _get_stress(
        face_frame: np.ndarray,
        stress_in_q: dai.DataInputQueue,
        stress_out_q: dai.DataOutputQueue,
    ) -> Optional[Tuple[Union[str, List[str]], Any]]:
        """Get output of stress.

        Returns
        -------
        float
            Stress represented by a number between 0 and 1.
        """

        nn_data = dai.NNData()
        nn_data.setLayer("prob", to_planar(face_frame, (224, 224)))

        stress_in_q.send(nn_data)

        stress_nn_out = stress_out_q.tryGet()
        if stress_nn_out:
            stress_labels = ["non_stress", "stress"]
            stress_value = np.array(stress_nn_out.getFirstLayerFp16())[0]
            label = stress_labels[stress_value > 0.5]

            return label, stress_value
        else:
            return None

    @staticmethod
    def _get_face(
        face_out_q: dai.DataOutputQueue,
    ) -> Optional[Tuple[Any, Any, Any, Any]]:
        """Get face detection.

        Returns
        -------
        dai.RawImgDetections
            Face detection.
        """
        bbox = None

        face_detections = face_out_q.tryGet()
        if face_detections:
            if len(face_detections.detections) > 0:
                d = face_detections.detections[0]

                bbox = (d.xmin, d.ymin, d.xmax, d.ymax)

        return bbox

    @staticmethod
    def _get_body(
        body_out_q: dai.DataOutputQueue,
    ) -> Optional[Tuple[Any, Any, Any, Any]]:
        """Get body detection.

        Returns
        -------
        dai.RawImgDetections
            Body detection.
        """
        bbox = None

        body_detections = body_out_q.tryGet()
        if body_detections:
            for d in body_detections.detections:
                label = LIST_LABELS[d.label]
                if label == "person":
                    bbox = (d.xmin, d.ymin, d.xmax, d.ymax)
                    break

        return bbox

    @staticmethod
    def _get_calculator(calculator_out_q: dai.DataOutputQueue):
        results = calculator_out_q.tryGet()

        if results is not None:
            results = results.getSpatialLocations()

        return results

    @staticmethod
    def _get_depth(depth_out_q: dai.DataOutputQueue) -> np.array:
        """Returns the depth image from the output depth node

        Parameters
        ----------
        depth_out_q : dai.DataOutputQueue
            Depth output node

        Returns
        -------
        np.array
            Depth image
        """
        depth_frame = depth_out_q.get().getFrame()

        depth_frame = cv2.normalize(
            depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1
        )
        depth_frame = cv2.equalizeHist(depth_frame)
        depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_HOT)

        return depth_frame

    def _create_depth_and_calculator(
        self, depth_name: str, calculator_name: str, calculator_config_name: str
    ):
        # DEPTH NODE
        depth = self.createStereoDepth()
        depth.setOutputDepth(True)

        # depth.setConfidenceThreshold(200)
        # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        depth.setMedianFilter(dai.StereoDepthProperties.MedianFilter.KERNEL_5x5)
        depth.setLeftRightCheck(False)
        depth.setExtendedDisparity(False)
        depth.setSubpixel(False)
        if self.depth_resolution is not None:
            depth.setInputResolution(self.depth_resolution[0], self.depth_resolution[1])

        # Link input to calculate depth
        self.mono_left_cam.out.link(depth.left)
        self.mono_right_cam.out.link(depth.right)

        # Link output
        depth_out = self.createXLinkOut()
        depth_out.setStreamName(depth_name)

        depth.depth.link(depth_out.input)

        # CALCULATOR
        calculator = self.createSpatialLocationCalculator()
        calculator.setWaitForConfigInput(False)
        # We need to be accurate, so we use a very small ROI
        top_left = dai.Point2f(self.breath_roi_corners[0], self.breath_roi_corners[1])
        bottom_right = dai.Point2f(
            self.breath_roi_corners[2], self.breath_roi_corners[3]
        )

        calculator.setWaitForConfigInput(False)
        config = dai.SpatialLocationCalculatorConfigData()

        # We measure depth in a very small range
        config.depthThresholds.lowerThreshold = 600
        config.depthThresholds.upperThreshold = 900

        config.roi = dai.Rect(top_left, bottom_right)
        calculator.initialConfig.addROI(config)

        # Link Inputs
        depth.depth.link(calculator.inputDepth)

        calc_config = self.createXLinkIn()
        calc_config.setStreamName(calculator_config_name)
        calc_config.out.link(calculator.inputConfig)

        # Link Output
        calc_out = self.createXLinkOut()
        calc_out.setStreamName(calculator_name)
        calculator.out.link(calc_out.input)

    def _show_results(
        self, frame, depth_frame, body_bbox, face_bbox, stress, breath_roi
    ):
        show_frame = frame.copy()

        if body_bbox is not None:
            show_frame = cv2.rectangle(
                show_frame,
                (body_bbox[0], body_bbox[1]),
                (body_bbox[2], body_bbox[3]),
                (255, 0, 0),
                2,
            )

            show_frame = cv2.putText(
                show_frame,
                f"body",
                (body_bbox[0], body_bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

        if face_bbox is not None and stress is not None:
            verde = (36, 255, 12)
            rojo = (36, 12, 255)

            if stress[0] == "stress":
                color = rojo
            else:
                color = verde

            show_frame = cv2.rectangle(
                show_frame,
                (face_bbox[0], face_bbox[1]),
                (face_bbox[2], face_bbox[3]),
                color,
                2,
            )
            show_frame = cv2.putText(
                show_frame,
                f"{stress[0]}",
                (face_bbox[0], face_bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )

            roi_bbox = frame_norm(frame, self.breath_roi_corners)

            show_frame = cv2.rectangle(
                show_frame,
                (roi_bbox[0], roi_bbox[1]),
                (roi_bbox[2], roi_bbox[3]),
                (0, 166, 255),
                2,
            )

            show_frame = cv2.putText(
                show_frame,
                f"breath",
                (roi_bbox[0], roi_bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 166, 255),
                2,
            )

        # cv2.imshow("rgb", show_frame)
        # cv2.imshow("depth", depth_frame)
        return show_frame

    def get_process_streams(
        self,
        frame,
        depth_frame,
        face_out_q,
        stress_in_q,
        stress_out_q,
        body_out_q,
        calculator_config_q,
        calculator_out_q,
    ) -> namedtuple:
        self.rgb_resolution = frame.shape

        if self.breath_roi_corners is not None:
            config = dai.SpatialLocationCalculatorConfigData()
            top_left = dai.Point2f(
                self.breath_roi_corners[0], self.breath_roi_corners[1]
            )
            bottom_right = dai.Point2f(
                self.breath_roi_corners[2], self.breath_roi_corners[3]
            )
            config.roi = dai.Rect(top_left, bottom_right)
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            calculator_config_q.send(cfg)

        face_bbox = self._get_face(face_out_q)
        if face_bbox is not None:
            face_bbox = frame_norm(frame, face_bbox)

        body_bbox = self._get_body(body_out_q)
        if body_bbox is not None:
            body_bbox = frame_norm(frame, body_bbox)

        if face_bbox is not None and self.stress_bool:
            face = frame[
                face_bbox[1] : face_bbox[3],
                face_bbox[0] : face_bbox[2],
                :,
            ]
            stress = self._get_stress(face, stress_in_q, stress_out_q)
        else:
            stress = None

        calculator_results = self._get_calculator(calculator_out_q)

        return PipelineOut(
            display=frame,
            face_detection=face_bbox,
            body_detection=body_bbox,
            stress=stress,
            depth=depth_frame,
            calculator_results=calculator_results,
            roi_breath=self.breath_roi_corners,
        )

    # ========= PUBLIC =========

    @abstractmethod
    def get(self, *args, **kwargs) -> namedtuple:
        """Get all the results that output the entire Pipeline.

        Returns
        -------
        NamedTuple
            Named Tuple containing all the data processed by the device.
        """
        pass
