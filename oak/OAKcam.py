from collections import namedtuple
from time import sleep
from typing import Iterable, Optional

import cv2
import depthai as dai
import numpy as np
import typer

from oak.OAK import OAKparent
from oak.utils.opencv import frame_norm

PipelineOut = namedtuple(
    "PipelineOut",
    "display face_detection body_detection stress depth calculator_results",
)


class OAKcam(OAKparent):
    width: int = 300
    height: int = 300

    cam_preview_name = "color_cam_prev"
    depth_output_name = "depth_out"
    calculator_output_name = "calculator_out"

    calculator_config_name = "calculator_config"

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
        super(OAKcam, self).__init__(
            path_model_body, path_model_face, path_model_stress
        )

        self._create_depth_and_calculator(
            self.depth_output_name,
            self.calculator_output_name,
            self.calculator_config_name,
        )

    def _create_in_stream(self):
        """Create cams as in_stream."""
        # Color Camera
        self.color_cam = self.createColorCamera()
        self.color_cam.setPreviewSize(self.width, self.height)
        self.color_cam.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1080_P
        )
        self.color_cam.setInterleaved(False)
        self.color_cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        # Define output stream
        cam_xout = self.createXLinkOut()
        cam_xout.setStreamName(self.cam_preview_name)
        self.color_cam.preview.link(cam_xout.input)

        # Right and Left Cameras
        self.mono_left_cam = self.createMonoCamera()
        self.mono_right_cam = self.createMonoCamera()

        self.mono_left_cam.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_400_P
        )
        self.mono_left_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
        self.mono_right_cam.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_400_P
        )
        self.mono_right_cam.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    def _link_input(self, nn: dai.MobileNetDetectionNetwork):
        """Assigns an input stream to given Neural Network
        Parameters
        ----------
        nn : depthai.MobileNetDetectionNetwork
            Neural Network
        """
        self.color_cam.preview.link(nn.input)

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
        top_left = dai.Point2f(0.4, 0.4)
        bottom_right = dai.Point2f(0.42, 0.42)

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

    def _get_cam_preview(self, cam_out_q: dai.DataOutputQueue):
        frame = (
            np.array(cam_out_q.get().getData())
            .reshape((3, self.height, self.width))
            .transpose(1, 2, 0)
            .astype(np.uint8)
        )

        return frame

    @staticmethod
    def _get_depth(depth_out_q: dai.DataOutputQueue):
        depth_frame = depth_out_q.get().getFrame()

        depth_frame = cv2.normalize(
            depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1
        )
        depth_frame = cv2.equalizeHist(depth_frame)
        depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_HOT)

        return depth_frame

    @staticmethod
    def _get_calculator(calculator_out_q: dai.DataOutputQueue):
        results = calculator_out_q.tryGet()

        if results is not None:
            results = results.getSpatialLocations()

        return results

    def get(
        self,
        show_results: bool = False,
        new_config: Optional[Iterable[dai.Point2f]] = None,
    ) -> namedtuple:
        """Get all the results that output the entire Pipeline.
        This function works as a generator, so it can be called several times.

        Returns
        -------
        NamedTuple
            Named Tuple containing all the data processed by the device.
            `display`: input frame.
            `face_detection`: bounding box corners of the face.
            `body_detection`: bounding box corners of the body.
            `stress`: stress value and label
        """

        # Initialize device and pipeline
        device = dai.Device(self)

        # Input queue
        calculator_config_q = device.getInputQueue(
            name=self.calculator_config_name, maxSize=4, blocking=False
        )

        # Output queues
        cam_out_q = device.getOutputQueue(
            name=self.cam_preview_name, maxSize=1, blocking=False
        )

        body_out_q = device.getOutputQueue(
            name=self.body_output_name, maxSize=1, blocking=False
        )
        face_out_q = device.getOutputQueue(
            name=self.face_output_name, maxSize=1, blocking=False
        )

        if self.stress_bool:
            stress_in_q = device.getInputQueue(
                name=self.stress_input_name, maxSize=1, blocking=False
            )

            stress_out_q = device.getOutputQueue(
                name=self.stress_output_name, maxSize=1, blocking=False
            )

        depth_out_q = device.getOutputQueue(
            name=self.depth_output_name, maxSize=1, blocking=False
        )

        calculator_out_q = device.getOutputQueue(
            name=self.calculator_output_name, maxSize=1, blocking=False
        )

        while True:
            if new_config is not None:
                config = dai.SpatialLocationCalculatorConfigData()
                config.roi = dai.Rect(new_config[0], new_config[1])
                cfg = dai.SpatialLocationCalculatorConfig()
                cfg.addROI(config)
                calculator_config_q.send(cfg)

            frame = self._get_cam_preview(cam_out_q)
            depth_frame = self._get_depth(depth_out_q)

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

            # Code for showing results in CV2
            if show_results:
                show_frame = frame.copy()

                if body_bbox is not None:
                    show_frame = cv2.rectangle(
                        show_frame,
                        (body_bbox[0], body_bbox[1]),
                        (body_bbox[2], body_bbox[3]),
                        (255, 0, 0),
                        2,
                    )

                if face_bbox is not None:
                    show_frame = cv2.rectangle(
                        show_frame,
                        (face_bbox[0], face_bbox[1]),
                        (face_bbox[2], face_bbox[3]),
                        (36, 255, 12),
                        2,
                    )

                    if stress is not None:
                        show_frame = cv2.putText(
                            show_frame,
                            f"{stress[0]} {stress[1]:.2}",
                            (face_bbox[0], face_bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (36, 255, 12),
                            2,
                        )

                cv2.imshow("rgb", show_frame)
                cv2.imshow("depth", depth_frame)

                if cv2.waitKey(1) == ord("q"):
                    break

            yield PipelineOut(
                display=frame,
                face_detection=face_bbox,
                body_detection=body_bbox,
                stress=stress,
                depth=depth_frame,
                calculator_results=calculator_results,
            )


def main(
    body_path_model: str = "models/mobilenet-ssd_openvino_2021.2_8shave.blob",
    face_path_model: str = "models/face-detection-openvino_2021.2_4shave.blob",
    stress_path_model: str = "models/mobilenet_stress_classifier_2021.2.blob",
):
    cam_processor = OAKcam(body_path_model, face_path_model, stress_path_model)

    for i, result in enumerate(cam_processor.get(True)):
        print(
            i,
            result.face_detection,
            result.body_detection,
            result.stress,
            result.calculator_results,
        )
        # Aquí simulamos que se estuviera haciendo
        # algún tipo de procesamiento con los datos
        sleep(0.1)


if __name__ == "__main__":
    typer.run(main)
