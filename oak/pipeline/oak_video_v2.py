from collections import namedtuple
from time import sleep
from typing import Iterable, Optional
from oak.utils.opencv import process_frame

import cv2
import depthai as dai
import numpy as np
import typer

from oak.pipeline.oak_parent import OAKParent
from oak.utils.opencv import frame_norm

PipelineOut = namedtuple(
    "PipelineOut",
    "display face_detection body_detection stress depth calculator_results",
)


class OAKVideo(OAKParent):
    width: int = 300
    height: int = 300

    # Steram names
    color_cam_name: str = "color_in"
    left_cam_name: str = "left_in"
    right_cam_name: str = "right_in"

    depth_output_name: str = "depth_out"
    calculator_output_name: str = "calculator_out"

    calculator_config_name: str = "calculator_config"

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
        super(OAKVideo, self).__init__(
            path_model_body, path_model_face, path_model_stress
        )

        self._create_depth_and_calculator(
            self.depth_output_name,
            self.calculator_output_name,
            self.calculator_config_name,
        )

    def _create_in_stream(self):
        """Create cams as in_stream."""
        # Color Camera vid
        self.color_cam = self.createXLinkIn()
        self.color_cam.setStreamName(self.color_cam_name)

        # Right and Left Cameras
        self.mono_left_cam = self.createXLinkIn()
        self.mono_left_cam.setStreamName(self.left_cam_name)
        self.mono_right_cam = self.createXLinkIn()
        self.mono_right_cam.setStreamName(self.right_cam_name)

    def _link_input(self, nn: dai.MobileNetDetectionNetwork):
        """Assigns an input stream to given Neural Network
        Parameters
        ----------
        nn : depthai.MobileNetDetectionNetwork
            Neural Network
        """
        self.color_cam.out.link(nn.input)

    def get(
        self,
        video_path: str,
        show_results: bool = False,
        new_config: Optional[Iterable[float]] = None,
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
        color_in_q = device.getInputQueue(
            name=self.color_cam_name, maxSize=4, blocking=False
        )
        left_in_q = device.getInputQueue(
            name=self.left_cam_name, maxSize=4, blocking=False
        )
        right_in_q = device.getInputQueue(
            name=self.right_cam_name, maxSize=4, blocking=False
        )

        calculator_config_q = device.getInputQueue(
            name=self.calculator_config_name, maxSize=4, blocking=False
        )

        # Output queues
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

        # Load videos
        cap_color = cv2.VideoCapture(f"{video_path}-center.mp4")
        cap_left = cv2.VideoCapture(f"{video_path}-left.mp4")
        cap_right = cv2.VideoCapture(f"{video_path}-right.mp4")

        while cap_color.isOpened():
            read_correctly_1, color_frame = cap_color.read()
            read_correctly_2, left_frame = cap_left.read()
            read_correctly_3, right_frame = cap_right.read()

            if not (read_correctly_1 and read_correctly_2 and read_correctly_3):
                break

            # Send images to pipeline
            img = process_frame(color_frame, self.width, self.height)
            color_in_q.send(img)

            img = process_frame(left_frame, 1280, 720)
            left_in_q.send(img)

            img = process_frame(right_frame, 1280, 720)
            right_in_q.send(img)

            if new_config is not None:
                self.breath_roi_corners = new_config
                config = dai.SpatialLocationCalculatorConfigData()
                top_left = dai.Point2f(new_config[0], new_config[1])
                bottom_right = dai.Point2f(new_config[2], new_config[3])
                config.roi = dai.Rect(top_left, bottom_right)
                cfg = dai.SpatialLocationCalculatorConfig()
                cfg.addROI(config)
                calculator_config_q.send(cfg)

            frame = color_frame
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
                self._show_results(
                    frame, depth_frame, body_bbox, face_bbox, stress, new_config
                )

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
    video_path: str = "videos_3_cams/22",
):
    video_processor = OAKVideo(body_path_model, face_path_model, stress_path_model)

    for i, result in enumerate(video_processor.get(video_path, True)):
        print(i, result.face_detection, result.body_detection, result.stress)
        # Aquí simulamos que se estuviera haciendo
        # algún tipo de procesamiento con los datos
        # sleep(0.05)


if __name__ == "__main__":
    typer.run(main)
