import time
from collections import namedtuple

import cv2
import depthai as dai
import typer

from oak.OAK import OAKparent
from oak.utils import process_frame

PipelineOut = namedtuple("PipelineOut", "display face_detection body_detection stress")


class OAKvideo(OAKparent):
    width: int = 300
    height: int = 300
    video_path: str = ""

    def _create_in_stream(self):
        """Create input Pipeline stream and stress input."""
        self.in_frame = self.createXLinkIn()
        self.in_frame.setStreamName("input")

    def _create_stress_in_stream(self):
        """Create stress input."""
        self.stress_in_frame = self.createXLinkIn()
        self.stress_in_frame.setStreamName("stress_input")

    def _link_body_face(self):
        """Assigns input and output streams to body and face Neural Networks"""
        # Link inputs
        self.in_frame.out.link(self.nn_body.input)
        self.in_frame.out.link(self.nn_face.input)

        # Link outputs
        body_out_frame = self.createXLinkOut()
        body_out_frame.setStreamName("body_out")
        self.nn_body.out.link(body_out_frame.input)

        face_out_frame = self.createXLinkOut()
        face_out_frame.setStreamName("face_out")
        self.nn_face.out.link(face_out_frame.input)

    def _link_stress(self):
        """Assigns input and output streams to stress Neural Networks"""
        # Link input
        self.stress_in_frame.out.link(self.nn_stress.input)

        # Link output
        stress_out_frame = self.createXLinkOut()
        stress_out_frame.setStreamName("stress_out")
        self.nn_stress.out.link(stress_out_frame.input)

    def _link_display(self):
        """Assigns input and output streams to frame display"""
        display_out_frame = self.createXLinkOut()
        display_out_frame.setStreamName("display_out")
        self.in_frame.out.link(display_out_frame.input)

    def start(self, video_path: str):
        """Initialize and define Pipeline and I/O streams."""

        # Initialize device and pipeline
        self.device = dai.Device(self)

        # Input queue
        self.in_q = self.device.getInputQueue(name="input", maxSize=10, blocking=True)

        # Output queues
        self.body_out_q = self.device.getOutputQueue(
            name="body_out", maxSize=10, blocking=True
        )
        self.face_out_q = self.device.getOutputQueue(
            name="face_out", maxSize=10, blocking=True
        )
        self.display_out_q = self.device.getOutputQueue(
            name="display_out", maxSize=10, blocking=True
        )

        if self.stress_bool:
            self.stress_in_q = self.device.getInputQueue(name="stress_input")

            self.stress_out_q = self.device.getOutputQueue(
                name="stress_out", maxSize=10, blocking=True
            )

        self.video_path = video_path

    def get(self) -> namedtuple:
        """Get all the results that output the entire Pipeline.
        This function works as a generator, so it can be called several times.

        Returns
        -------
        NamedTuple
            Named Tuple containing all the data processed by the device.
        """

        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened():
            read_correctly, frame = cap.read()

            if not read_correctly:
                break

            img = process_frame(frame, self.width, self.height)
            self.in_q.send(img)

            display = display = self.get_display()
            face_detection = self.get_face()
            body_detection = self.get_body()

            if self.stress_bool:
                stress = self.get_stress(face_detection)
            else:
                stress = None

            yield PipelineOut(
                display=display,
                face_detection=face_detection,
                body_detection=body_detection,
                stress=stress,
            )


def main(
    path_model_body: str = "models/mobilenet-ssd_openvino_2021.2_8shave.blob",
    path_model_face: str = "models/face-detection-openvino_2021.2_4shave.blob",
    video_path: str = "videos/22-center-2.mp4",
    frame_rate: int = 20,
):
    video_processor = OAKvideo(path_model_body, path_model_face, None)
    video_processor.start(video_path)

    for i, result in enumerate(video_processor.get()):
        print(i, result)
        time.sleep(1 / frame_rate)

        if result.display is not None:
            # TOD: esto no muestra la imagen, arreglar
            cv2.namedWindow("display", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("display", 700, 600)
            cv2.imshow("display", result.display)


if __name__ == "__main__":
    typer.run(main)
