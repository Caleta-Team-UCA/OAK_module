from OAK import OAK
import depthai as dai
from typing import Optional
from collections import namedtuple
import cv2
from OAK.utils import process_frame
import typer

PipelineOut = namedtuple("PipelineOut", "display face_detection body_detection stress")


class OAKvideo(OAK):
    width: int = 300
    height: int = 300

    def __init__(
        self,
        path_model_body: Optional[
            str
        ] = "models/mobilenet-ssd_openvino_2021.2_8shave.blob",
        path_model_face: Optional[
            str
        ] = "models/face-detection-openvino_2021.2_4shave.blob",
        path_model_stress: Optional[str] = "models/stress_classifier_2021.2.blob",
    ):
        """Create and configure the parameters of the pipeline.

        Parameters
        ----------
        path_model_body : Optional[str], optional
            Path to body detection ".blob" model, by default "models/mobilenet-ssd_openvino_2021.2_8shave.blob"
        path_model_face : Optional[str], optional
            Path to face detection ".blob" model, by default "models/face-detection-openvino_2021.2_4shave.blob"
        path_model_stress : Optional[str], optional
            Path to stress classification ".blob" model, by default "models/stress_classifier_2021.2.blob"
        """
        super(OAKvideo, self).__init__(
            path_model_body, path_model_face, path_model_stress
        )
        self._create_in_stream(self)

    def _create_in_stream(self):
        """Create input Pipeline stream and stress input."""
        self.in_frame = self.createXLinkIn()
        self.in_frame.setStreamName("input")

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
        self.device.startPipeline()

        # Input queue
        self.in_q = self.getInputQueue(name="input")
        self.stress_in_q = self.getInputQueue(name="stress_input")

        # Output queues
        self.body_out_q = self.getOutputQueue(
            name="body_out", maxSize=4, blocking=False
        )
        self.face_out_q = self.getOutputQueue(
            name="face_out", maxSize=4, blocking=False
        )
        self.stress_out_q = self.getOutputQueue(
            name="stress_out", maxSize=4, blocking=False
        )
        self.display_out_q = self.getOutputQueue(
            name="display_out", maxSize=4, blocking=False
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
):
    video_processor = OAKvideo(path_model_body, path_model_face, None)
    video_processor.start(video_path)

    for result in video_processor.get():
        print(result)


if __name__ == "__main__":
    typer.run(main)
