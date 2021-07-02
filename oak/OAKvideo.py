from oak.OAK import OAKparent
import depthai as dai
from collections import namedtuple
import cv2
from oak.utils import process_frame, frame_norm
import typer
from time import sleep

PipelineOut = namedtuple("PipelineOut", "display face_detection body_detection stress")


class OAKvideo(OAKparent):
    width: int = 300
    height: int = 300

    def _create_in_stream(self):
        """Create input Pipeline stream."""
        self.in_frame = self.createXLinkIn()
        self.in_frame.setStreamName(self.input_name)

    def _link_input(self, nn: dai.MobileNetDetectionNetwork):
        """Assigns an input stream to given Neural Network
        Parameters
        ----------
        nn : depthai.MobileNetDetectionNetwork
            Neural Network
        """
        self.in_frame.out.link(nn.input)

    def get(self, video_path, show_results: bool = False) -> namedtuple:
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
        in_q = device.getInputQueue(name="input", maxSize=1, blocking=False)

        # Output queues
        body_out_q = device.getOutputQueue(name="body_out", maxSize=1, blocking=True)
        face_out_q = device.getOutputQueue(name="face_out", maxSize=1, blocking=True)

        if self.stress_bool:
            stress_in_q = device.getInputQueue(
                name="stress_input", maxSize=1, blocking=False
            )

            stress_out_q = device.getOutputQueue(
                name="stress_out", maxSize=1, blocking=True
            )

        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            read_correctly, frame = cap.read()

            if not read_correctly:
                break

            img = process_frame(frame, self.width, self.height)
            in_q.send(img)

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

                cv2.imshow("", show_frame)

                if cv2.waitKey(1) == ord("q"):
                    break

            yield PipelineOut(
                display=frame,
                face_detection=face_bbox,
                body_detection=body_bbox,
                stress=stress,
            )


def main(
    body_path_model: str = "models/mobilenet-ssd_openvino_2021.2_8shave.blob",
    face_path_model: str = "models/face-detection-openvino_2021.2_4shave.blob",
    stress_path_model: str = "models/mobilenet_stress_classifier_2021.2.blob",
    video_path: str = "videos/22-center-2.mp4",
):
    video_processor = OAKvideo(body_path_model, face_path_model, stress_path_model)

    for i, result in enumerate(video_processor.get(video_path, True)):
        print(i, result.face_detection, result.body_detection, result.stress)
        # Aquí simulamos que se estuviera haciendo
        # algún tipo de procesamiento con los datos
        sleep(0.05)


if __name__ == "__main__":
    typer.run(main)
