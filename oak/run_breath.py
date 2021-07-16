import typer

from oak.pipeline.oak_cam import OAKCam
from oak.process.breath import Breath, BreathConfig


def main(
    body_path_model: str = "models/mobilenet-ssd_openvino_2021.2_8shave.blob",
    face_path_model: str = "models/face-detection-openvino_2021.2_4shave.blob",
    stress_path_model: str = "models/mobilenet_stress_classifier_2021.2.blob",
    video_path: str = None,
):
    breath = Breath(BreathConfig())
    if video_path is None:
        processor = OAKCam(body_path_model, face_path_model, stress_path_model)
        for i, result in enumerate(processor.get(
            True,
            [ breath.get_breath_config().topLeft, breath.get_breath_config().bottomRight ],
            [ breath.get_breath_config().xmin, breath.get_breath_config().xmax, breath.get_breath_config().ymin, breath.get_breath_config().ymax ]
        )):
            breath.update(
                result.face_detection, result.depth, result.calculator_results
            )
    else:
        raise NameError("InvalidOption")


if __name__ == "__main__":
    typer.run(main)
