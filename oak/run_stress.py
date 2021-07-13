import typer

from oak.pipeline.oak_cam import OAKCam
from oak.pipeline.oak_video import OAKVideo
from oak.process.stress import Stress


def main(
    body_path_model: str = "models/mobilenet-ssd_openvino_2021.2_8shave.blob",
    face_path_model: str = "models/face-detection-openvino_2021.2_4shave.blob",
    stress_path_model: str = "models/mobilenet_stress_classifier_2021.2.blob",
    video_path: str = None,#"videos/22-center-2.mp4",
):
    """Runs the OAK pipeline, streaming from a video file or from the camera, if
    no video file is provided. The pipeline shows on screen the video on real time,
    marking the body and face of the baby, as well as the depth map.

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
    video_path : str, optional
        Path to the video file. If None provided, uses the cam. By default None
    """
    stress = Stress()
    if video_path is None:
        processor = OAKCam(body_path_model, face_path_model, stress_path_model)
        for i, result in enumerate(processor.get(True)):
            stress_score = result.stress
            if stress_score is not None:
                stress.update(stress_score[0] == "stress")
    else:
        processor = OAKVideo(body_path_model, face_path_model, stress_path_model)
        for i, result in enumerate(processor.get(video_path, True)):
            stress_score = result.stress
            if stress_score is not None:
                stress.update(stress_score[0] == "stress")


if __name__ == "__main__":
    typer.run(main)