import threading

import typer

from oak.run_pipeline import run_pipeline
from oak.streaming.stream_cam import push_frame


def run_streaming(
    body_path_model: str = "models/mobilenet-ssd_openvino_2021.2_8shave.blob",
    face_path_model: str = "models/face-detection-openvino_2021.2_4shave.blob",
    stress_path_model: str = "models/mobilenet_stress_classifier_2021.2.blob",
    frequency: float = 5,
):
    """Runs the OAK pipeline, streaming from a video file or from the camera, if
    no video file is provided. The pipeline shows on screen the video on real time,
    marking the body and face of the baby, as well as the depth map.

    Parameters
    ----------
    path_model_body : str
        Path to body detection ".blob" model, by de1qfault
        "models/mobilenet-ssd_openvino_2021.2_8shave.blob"
    path_model_face : str
        Path to face detection ".blob" model, by default
        "models/face-detection-openvino_2021.2_4shave.blob"
    path_model_stress : Optional[str], optional
        Path to stress classification ".blob" model, by default
        "models/stress_classifier_2021.2.blob"
    frequency : float, optional
        Rate at which plots are updated and values are sent to the server, in seconds, by default 5
    """

    kwargs = {
        "body_path_model": body_path_model,
        "face_path_model": face_path_model,
        "stress_path_model": stress_path_model,
        "video_path": None,
        "frequency": frequency,
        "plot_results": False,
        "post_server": False,
    }

    threads = [
        threading.Thread(target=run_pipeline, args=(), kwargs=kwargs),
        threading.Thread(target=push_frame, args=()),
    ]
    [thread.setDaemon(True) for thread in threads]
    [thread.start() for thread in threads]


if __name__ == "__main__":
    typer.run(run_streaming)
