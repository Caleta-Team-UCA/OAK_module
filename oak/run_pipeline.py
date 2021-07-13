from time import sleep

import typer

from oak.utils.series import PlotSeries
from oak.pipeline.oak_cam import OAKCam
from oak.pipeline.oak_video import OAKVideo
from oak.process.activity import Activity
from oak.process.stress import Stress


def main(
    body_path_model: str = "models/mobilenet-ssd_openvino_2021.2_8shave.blob",
    face_path_model: str = "models/face-detection-openvino_2021.2_4shave.blob",
    stress_path_model: str = "models/mobilenet_stress_classifier_2021.2.blob",
    video_path: str = "videos/22-center-2.mp4",
    frequency: int = 12,
    size: int = 240,
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
    size : int, optional
        Length of the stored series, by default 1000
    frequency : int, optional
        Rate at which plots are updated, in frames, by default 24
    """
    act = Activity(size=size, frequency=frequency)
    stre = Stress(size=size, frequency=frequency)

    plot_series = PlotSeries(
        [stre.ser_score, act.ser_score],
        xlim=(0, size),
        ylim=(-0.15, 1.15),
    )

    if video_path is None:
        processor = OAKCam(body_path_model, face_path_model, stress_path_model)

        for i, result in enumerate(processor.get(True)):
            print(
                i,
                result.face_detection,
                result.body_detection,
                result.stress,
                result.calculator_results,
            )

            if result.body_detection is not None and result.face_detection is not None:
                act.update(result.body_detection, result.face_detection)

            if result.stress is not None:
                stre.update(result.stress[0] == "stress")

            if i % frequency == 0:
                plot_series.update("movavg")

            # Aquí simulamos que se estuviera haciendo
            # algún tipo de procesamiento con los datos
            sleep(0.1)
    else:
        processor = OAKVideo(body_path_model, face_path_model, stress_path_model)
        for i, result in enumerate(processor.get(video_path, True)):
            print(i, result.face_detection, result.body_detection, result.stress)

            if result.body_detection is not None and result.face_detection is not None:
                act.update(result.body_detection, result.face_detection)

            if result.stress is not None:
                stre.update(result.stress[0] == "stress")

            if i % frequency == 0:
                plot_series.update("movavg")

            # Aquí simulamos que se estuviera haciendo
            # algún tipo de procesamiento con los datos
            sleep(0.05)


if __name__ == "__main__":
    typer.run(main)
