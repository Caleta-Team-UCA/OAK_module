from time import sleep, time
from requests import post

import typer

from oak.utils.results import PlotSeries
from oak.pipeline.oak_cam import OAKCam
from oak.pipeline.oak_video_v2 import OAKVideo
from oak.process.activity import Activity
from oak.process.stress import Stress
from oak.process.breath import Breath, BreathConfig

# TODO: meter este diccionario en un config, para poder modificarlo más fácilmente
server_url = "vai.uca.es/event"
post_params = {
    "name": "Juan",
    "comments": "",
    "anomaly": False,
    "type": "",
    "value": 0,
    "babyid": 0,
}


def main(
    body_path_model: str = "models/mobilenet-ssd_openvino_2021.2_8shave.blob",
    face_path_model: str = "models/face-detection-openvino_2021.2_4shave.blob",
    stress_path_model: str = "models/mobilenet_stress_classifier_2021.2.blob",
    video_path: str = None,  # "videos_3_cams/22",
    frequency: float = 5,
    plot_results: bool = True,
    post_server: bool = False,
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
    frequency : float, optional
        Rate at which plots are updated and values are sent to the server, in seconds, by default 5
    plot_results : bool, optional
        Whether to plot results or not.
    post_server : bool, optional
        Wheter to send results to the server or not.
    """

    act = Activity()
    stre = Stress()
    breath = Breath(BreathConfig())

    if plot_results:
        plot_series = PlotSeries([stre, act, breath])

    if video_path is None:
        processor = OAKCam(body_path_model, face_path_model, stress_path_model)
        processor_parameters = {"show_results": plot_results}
    else:
        processor = OAKVideo(body_path_model, face_path_model, stress_path_model)
        processor_parameters = {"video_path": video_path, "show_results": plot_results}

    start_time = time()
    generator = processor.get(**processor_parameters)

    while True:
        roi_points = breath.get_breath_config().get_roi_points()
        next(generator)
        result = generator.send(
            [
                roi_points[0]["x"],
                roi_points[0]["y"],
                roi_points[1]["x"],
                roi_points[1]["y"],
            ]
        )

        # Process activity
        if result.body_detection is not None and result.face_detection is not None:
            act.update(result.body_detection, result.face_detection)

        # Process stress
        if result.stress is not None:
            stre.update(result.stress[0] == "stress")

        # Process breath
        if result.body_detection is not None and result.depth is not None:
            breath.update(
                result.face_detection, result.depth, result.calculator_results
            )

        if time() - start_time >= frequency:
            if plot_results:
                plot_series.update("movavg")

            if post_server:
                post_params["stress"] = stre.get_moving_average().tolist()
                post_params["act"] = act.get_moving_average().tolist()
                response = post(server_url, data=post_params)
                print(response)

            act.restart_series()
            stre.restart_series()
            breath.restart_series()
            start_time = time()

        # Aquí simulamos que se estuviera haciendo
        # algún tipo de procesamiento con los datos
        sleep(0.05)


if __name__ == "__main__":
    typer.run(main)
