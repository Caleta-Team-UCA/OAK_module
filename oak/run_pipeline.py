from time import time
from requests import post

import typer
import cv2
import numpy as np

from oak.utils.results import PlotSeries
from oak.pipeline.oak_cam import OAKCam
from oak.pipeline.oak_video import OAKVideo
from oak.process.activity import Activity
from oak.process.stress import Stress
from oak.process.breath import Breath
from oak.utils.requests import ServerPost

RASPBERRY_RESOLUTION = (480, 640)


def main(
    body_path_model: str = "models/mobilenet-ssd_openvino_2021.2_8shave.blob",
    face_path_model: str = "models/face-detection-openvino_2021.2_4shave.blob",
    stress_path_model: str = "models/mobilenet_stress_classifier_2021.2.blob",
    video_path: str = None,  # "videos_3_cams/21",
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
    breath = Breath()

    win_name = "OAK results"

    if plot_results:
        plot_series = PlotSeries([stre, act, breath])

    if video_path is None:
        processor = OAKCam(body_path_model, face_path_model, stress_path_model)
        processor_parameters = {"show_results": plot_results}
    else:
        processor = OAKVideo(body_path_model, face_path_model, stress_path_model)
        processor_parameters = {"video_path": video_path, "show_results": plot_results}

    if post_server:
        post_server = ServerPost()

    start_time = time()
    generator = processor.get(**processor_parameters)

    plot_img = plot_series.update("movavg")
    while True:
        next(generator)
        result, pipeline_image = generator.send(breath.get_roi_corners)

        # Process activity
        if result.body_detection is not None and result.face_detection is not None:
            act.update(result.body_detection, result.face_detection)

        # Process stress
        if result.stress is not None:
            stre.update(result.stress[0] == "stress")

        # Process breath
        if result.body_detection is not None:
            breath.update(
                result.face_detection,
                processor.rgb_resolution,
                result.calculator_results,
            )

        if time() - start_time >= frequency:
            if plot_results:
                plot_img = plot_series.update("movavg")

            if post_server:
                post_server.save(
                    ServerPost.TYPE_ACTIVITY,
                    act.score.mean(),
                    "1",
                    "F9qkMQ1151Xn7k7Q5CR3",
                )

                post_server.save(
                    ServerPost.TYPE_RESPIRATION,
                    breath.score.mean(),
                    "1",
                    "F9qkMQ1151Xn7k7Q5CR3",
                )

                post_server.save(
                    ServerPost.TYPE_STRESS,
                    stre.score.mean(),
                    "1",
                    "F9qkMQ1151Xn7k7Q5CR3",
                )

            act.restart_series()
            stre.restart_series()
            breath.restart_series()
            start_time = time()

        # Aquí simulamos que se estuviera haciendo
        # algún tipo de procesamiento con los datos
        # sleep(0.05)

        if plot_results:
            # print(1, plot_img.shape, pipeline_image.shape)

            pipeline_image = cv2.resize(
                pipeline_image, (RASPBERRY_RESOLUTION[0], RASPBERRY_RESOLUTION[0])
            )

            res_dif = RASPBERRY_RESOLUTION[1] - RASPBERRY_RESOLUTION[0]
            scale_proportion = res_dif / plot_img.shape[1]
            plot_img = cv2.resize(
                plot_img,
                (
                    res_dif,
                    int(plot_img.shape[0] * scale_proportion),
                ),
            )

            # print(2, plot_img.shape, pipeline_image.shape)

            new_image = (
                np.ones((RASPBERRY_RESOLUTION[0], RASPBERRY_RESOLUTION[1], 3), np.uint8)
                * 255
            )

            new_image[0:, 0 : pipeline_image.shape[1], :3] = pipeline_image
            new_image[
                0 : plot_img.shape[0],
                pipeline_image.shape[1] : pipeline_image.shape[1] + plot_img.shape[1],
                :3,
            ] = plot_img

            cv2.imshow(win_name, new_image)

            if cv2.waitKey(1) == ord("q"):
                break


if __name__ == "__main__":
    typer.run(main)
