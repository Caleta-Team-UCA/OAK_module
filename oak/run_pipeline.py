from multiprocessing import Process, Queue
from time import time
from numpy.lib.npyio import save
import requests
import typer
import cv2
import json
import numpy as np
import subprocess as sp

from oak.utils.results import PlotSeries
from oak.pipeline.oak_cam import OAKCam
from oak.pipeline.oak_video import OAKVideo
from oak.process.activity import Activity
from oak.process.stress import Stress
from oak.process.breath import Breath
from oak.utils.requests import ServerPost


def show_cv2_window(
    win_name,
    raspberry_resolution,
    pipeline_img_queue,
    plot_img_queue,
    instruction_queue,
    bpm_queue,
):
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    bpm = 0
    while True:
        pipeline_img = pipeline_img_queue.get()
        plot_img = plot_img_queue.get()

        pipeline_img = cv2.resize(
            pipeline_img, (raspberry_resolution[0], raspberry_resolution[0])
        )

        res_dif = raspberry_resolution[1] - raspberry_resolution[0]
        scale_proportion = res_dif / plot_img.shape[1]
        plot_img = cv2.resize(
            plot_img,
            (
                res_dif,
                int(plot_img.shape[0] * scale_proportion),
            ),
        )

        new_image = (
            np.ones((raspberry_resolution[0], raspberry_resolution[1], 3), np.uint8)
            * 255
        )

        new_image[0:, 0 : pipeline_img.shape[1], :3] = pipeline_img
        x1_plot = 0
        x2_plot = plot_img.shape[0]
        y1_plot = pipeline_img.shape[1]
        y2_plot = pipeline_img.shape[1] + plot_img.shape[1]
        new_image[
            x1_plot:x2_plot,
            y1_plot:y2_plot,
            :3,
        ] = plot_img

        if not bpm_queue.empty():
            bpm = bpm_queue.get()

        new_image = cv2.putText(
            new_image,
            f"{int(bpm)} bpm",
            (
                y1_plot + int((y2_plot - y1_plot) / 8),
                x2_plot + int((x2_plot - x1_plot) / 8),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )

        cv2.imshow(win_name, new_image)
        key = cv2.waitKey(1)
        instruction_queue.put(key)


rtmp_url = "rtsp://vai.uca.es:1935/mystream"
command = [
    "ffmpeg",
    "-y",
    "-f",
    "rawvideo",
    "-vcodec",
    "rawvideo",
    "-pix_fmt",
    "bgr24",
    "-s",
    "{}x{}".format(300, 300),
    "-i",
    "-",
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-preset",
    "ultrafast",
    "-f",
    "rtsp",
    rtmp_url,
]


def push_frame(frame_queue):
    p = None

    while True:
        if len(command) > 0:
            p = sp.Popen(command, stdin=sp.PIPE)
            break

    while True:
        if frame_queue.empty() != True:
            frame = frame_queue.get()
            p.stdin.write(frame.tostring())


RASPBERRY_RESOLUTION = (1080, 1920)


def run_pipeline(
    body_path_model: str = "models/mobilenet-ssd_openvino_2021.2_8shave.blob",
    face_path_model: str = "models/face-detection-openvino_2021.2_4shave.blob",
    stress_path_model: str = "models/mobilenet_stress_classifier_2021.2.blob",
    video_path: str = "demo/sample",
    frequency: float = 20,
    plot_results: bool = True,
    post_server: bool = True,
    stream: bool = True,
    save_results: bool = True,
    server_url: str = "http://vai.uca.es",
    server_port: str = "5000",
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

    if stream:
        streaming_queue = Queue()
        stream_process = Process(target=push_frame, args=(streaming_queue,))
        stream_process.start()

    if plot_results:
        plot_series = PlotSeries([stre, act, breath])

        win_name = "OAK results"
        pipeline_img_queue = Queue()
        plot_img_queue = Queue()
        instruction_queue = Queue()
        bpm_queue = Queue()
        show_process = Process(
            target=show_cv2_window,
            args=(
                win_name,
                RASPBERRY_RESOLUTION,
                pipeline_img_queue,
                plot_img_queue,
                instruction_queue,
                bpm_queue,
            ),
        )
        show_process.start()

    if video_path is None:
        processor = OAKCam(body_path_model, face_path_model, stress_path_model)
        processor_parameters = {"show_results": plot_results}
    else:
        processor = OAKVideo(body_path_model, face_path_model, stress_path_model)
        processor_parameters = {"video_path": video_path, "show_results": plot_results}

    if post_server:
        post_server = ServerPost(server_url, server_port)

    if save_results:
        post_list = []
        video_images = []

    start_time = time()
    generator = processor.get(**processor_parameters)

    plot_img = plot_series.update("movavg")
    while True:
        try:
            next(generator)
            result, pipeline_image = generator.send(breath.get_roi_corners)
        except StopIteration:
            break

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

        delay = time() - start_time
        if delay >= frequency:
            if plot_results:
                plot_img = plot_series.update("movavg")
                bpm_queue.put(breath.get_bpm(delay))

            if post_server:
                activity_data = post_server.save(
                    ServerPost.TYPE_ACTIVITY,
                    act.dict_scores,
                    "1",
                    "F9qkMQ1151Xn7k7Q5CR3",
                )

                respiration_data = post_server.save(
                    ServerPost.TYPE_RESPIRATION,
                    breath.get_bpm(delay),
                    "1",
                    "F9qkMQ1151Xn7k7Q5CR3",
                )

                stress_data = post_server.save(
                    ServerPost.TYPE_STRESS,
                    stre.score.mean() * 100,
                    "1",
                    "F9qkMQ1151Xn7k7Q5CR3",
                )

                if save_results:
                    post_list.append(activity_data)
                    post_list.append(respiration_data)
                    post_list.append(stress_data)

            act.restart_series()
            stre.restart_series()
            breath.restart_series()
            start_time = time()

        # Aqu?? simulamos que se estuviera haciendo
        # alg??n tipo de procesamiento con los datos
        # sleep(0.05)

        if stream:
            streaming_queue.put(result.display)
            video_images.append(result.display)

        if plot_results:
            pipeline_img_queue.put(pipeline_image)
            plot_img_queue.put(plot_img)

            step_size = 0.01

            key = instruction_queue.get()

            if key == ord("q"):
                break
            elif key == ord("w"):
                breath.dy -= step_size
            elif key == ord("a"):
                breath.dx -= step_size
            elif key == ord("s"):
                breath.dy += step_size
            elif key == ord("d"):
                breath.dx += step_size
            elif key == ord("n") and breath.width_roi - 1 > 0:
                breath.width_roi -= 1
            elif key == ord("m"):
                breath.width_roi += 1
            elif key == ord("c"):
                plot_series.clear()

    stream_process.terminate()
    show_process.terminate()

    if save_results:
        with open("results/requests.json", "w") as f:
            f.write(json.dumps(post_list))

        video_writer = cv2.VideoWriter(
            "results/video_processed.avi",
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            15,
            (video_images[0].shape[1], video_images[0].shape[0]),
        )
        for img in video_images:
            video_writer.write(img)

        video_writer.release()


if __name__ == "__main__":
    typer.run(run_pipeline)
