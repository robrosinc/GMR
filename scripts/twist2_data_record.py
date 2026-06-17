#!/usr/bin/env python3

"""
TWIST2-compatible Redis data recorder.

Records the same state/action Redis contract used by TWIST2, with optional GMR
retarget extension fields when the teleop producer publishes them.
"""

import argparse
import os
import time
from datetime import datetime
from multiprocessing import shared_memory
from threading import Thread

import cv2
import numpy as np
from rich import print

from general_motion_retargeting.utils.episode_writer import EpisodeWriter
from general_motion_retargeting.utils.twist2_redis import Twist2RedisReader
from general_motion_retargeting.utils.vision_client import VisionClient


def _controller_pressed(controller_data, hand, key):
    if not isinstance(controller_data, dict):
        return False
    return bool(controller_data.get(hand, {}).get(key, False))


def main(args):
    redis_reader = Twist2RedisReader(args.redis_host, args.robot)
    print(f"Connected to Redis at {args.redis_host}:6379")

    num_cameras = max(1, args.num_cameras)
    image_shape = (args.image_height, args.image_width * num_cameras, 3)
    image_shm = shared_memory.SharedMemory(
        create=True,
        size=int(np.prod(image_shape) * np.uint8().itemsize),
    )
    image_array = np.ndarray(image_shape, dtype=np.uint8, buffer=image_shm.buf)

    vision_client = VisionClient(
        server_address=args.robot_ip,
        port=args.vision_port,
        img_shape=image_shape,
        img_shm_name=image_shm.name,
        image_show=False,
        depth_show=False,
        unit_test=args.measure_vision_fps,
    )
    vision_thread = Thread(target=vision_client.receive_process, daemon=True)
    vision_thread.start()

    task_dir = os.path.join(args.data_folder, args.task_name)
    recorder = EpisodeWriter(
        task_dir=task_dir,
        frequency=args.frequency,
        image_shape=image_shape,
        data_keys=["rgb"],
    )
    recorder.text_desc(goal=args.goal, desc=args.desc, steps=args.steps)

    recording = False
    prev_toggle_pressed = False
    control_dt = 1.0 / args.frequency
    step_count = 0
    window_name = "TWIST2 data record"

    try:
        while True:
            start_time = time.time()
            controller_data = redis_reader.read_controller_data()

            toggle_pressed = _controller_pressed(
                controller_data, "LeftController", args.toggle_key
            )
            quit_pressed = _controller_pressed(
                controller_data, "LeftController", args.quit_key
            )

            if quit_pressed:
                print("Recording stopped.")
                break

            if toggle_pressed and not prev_toggle_pressed:
                recording = not recording
                if recording:
                    if recorder.create_episode():
                        step_count = 0
                        print("Episode recording started.")
                    else:
                        recording = False
                else:
                    recorder.save_episode()
                    print("Episode save requested.")
            prev_toggle_pressed = toggle_pressed

            if recording:
                data_dict = {
                    "idx": step_count,
                    "rgb": image_array.copy(),
                    "t_img": int(time.time() * 1000),
                }
                data_dict.update(redis_reader.read_record_frame())
                recorder.add_item(data_dict)
                step_count += 1

            if args.image_show:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, image_array)
                cv2.waitKey(1)

            elapsed = time.time() - start_time
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)

    except KeyboardInterrupt:
        print("Interrupted, exiting data recorder.")
    finally:
        recorder.close()
        image_shm.close()
        image_shm.unlink()
        cv2.destroyAllWindows()
        print(f"Done. Episodes saved under {task_dir}")


def parse_args():
    cur_time = datetime.now().strftime("%Y%m%d_%H%M")
    parser = argparse.ArgumentParser(description="Record TWIST2-compatible Redis data.")
    parser.add_argument("--data_folder", default="twist2_demonstration")
    parser.add_argument("--task_name", default=cur_time)
    parser.add_argument("--frequency", default=30, type=int)
    parser.add_argument(
        "--robot",
        default="unitree_g1",
        choices=["unitree_g1", "unitree_g1_with_hands", "robros_igris_c_v2"],
    )
    parser.add_argument("--redis_host", default="localhost")
    parser.add_argument("--robot_ip", default="192.168.123.164")
    parser.add_argument("--vision_port", default=5555, type=int)
    parser.add_argument("--num_cameras", default=2, type=int)
    parser.add_argument("--image_height", default=360, type=int)
    parser.add_argument("--image_width", default=640, type=int)
    parser.add_argument("--image_show", action="store_true")
    parser.add_argument("--measure_vision_fps", action="store_true")
    parser.add_argument("--toggle_key", default="key_two")
    parser.add_argument("--quit_key", default="axis_click")
    parser.add_argument("--goal", default="walk ahead and pick a box.")
    parser.add_argument(
        "--desc",
        default="A humanoid robot walks ahead and picks a box from the table.",
    )
    parser.add_argument(
        "--steps",
        default="step1: walk ahead 1 meter. step2: pick a box from the table.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
