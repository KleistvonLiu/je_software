#!/usr/bin/env python
import logging
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import multiprocessing
import queue
import threading
from pathlib import Path
import time
import os

import numpy as np
import PIL.Image


def safe_stop_image_writer(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            dataset = kwargs.get("dataset")
            image_writer = getattr(dataset, "image_writer", None) if dataset else None
            if image_writer is not None:
                print("Waiting for image writer to terminate...")
                image_writer.stop()
            raise e

    return wrapper


def image_array_to_pil_image(image_array: np.ndarray, range_check: bool = True) -> PIL.Image.Image:
    # TODO(aliberts): handle 1 channel and 4 for depth images
    if image_array.ndim != 3 and image_array.ndim != 1:
        raise ValueError(f"The array has {image_array.ndim} dimensions, but 1 or 3 is expected for an image.")

    if image_array.shape[0] == 3 or image_array.shape[0] == 1:
        # Transpose from pytorch convention (C, H, W) to (H, W, C)
        image_array = image_array.transpose(1, 2, 0)

    elif image_array.shape[-1] != 3 and image_array.shape[-1] != 1:
        raise NotImplementedError(
            f"The image has {image_array.shape[-1]} channels, but 1 or 3 is required for now."
        )

    if image_array.dtype != np.uint8 and image_array.dtype != np.uint16:
        logging.warning(f"The image has {image_array.dtype} dtype")
        if range_check:
            max_ = image_array.max().item()
            min_ = image_array.min().item()
            if max_ > 1.0 or min_ < 0.0:
                raise ValueError(
                    "The image data type is float, which requires values in the range [0.0, 1.0]. "
                    f"However, the provided range is [{min_}, {max_}]. Please adjust the range or "
                    "provide a uint8 image with values in the range [0, 255]."
                )

        image_array = (image_array * 255).astype(np.uint8)

    if image_array.ndim == 3 and image_array.shape[2] == 1:
        image_array = image_array[..., 0]
    return PIL.Image.fromarray(image_array)


def write_image(image: np.ndarray | PIL.Image.Image, fpath: Path):
    try:
        if isinstance(image, np.ndarray):
            img = image_array_to_pil_image(image)
        elif isinstance(image, PIL.Image.Image):
            img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        img.save(fpath)
    except Exception as e:
        print(f"Error writing image {fpath}: {e}")


def worker_thread_loop(queue: queue.Queue, parent=None):
    """Worker thread: write image then update parent's counters if provided.
    parent: AsyncImageWriter instance or None (for process-spawned threads parent will be None).
    """
    while True:
        item = queue.get()
        if item is None:
            queue.task_done()
            break
        image_array, fpath = item
        # write image; avoid verbose per-image prints to reduce log spam
        try:
            write_image(image_array, fpath)
            # update counters on parent writer if available
            if parent is not None:
                try:
                    parent._written_count += 1
                    try:
                        parent._written_bytes += int(os.path.getsize(str(fpath)))
                    except Exception:
                        # best-effort: ignore file-size errors
                        pass
                except Exception:
                    pass
        except Exception:
            # write_image already prints error; continue
            pass
        finally:
            queue.task_done()


def worker_process(queue: queue.Queue, num_threads: int):
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker_thread_loop, args=(queue,))
        t.daemon = True
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


class AsyncImageWriter:
    """
    This class abstract away the initialisation of processes or/and threads to
    save images on disk asynchronously, which is critical to control a robot and record data
    at a high frame rate.

    When `num_processes=0`, it creates a threads pool of size `num_threads`.
    When `num_processes>0`, it creates processes pool of size `num_processes`, where each subprocess starts
    their own threads pool of size `num_threads`.

    The optimal number of processes and threads depends on your computer capabilities.
    We advise to use 4 threads per camera with 0 processes. If the fps is not stable, try to increase or lower
    the number of threads. If it is still not stable, try to use 1 subprocess, or more.
    """

    def __init__(self, num_processes: int = 0, num_threads: int = 1):
        self.num_processes = num_processes
        self.num_threads = num_threads
        self.queue = None
        self.threads = []
        self.processes = []
        self._stopped = False

        # new runtime metrics
        self._written_count = 0
        self._written_bytes = 0
        self._monitor_thread = None

        if num_threads <= 0 and num_processes <= 0:
            raise ValueError("Number of threads and processes must be greater than zero.")

        if self.num_processes == 0:
            # Use threading
            self.queue = queue.Queue()
            for _ in range(self.num_threads):
                # pass self as parent so worker can update counters
                t = threading.Thread(target=worker_thread_loop, args=(self.queue, self))
                t.daemon = True
                t.start()
                self.threads.append(t)
        else:
            # Use multiprocessing
            self.queue = multiprocessing.JoinableQueue()
            for _ in range(self.num_processes):
                p = multiprocessing.Process(target=worker_process, args=(self.queue, self.num_threads))
                p.daemon = True
                p.start()
                self.processes.append(p)

        # start monitor thread (daemon) to print queue depth and basic counters
        def _monitor():
            try:
                last_print_t = 0.0
                while not self._stopped:
                    try:
                        qsize = self.queue.qsize()
                    except Exception:
                        qsize = -1
                    now_t = time.time()
                    # Print only when queue non-empty or every 5 seconds to reduce noise
                    if qsize > 0 or (now_t - last_print_t) >= 5.0:
                        print(f"[image_writer] qsize={qsize} written_count={self._written_count} written_bytes={self._written_bytes}")
                        last_print_t = now_t
                    time.sleep(1.0)
            except Exception:
                pass

        self._monitor_thread = threading.Thread(target=_monitor, name='image-writer-monitor', daemon=True)
        self._monitor_thread.start()

    def save_image(self, image: np.ndarray | PIL.Image.Image, fpath: Path):
        # if isinstance(image, torch.Tensor):
        #     # Convert tensor to numpy array to minimize main process time
        #     image = image.cpu().numpy()
        self.queue.put((image, fpath))

    def wait_until_done(self):
        self.queue.join()

    def stop(self):
        if self._stopped:
            return

        # signal monitor to stop
        self._stopped = True
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=1.0)

        if self.num_processes == 0:
            for _ in self.threads:
                self.queue.put(None)
            for t in self.threads:
                t.join()
        else:
            num_nones = self.num_processes * self.num_threads
            for _ in range(num_nones):
                self.queue.put(None)
            for p in self.processes:
                p.join()
                if p.is_alive():
                    p.terminate()
            self.queue.close()
            self.queue.join_thread()

        self._stopped = True

    def is_idle(self):
        """Return True if all image write tasks are done (queue is empty)."""
        return self.queue.empty()
