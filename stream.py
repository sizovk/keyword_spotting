import logging
import multiprocessing as mp

import torch
from torchaudio.io import StreamReader

logger = logging.getLogger(__file__)
FRAMES_PER_CHUNK = 16_000
SAMPLE_RATE = 16_000


def audio_stream(queue: mp.Queue):
    """
    Learn more about how to install and use streaming audio here
    https://pytorch.org/audio/stable/tutorials/streaming_api2_tutorial.html
    """

    streamer = StreamReader(src=":0", format="avfoundation")
    streamer.add_basic_audio_stream(frames_per_chunk=FRAMES_PER_CHUNK, sample_rate=SAMPLE_RATE)
    stream_iterator = streamer.stream(-1, 1)

    logger.info("Start audio streaming")
    while True:
        (chunk_,) = next(stream_iterator)
        logger.info("Put chunk to queue")
        queue.put(chunk_)


if __name__ == "__main__":
    model = torch.load("stream_model.pth").eval()

    ctx = mp.get_context("spawn")
    chunk_queue = ctx.Queue()
    streaming_process = ctx.Process(target=audio_stream, args=(chunk_queue,))

    streaming_process.start()
    ind = 0
    while True:
        try:
            chunk = chunk_queue.get()
            chunk = chunk.view(1, -1)
            print(f"{chunk.shape=}")

            with torch.inference_mode():
                result = model.forward_chunk(chunk)

            if result > 0.7:
                print("DETECTED KEY WORD", f"chunk number {ind}")
            
            ind += 1

        except KeyboardInterrupt:
            break
        except Exception as exc:
            raise exc

    streaming_process.join()