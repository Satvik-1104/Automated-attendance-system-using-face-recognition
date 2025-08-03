import multiprocessing
import signal
import sys
import os
from alignment_video import process_video_stream
from recognizer import monitor_and_recognize
import argparse


def signal_handler(sig, frame, detection_process, recognition_process):
    print("Terminating processes...")
    detection_process.terminate()
    recognition_process.terminate()
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Detection and Recognition Pipeline")
    parser.add_argument("--input_source", required=True, help="Video file path or stream URL (e.g., rtsp://...)")
    parser.add_argument("--output_folder", required=True, help="Output directory")
    parser.add_argument("--stream_id", required=True, help="Unique stream identifier")
    parser.add_argument("--mode", choices=["video", "stream"], default=None, help="Processing mode (video or stream)")
    args = parser.parse_args()

    faces_dir = os.path.join(args.output_folder, "faces")
    os.makedirs(faces_dir, exist_ok=True)

    detection_process = multiprocessing.Process(
        target=process_video_stream,
        args=(args.input_source, args.output_folder, args.stream_id, args.mode)
    )
    recognition_process = multiprocessing.Process(
        target=monitor_and_recognize,
        args=(faces_dir, os.path.join(args.output_folder, "recognition_log.txt"))
    )

    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, detection_process, recognition_process))

    recognition_process.start()
    detection_process.start()

    detection_process.join()
    recognition_process.join()
