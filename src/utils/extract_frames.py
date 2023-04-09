import os
from pathlib import Path

import cv2


def filter_files_by_ext(path: str, ext: str):
    for path in Path(path).rglob(f"*{ext}"):  # type: ignore
        yield str(path)


# Function to extract frames
def capture_frames(video_path: str, save_dest: str) -> int:
    """Extracts frames from a video and saves them to a directory. If video filename ends with _1,
    then label is real, otherwise spoof.

    {video_filename}_frame_{cur_frame}_{label}.jpg
    {video_filename}_frame_{cur_frame}_{label}.json

    Args:
        video_path: path to the video file
        save_dest: path to the directory where frames will be saved
    Returns:
        cur_frame: number of frames extracted
    """
    # Open and start to read the video
    cap = cv2.VideoCapture(video_path)
    print("Started extracting frames from {}".format(Path(video_path).stem))

    if cap.isOpened():
        cur_frame = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save the frame
            label = "1" if Path(video_path).stem.endswith("_1") else "0"
            filename = (
                Path(video_path).stem
                + "_frame_"
                + str(cur_frame)
                + "_"
                + label
                + ".jpg"
            )
            cv2.imwrite(os.path.join(save_dest, filename), frame)

            cur_frame += 1
        cap.release()

        print("Finished extracting frames: {}".format(Path(video_path).stem))
        print("Total frames: {}".format(cur_frame))
        cv2.destroyAllWindows()
        return cur_frame

    else:
        print("Cannot open video file")
        return -1


def extract_frames(
    video_src: str,
    dest: str,
    video_ext: str = ".avi",
) -> None:
    """Extracts frames and metadata from all videos in src and saves them to dest with the
    following structure:

    dest/client_name/live/video_name/video_name_frame_{frame number}.jpg
    dest/client_name/spoof/video_name/video_name_frame_{frame number}.jpg

    Args:
        video_src (str): path to the directory containing the videos
        dest (str): path to the directory where frames will be saved
        video_ext (str): video file extension
    """
    # Get list of video files
    video_files = filter_files_by_ext(video_src, ext=video_ext)

    # Extract frames
    total_frame_count = 0
    for video_file in video_files:
        input_filename = Path(video_file).stem

        # Get the client name
        client = Path(video_file).parent.stem
        label = input_filename.split("_")[-1]
        if label == "1":
            # live video
            output_directory = Path(dest) / client / "live" / input_filename
        elif label == "0":
            # spoof video
            output_directory = Path(dest) / client / "spoof" / input_filename
        else:
            print("Invalid label: {}".format(label))
            continue

        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        frame_count = capture_frames(video_file, str(output_directory))
        total_frame_count += frame_count

    print("Total frames extracted: {}".format(total_frame_count))


def extract_metadata():
    pass


def main():
    extract_frames(
        video_src="data/casia/test",
        dest="data/casia/test_frames",
    )


if __name__ == "__main__":
    main()
