import os
import sys
from functools import reduce

import cv2

# Add parent directory to path for easy import
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.dataset.read_metadata import read_metadata


def get_frame_numbers(metadata):
    """The keys in json files are frame numbers in str."""
    return metadata.keys()


def get_frame_details(metadata, frame_number):
    """Return details about a specific frame."""
    return metadata[str(frame_number)]


def draw_face_rectangle(frame, face_rect):
    """Draw a rectangle around the face."""
    x, y, w, h = face_rect
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame


def draw_landmark_points(frame, landmark_points):
    """
    Draw landmark points on the face.
    landmark_points is a list of [x, y] coordinates of length 14.
    To show the 7 landmark points, we need to zip the list and iterate over it.
    """
    for (x, y) in list(zip(landmark_points[0::2], landmark_points[1::2])):
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

    return frame


def show_frame(frame):
    """Show the frame then wait for user to close it."""
    cv2.imshow("frame", frame)
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()


# visualize face rectangles and landmark points from the meta
def main():
    frame_src = "tests/sample/frames/frame1.jpg"
    meta_src = "tests/sample/test.json"

    # print(f"Visualizing rectangles of {video_src} from {meta_src}")

    metadata = read_metadata(meta_src).send(None)

    # get frame numbers
    frame_numbers = get_frame_numbers(metadata)
    # print(f"Number of frames in the video: {len(frame_numbers)}")

    # get frame details
    frame_details = get_frame_details(metadata, 1)
    # print(frame_details)

    # get a sample frame
    frame = cv2.imread(frame_src)

    # show_frame(draw_face_rectangle(frame, frame_details.get("face_rect")))
    # show_frame(draw_landmark_points(frame, frame_details.get("lm7pt")))
    landmarks_7pt = draw_landmark_points(frame, frame_details.get("lm7pt"))
    show_frame(landmarks_7pt)


if __name__ == "__main__":
    main()
