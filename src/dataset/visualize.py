from typing import Dict

import cv2


def get_frame_numbers(metadata: Dict):
    """The keys in json files are frame numbers in str."""
    return metadata.keys()


def get_frame_details(metadata: Dict, frame_number: int):
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
    for x, y in list(zip(landmark_points[0::2], landmark_points[1::2])):
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

    return frame


def show_frame(frame, title: str = "frame"):
    """Show the frame then wait for user to close it."""
    cv2.imshow(title, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
