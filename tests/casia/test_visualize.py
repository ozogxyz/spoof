from typing import Any, Dict

from numpy import dtype, generic, ndarray

from src.visualize import (
    draw_face_rectangle,
    draw_landmark_points,
    show_frame,
)


def test_draw_rectangles(test_sample: Dict[Any, Any]):
    """Test draw_face_rectangle."""
    frame = test_sample.get("image")
    face_rect = test_sample["meta"].get("face_rect")
    image = draw_face_rectangle(frame, face_rect)
    show_frame(image)


def test_draw_landmark(
    test_frame: ndarray[int, dtype[generic]], test_sample: Dict[Any, Any]
):
    """Test draw_landmark_points."""
    frame = test_sample.get("image")
    face_landmark = test_sample["meta"].get("face_landmark")
    image = draw_landmark_points(frame, face_landmark)
    show_frame(image)
