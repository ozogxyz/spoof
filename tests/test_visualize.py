import cv2
import pytest
from src.dataset.visualize import (
    draw_face_rectangle,
    draw_landmark_points,
    show_frame,
)


def test_draw_rectangles(test_frame, test_sample):
    """Test draw_face_rectangle."""
    face_rect = test_sample.get("face_rect")
    image = draw_face_rectangle(test_frame, face_rect)

    show_frame(image)

    assert image.shape == (1280, 720, 3)


def test_draw_landmark(test_frame, test_sample):
    """Test draw_landmark_points."""
    landmark_points = test_sample.get("lm7pt")
    image = draw_landmark_points(test_frame, landmark_points)

    show_frame(image)

    assert image.shape == (1280, 720, 3)
