import cv2
from src.dataset.visualize import (
    draw_face_rectangle,
    draw_landmark_points,
)


# def test_draw_rectangles():
#     """Test draw_face_rectangle."""
#     frame = cv2.imread("sample/frames/frame1.jpg")
#     meta_src = "tests/sample/test.json"
#     face_rect = [0, 0, 100, 100]
#     image = draw_face_rectangle(frame, face_rect)

#     cv2.imshow("image", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     assert image.shape == (1280, 720, 3)


def test_draw_landmark():
    """Test draw_landmark_points."""
    frame = cv2.imread("sample/frames/frame1.jpg")
    meta_src = "tests/sample/test.json"
    landmark_points = [[0, 0], [100, 100]]
    image = draw_landmark_points(frame, landmark_points)

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    assert image.shape == (1280, 720, 3)
