from typing import Dict

import cv2
import numpy as np
import torch


def get_frame_numbers(metadata: Dict):
    """The keys in json files are frame numbers in str."""
    return metadata.keys()


def get_frame_details(metadata: Dict, frame_number: int):
    """Return details about a specific frame."""
    return metadata[str(frame_number)]


def draw_face_rectangle(frame, face_rect):
    """Draw a rectangle around the face."""
    x, y, w, h = face_rect
    # TODO what does cv2.rectangle return?
    return cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_landmarks(frame, landmark_points):
    """Draw landmark points on the face.

    landmark_points is a list of [x, y] coordinates of length 14. To show the 7 landmark points, we
    need to zip the list and iterate over it.
    """
    landmark_points = landmark_points.reshape(-1, 2)
    for x, y in landmark_points:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    return frame


def show_frame(frame, title: str = "frame"):
    """Show the frame then wait for user to close it."""
    cv2.imshow(title, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_scores_on_tensor(
    tensor: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor = None
) -> torch.Tensor:
    np_ten = np.uint8(
        255 * tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
    )
    img_list = list(np_ten)
    if labels is None:
        labels = torch.ones_like(scores.detach().cpu())
    score_list = scores.detach().cpu().view(-1).tolist()
    label_list = labels.view(-1).tolist()

    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = img_list[0].shape[:2]
    res_list = []
    for im, sc, lbl in zip(img_list, score_list, label_list):
        condition = (sc > 0) == (lbl > 0)
        # print(f"score: {sc:.2f} | label: {lbl} | {'RED' if condition else 'GREEN'}")
        color = (0, 0, 255) if condition else (0, 255, 0)
        im[: h // 4] = np.uint8(np.clip(np.int32(im)[: h // 4] - 100, 0, 255))
        # res_list.append(
        #     cv2.putText(
        #         cv2.UMat(im),
        #         f"{sc:.4f}",
        #         (10, h // 4),
        #         font,
        #         1,
        #         color,
        #         1,
        #         cv2.LINE_AA,
        #     ).get()
        # )
        text = f"{sc:.4f}"
        dst = cv2.putText(
            im.copy(), text, (10, h // 4), font, 1, color, 1, cv2.LINE_AA
        )
        res_list.append(dst)

    np_result = np.array(
        [np.float32(im).transpose(2, 0, 1) / 255.0 for im in res_list]
    )
    return torch.from_numpy(np_result).to(tensor.device, non_blocking=True)
