import numpy as np

def calculate_mouse_centers(keypoints_sequence):
    """Return the mean keypoint center for each mouse in each frame.

    Parameters
    ----------
    keypoints_sequence : array-like, shape (frames, 2, 2, 7)
        CalMS21 keypoints indexed as [frame, mouse, coord, keypoint].

    Returns
    -------
    np.ndarray, shape (frames, 2, 2)
        Mouse centers indexed as [frame, mouse, coord], where coord is x/y.
    """
    keypoints_sequence = np.asarray(keypoints_sequence)
    if keypoints_sequence.ndim != 4 or keypoints_sequence.shape[1:3] != (2, 2):
        raise ValueError(
            "keypoints_sequence must have shape (frames, 2, 2, keypoints)"
        )

    return np.nanmean(keypoints_sequence, axis=3)


def calculate_mouse_distance(keypoints_sequence):
    """Return center-to-center distance between the two mice for each frame.

    The distance is computed from the average location of all keypoints for
    each mouse, so the result is in image pixels.
    """
    centers = calculate_mouse_centers(keypoints_sequence)
    return np.linalg.norm(centers[:, 0, :] - centers[:, 1, :], axis=1)
