import numpy as np
import cv2
from sklearn.metrics import rand_score

def build_label_masks(label_img: np.ndarray) -> dict:
    """
    Build binary masks for each label in a segmentation map.

    Parameters
    ----------
    label_img : np.ndarray
        2D array of labeled regions.

    Returns
    -------
    dict of int -> np.ndarray
        Dictionary mapping each label value to a boolean mask.
    """
    return {label: (label_img == label) for label in np.unique(label_img)}

def local_refinement_error(img1: np.ndarray, img2: np.ndarray, masks1: dict, masks2: dict) -> float:
    """
    Compute the Local Refinement Error (LRE) from img1 to img2.

    Parameters
    ----------
    img1 : np.ndarray
        First labeled image (reference).
    img2 : np.ndarray
        Second labeled image (comparison).
    masks1 : dict
        Label masks for img1.
    masks2 : dict
        Label masks for img2.

    Returns
    -------
    float
        Sum of per-pixel refinement error from img1 to img2.
    """
    error = 0.0

    # Get unique (label1, label2) pairs and how often they occur
    pairs, counts = np.unique(
        np.stack((img1.ravel(), img2.ravel()), axis=1),
        axis=0,
        return_counts=True
    )

    for (label1, label2), count in zip(pairs, counts):
        region1 = masks1[label1]
        region2 = masks2[label2]

        region_size = np.count_nonzero(region1)
        if region_size == 0:
            continue

        diff = np.logical_and(region1, ~region2)
        local_error = np.count_nonzero(diff) / region_size

        error += count * local_error

    return error

def global_consistency_error(
    segmented_image: str | np.ndarray,
    ground_truth: str | np.ndarray
) -> float:
    """
    Compute the Global Consistency Error (GCE) between two segmentation maps.

    Parameters
    ----------
    segmented_image : str or np.ndarray
        Segmentation result image (file path or 2D array).
    ground_truth : str or np.ndarray
        Ground truth image (file path or 2D array).

    Returns
    -------
    float
        Global Consistency Error in [0, 1].
    """
    seg = cv2.imread(segmented_image, cv2.IMREAD_GRAYSCALE) if isinstance(segmented_image, str) else segmented_image.copy()
    gt = cv2.imread(ground_truth, cv2.IMREAD_GRAYSCALE) if isinstance(ground_truth, str) else ground_truth.copy()

    if seg is None or gt is None:
        raise ValueError("Image could not be loaded")
    if seg.shape != gt.shape:
        raise ValueError("Images have different shapes")

    seg_masks = build_label_masks(seg)
    gt_masks = build_label_masks(gt)

    lre1 = local_refinement_error(seg, gt, seg_masks, gt_masks)
    lre2 = local_refinement_error(gt, seg, gt_masks, seg_masks)

    return min(lre1, lre2) / gt.size

def var_information(
    segmented_image: str | np.ndarray,
    ground_truth: str | np.ndarray
) -> float:
    """
    Compute the Variation of Information (VI) between two segmentation maps.

    Parameters
    ----------
    segmented_image : str or np.ndarray
        Predicted segmentation image (file path or 2D array).
    ground_truth : str or np.ndarray
        Ground truth segmentation image (file path or 2D array).

    Returns
    -------
    float
        Variation of Information metric.
    """
    seg = cv2.imread(segmented_image, cv2.IMREAD_GRAYSCALE) if isinstance(segmented_image, str) else segmented_image.copy()
    gt = cv2.imread(ground_truth, cv2.IMREAD_GRAYSCALE) if isinstance(ground_truth, str) else ground_truth.copy()

    if seg is None or gt is None:
        raise ValueError("Image could not be loaded")
    if seg.shape != gt.shape:
        raise ValueError("Images have different shapes")

    seg = seg.flatten()
    gt = gt.flatten()

    seg_labels, seg = np.unique(seg, return_inverse=True)
    gt_labels, gt = np.unique(gt, return_inverse=True)

    P_ij = np.histogram2d(seg, gt, bins=(len(seg_labels), len(gt_labels)))[0]
    P_ij /= P_ij.sum()

    P_i = P_ij.sum(axis=1, keepdims=True)
    P_j = P_ij.sum(axis=0, keepdims=True)

    eps = 1e-12
    entropy_seg = -np.sum(P_i * np.log2(P_i + eps))
    entropy_gt  = -np.sum(P_j * np.log2(P_j + eps))

    mutual_info = np.sum(P_ij * np.log2(P_ij / (P_i @ P_j + eps) + eps))
    return float(entropy_seg + entropy_gt - 2 * mutual_info)

def rand_index(
    segmented_image: str | np.ndarray, 
    ground_truth: str | np.ndarray
) -> float:
    """
    Compute the Rand Index (RI) between two labeled segmentation maps.

    Parameters
    ----------
    segmented_image : str or np.ndarray
        Segmented label image or file path to grayscale segmentation.
    ground_truth : str or np.ndarray
        Ground truth label image or file path to grayscale segmentation.

    Returns
    -------
    float
        Rand Index score between 0 and 1.
    """
    seg = cv2.imread(segmented_image, cv2.IMREAD_GRAYSCALE) if isinstance(segmented_image, str) else segmented_image
    gt = cv2.imread(ground_truth, cv2.IMREAD_GRAYSCALE) if isinstance(ground_truth, str) else ground_truth

    if seg is None or gt is None:
        raise ValueError("Could not load one or both images.")
    if seg.shape != gt.shape:
        raise ValueError("Images must have the same shape.")

    seg = np.unique(seg, return_inverse=True)[1].ravel()
    gt = np.unique(gt, return_inverse=True)[1].ravel()

    return rand_score(gt, seg)
    
