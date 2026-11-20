from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class IlluminationResult:
    gray: np.ndarray
    background: np.ndarray
    normalized: np.ndarray
    contrast: np.ndarray
    otsu_mask: np.ndarray
    adaptive_mask: np.ndarray
    foreground_mask: np.ndarray
    otsu_threshold: float


def estimate_background(gray: np.ndarray) -> np.ndarray:
    """
    Estimate smooth background illumination using downscaled morphological closing
    followed by Gaussian smoothing. Compensates for vignetting, shadows, and uneven lighting
    even in densely packed regions.
    """
    h, w = gray.shape
    max_dim = max(h, w)
    scale = min(1.0, 800.0 / max_dim)
    small_w = max(1, int(round(w * scale)))
    small_h = max(1, int(round(h * scale)))
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Closing kernel must exceed individual object stroke/head thickness.
    # Pad with BORDER_REPLICATE so strong lens vignetting at image corners is tracked smoothly.
    ksize = max(25, int(min(small_h, small_w) * 0.085) | 1)
    pad = ksize // 2 + 1
    small_pad = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bg_pad = cv2.morphologyEx(small_pad, cv2.MORPH_CLOSE, kernel)
    blur_k = max(15, (ksize // 2) | 1)
    bg_pad = cv2.GaussianBlur(bg_pad, (blur_k, blur_k), 0)
    bg_small = bg_pad[pad : pad + small_h, pad : pad + small_w]
    return cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)


def normalize_illumination_and_segment(img_bgr: np.ndarray) -> IlluminationResult:
    """
    Stage 1: Illumination normalization + accurate foreground mask generation.
    Combines multiplicative background division (vignetting compensation),
    Otsu + gated adaptive thresholding, and scale-aware morphological cleanup.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Background estimation
    background = estimate_background(gray)

    # 2. Illumination compensation via background division (cancels multiplicative vignetting)
    bg_f = np.maximum(background.astype(np.float32), 1.0)
    norm_f = (gray.astype(np.float32) / bg_f) * 255.0
    normalized = np.clip(norm_f, 0, 255).astype(np.uint8)

    # Dark objects on bright background -> invert normalized intensity to get contrast map
    contrast = cv2.subtract(255, normalized)

    # Denoise slightly to suppress sensor grain (e.g. noisy images) while preserving edges
    blur_k = 5 if max(h, w) >= 1200 else 3
    contrast_blur = cv2.GaussianBlur(contrast, (blur_k, blur_k), 0)
    norm_blur = cv2.GaussianBlur(normalized, (blur_k, blur_k), 0)

    # 3a. Global Otsu on illumination-compensated contrast (with safe noise floor)
    otsu_raw, _ = cv2.threshold(contrast_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_val = max(float(otsu_raw), 26.0)
    _, otsu_mask = cv2.threshold(contrast_blur, otsu_val, 255, cv2.THRESH_BINARY)

    # 3b. Adaptive Gaussian thresholding gated by contrast floor to suppress flat-background noise
    block_size = max(31, int(min(h, w) * 0.045) | 1)
    block_size = min(block_size, 151)
    c_offset = max(12.0, 0.35 * otsu_val)
    adaptive_raw = cv2.adaptiveThreshold(
        norm_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_offset,
    )
    contrast_gate = (contrast_blur >= max(22.0, 0.65 * otsu_val)).astype(np.uint8) * 255
    adaptive_mask = cv2.bitwise_and(adaptive_raw, contrast_gate)

    # Combine Otsu and gated adaptive masks
    combined = cv2.bitwise_or(otsu_mask, adaptive_mask)

    # 4. Morphological noise removal & gap closing (gentle to preserve nut holes)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k3, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k3, iterations=1)

    # Remove tiny isolated dust specks and extreme corner-vignetting slivers touching 2 image borders
    min_speck_area = max(25, int(h * w * 0.000012))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    fg_mask = np.zeros_like(cleaned)
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_speck_area:
            continue
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        touches_x = (x == 0) or (x + bw >= w)
        touches_y = (y == 0) or (y + bh >= h)
        if touches_x and touches_y:
            # Corner-hugging lens vignetting artifact
            continue
        fg_mask[labels == lbl] = 255

    return IlluminationResult(
        gray=gray,
        background=background,
        normalized=normalized,
        contrast=contrast,
        otsu_mask=otsu_mask,
        adaptive_mask=adaptive_mask,
        foreground_mask=fg_mask,
        otsu_threshold=otsu_val,
    )
