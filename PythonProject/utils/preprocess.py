import cv2
import numpy as np

def preprocess_frame(
    frame,
    dark_threshold=150,
    keep_ground_ratio=0.20  # keep bottom 15% of image
):
    # 1. Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # 2. Dark pixel mask (pipes + bird outlines)
    _, mask = cv2.threshold(
        gray,
        dark_threshold,
        255,
        cv2.THRESH_BINARY_INV
    )

    # 3. Explicitly keep the ground band
    h = mask.shape[0]
    ground_start = int(h * (1.0 - keep_ground_ratio))
    mask[ground_start:, :] = 255  # force ground visible

    # 4. Resize
    mask = cv2.resize(mask, (84, 84), interpolation=cv2.INTER_NEAREST)

    # 5. Normalize
    return (mask / 255.0).astype(np.float32)
