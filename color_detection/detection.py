import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import imutils

def detect_colors(image_path, k=10, min_area=300):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or cannot be opened.")

    # Slight blur to reduce noise
    image = cv2.medianBlur(image, 3)

    # Resize if large
    max_dim = 800
    if max(image.shape) > max_dim:
        scale = max_dim / max(image.shape)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Convert to Lab for clustering
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    h, w = lab_image.shape[:2]
    pixels = lab_image.reshape((-1, 3)).astype(np.float32)

    # KMeans cluster
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(pixels)
    centers_lab = km.cluster_centers_

    # Convert each Lab centroid back to BGR, then RGB, then HEX
    centers_bgr = cv2.cvtColor(np.uint8([centers_lab]), cv2.COLOR_LAB2BGR)[0]
    centers_rgb = centers_bgr[:, ::-1]  # BGR to RGB
    hex_colors = ['#%02x%02x%02x' % tuple(map(int, rgb)) for rgb in centers_rgb]

    detected_colors = {}
    output = image.copy()

    for idx, center_lab in enumerate(centers_lab):
        color_hex = hex_colors[idx]
        count = np.count_nonzero(labels == idx)
        detected_colors[color_hex] = detected_colors.get(color_hex, 0) + count

        # Create mask for this cluster
        mask = (labels == idx).reshape((h, w)).astype(np.uint8) * 255
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    text = color_hex.upper()
                    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(output, (cx - tw//2 - 5, cy - th - 5),
                                  (cx + tw//2 + 5, cy + baseline), (0, 0, 0), -1)
                    cv2.putText(output, text, (cx - tw//2, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    total_pixels = h * w
    color_summary = {
        hex: round((cnt / total_pixels) * 100, 2) for hex, cnt in detected_colors.items()
    }

    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(output_rgb)
    return pil_img, color_summary

# Example usage:
if __name__ == "__main__":
    annotated_img, summary = detect_colors("your_image.png", k=10)
    annotated_img.show()  # shows annotated image
    print("Detected colours summary (hex: %):", summary)
