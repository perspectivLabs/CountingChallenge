import cv2
import argparse
from ultralytics import YOLO


def process_image(model, img, conf):
    results = model(img, conf=conf)[0]

    canvas = img.copy()
    count = 0

    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        count = len(boxes)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
            canvas = cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0)

    cv2.putText(canvas, f"COUNT: {count}",
                (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (0, 0, 255),
                4)

    return canvas, count


def run(model_path, input_path, output_path, conf):
    model = YOLO(model_path)

    img = cv2.imread(input_path)
    if img is None:
        print("Failed to load image")
        return

    output_img, count = process_image(model, img, conf)

    cv2.imwrite(output_path, output_img)

    print(f"Saved: {output_path}")
    print(f"Count: {count}")


# -------------------------------
# CLI Entry
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Object Counting (Single Image)")

    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model (best.pt)")
    parser.add_argument("--input", type=str, required=True,
                        help="Input image path")
    parser.add_argument("--output", type=str, required=True,
                        help="Output image path")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold")

    args = parser.parse_args()

    run(
        model_path=args.model,
        input_path=args.input,
        output_path=args.output,
        conf=args.conf
    )