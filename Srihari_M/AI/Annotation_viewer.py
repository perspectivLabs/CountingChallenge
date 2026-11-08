import os
import cv2


import argparse
import numpy as np
import json

# YOLO Annotation Viewer
class AnnotationViewer:
    def __init__(self, input_dir, obb=False, seg=False, json_dir=None):
        self.image_dir = os.path.join(input_dir, "images")
        self.obb = obb
        self.seg = seg
        self.json_dir = json_dir
        self.label_dir = os.path.join(input_dir, "labels")
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg', 'tiff'))])
        self.current_index = 0
        self.class_colors = {}
        self.saved_images = []
        self.saved_file = os.path.join(input_dir, "saved_images.txt")
        self.accepted_image = os.path.join(input_dir, "accepted_image.txt")
        self.accepted_image_with_text = os.path.join(input_dir, "accepted_image_with_text.txt")
        self.not_annotated_images = os.path.join(input_dir, "not_annotated_images.txt")
        self.not_annotated_images_with_text = os.path.join(input_dir, "not_annotated_images_with_text.txt")
        self.rejected_images = os.path.join(input_dir, "rejected_images.txt")
        self.miscellaneous_images = os.path.join(input_dir, "miscellaneous_images.txt")

    def load_annotations(self, annotation_path):
        """Load YOLO annotations from a file."""
        boxes = []
        try:
            with open(annotation_path, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    
                    if self.obb:
                        if len(parts) >= 6:
                            class_id, x, y, w, h, angle = map(float, parts[:6])
                            boxes.append((class_id, x, y, w, h, angle))
                        elif len(parts) >= 5:
                            class_id, x, y, w, h = map(float, parts[:5])
                            boxes.append((class_id, x, y, w, h, 0.0))
                    else:
                        if len(parts) >= 5:
                            class_id, x, y, w, h = map(float, parts[:5])
                            boxes.append((class_id, x, y, w, h))
        except FileNotFoundError:
            pass  # No annotation file, return empty list
        return boxes

    def load_segmentations(self, json_path):
        """Load segmentation polygons from a JSON file."""
        segmentations = []
        try:
            with open(json_path, 'r') as file:
                data = json.load(file)
                annotations = data.get("annotations", [])
                for start_annotation in annotations:
                    class_id = start_annotation.get("class_id", -1)
                    points = start_annotation.get("segmentation_points", [])
                    if points:
                        segmentations.append((class_id, np.array(points, dtype=np.int32)))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return segmentations

    def get_color_for_class(self, class_id):
        """Get a unique color for each class ID."""
        if class_id not in self.class_colors:
            if class_id != 0:
                self.class_colors[class_id] = (
                int(class_id * 123) % 255, int(class_id * 456) % 255, int(class_id * 789) % 255)

            else:
                self.class_colors[class_id] = (
                int(9 * 123) % 255, int(9 * 456) % 255, int(9 * 789) % 255)
        return self.class_colors[class_id]

    def draw_annotations(self, image, boxes):
        """Draw bounding boxes on the image."""
        h_img, w_img = image.shape[:2]
        for box in boxes:
            if self.obb and len(box) >= 6:
                class_id, x, y, bw, bh, angle = box[:6]
                center = (x * w_img, y * h_img)
                size = (bw * w_img, bh * h_img)
                # Convert angle from radians to degrees
                angle_deg = angle * 180 / np.pi
                
                rect = (center, size, angle_deg)
                box_points = cv2.boxPoints(rect)
                box_points = np.int0(box_points)
                
                color = self.get_color_for_class(class_id)
                cv2.drawContours(image, [box_points], 0, color, 2)
                
                # Text position
                cv2.putText(image, str(int(class_id)), (int(center[0]), int(center[1] - bh * h_img / 2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            elif not self.seg:
                class_id, x, y, bw, bh = box[:5]
                x1 = int((x - bw / 2) * w_img)
                y1 = int((y - bh / 2) * h_img)
                x2 = int((x + bw / 2) * w_img)
                y2 = int((y + bh / 2) * h_img)
                color = self.get_color_for_class(class_id)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, str(int(class_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

    def draw_segmentations(self, image, segmentations):
        """Draw segmentation polygons on the image."""
        for class_id, points in segmentations:
            color = self.get_color_for_class(class_id)
            cv2.polylines(image, [points.reshape((-1, 1, 2))], isClosed=True, color=color, thickness=2)
            # Optional: Fill polygon with transparency
            overlay = image.copy()
            cv2.fillPoly(overlay, [points.reshape((-1, 1, 2))], color)
            cv2.addWeighted(overlay, 0.0, image, 1.0, 0, image)
        return image


    def save_current_image(self):
        """Save the current image path to a list and write to a file."""
        image_file = self.image_files[self.current_index]
        image_path = os.path.join(self.image_dir, image_file)
        if image_path not in self.saved_images:
            self.saved_images.append(image_path)
            with open(self.saved_file, 'a') as file:
                file.write(image_path + '\n')
            print(f"Saved {image_path} to list.")
        else:
            print(f"{image_path} is already in the list.")

    def save_accepted_image(self):
        """Save the current image path to a list and write to a file."""
        image_file = self.image_files[self.current_index]
        image_path = os.path.join(self.image_dir, image_file)
        if image_path not in self.saved_images:
            self.saved_images.append(image_path)
            with open(self.accepted_image, 'a') as file:
                file.write(image_path + '\n')
            print(f"Saved {image_path} to list.")
        else:
            print(f"{image_path} is already in the list.")

    def save_accepted_image_with_text(self):
        """Save the current image path to a list and write to a file."""
        image_file = self.image_files[self.current_index]
        image_path = os.path.join(self.image_dir, image_file)
        if image_path not in self.saved_images:
            self.saved_images.append(image_path)
            with open(self.accepted_image_with_text, 'a') as file:
                file.write(image_path + '\n')
            print(f"Saved {image_path} to list.")
        else:
            print(f"{image_path} is already in the list.")

    def save_not_annotated_images(self):
        """Save the current image path to a list and write to a file."""
        image_file = self.image_files[self.current_index]
        image_path = os.path.join(self.image_dir, image_file)
        if image_path not in self.saved_images:
            self.saved_images.append(image_path)
            with open(self.not_annotated_images, 'a') as file:
                file.write(image_path + '\n')
            print(f"Saved {image_path} to list.")
        else:
            print(f"{image_path} is already in the list.")

    def save_not_annotated_images_with_text(self):
        """Save the current image path to a list and write to a file."""
        image_file = self.image_files[self.current_index]
        image_path = os.path.join(self.image_dir, image_file)
        if image_path not in self.saved_images:
            self.saved_images.append(image_path)
            with open(self.not_annotated_images_with_text, 'a') as file:
                file.write(image_path + '\n')
            print(f"Saved {image_path} to list.")
        else:
            print(f"{image_path} is already in the list.")

    def save_rejected_images(self):
        """Save the current image path to a list and write to a file."""
        image_file = self.image_files[self.current_index]
        image_path = os.path.join(self.image_dir, image_file)
        if image_path not in self.saved_images:
            self.saved_images.append(image_path)
            with open(self.rejected_images, 'a') as file:
                file.write(image_path + '\n')
            print(f"Saved {image_path} to list.")
        else:
            print(f"{image_path} is already in the list.")

    def save_miscellaneous_images(self):
        """Save the current image path to a list and write to a file."""
        image_file = self.image_files[self.current_index]
        image_path = os.path.join(self.image_dir, image_file)
        if image_path not in self.saved_images:
            self.saved_images.append(image_path)
            with open(self.miscellaneous_images, 'a') as file:
                file.write(image_path + '\n')
            print(f"Saved {image_path} to list.")
        else:
            print(f"{image_path} is already in the list.")

    def show_image(self):
        """Display the current image with annotations."""
        if not self.image_files:
            print("No images found in the directory.")
            return

        image_file = self.image_files[self.current_index]
        image_path = os.path.join(self.image_dir, image_file)
        annotation_path = os.path.join(self.label_dir, os.path.splitext(image_file)[0] + ".txt")

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load {image_path}")
            return

        boxes = self.load_annotations(annotation_path)
        image_with_annotations = self.draw_annotations(image, boxes)
        
        if self.seg and self.json_dir:
            json_path = os.path.join(self.json_dir, os.path.splitext(image_file)[0] + ".json")
            if os.path.exists(json_path):
                segmentations = self.load_segmentations(json_path)
                image_with_annotations = self.draw_segmentations(image_with_annotations, segmentations)
            else:
                 print(f"JSON file not found: {json_path}")

        print(image_file)

        # Add image name to the top of the image
        cv2.putText(image_with_annotations, image_file, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

        # Make the window resizable
        cv2.namedWindow("Annotation Viewer", cv2.WINDOW_NORMAL)
        cv2.imshow("Annotation Viewer", image_with_annotations)

    def run(self):
        """Run the annotation viewer."""
        while True:
            self.show_image()
            key = cv2.waitKey(0)

            if key == 27:  # Esc key
                break
            elif key == 97 or key == 81:  # Left key (a)
                self.current_index = (self.current_index - 1) % len(self.image_files)
            elif key == 100 or key == 83:  # Right key (d)
                self.current_index = (self.current_index + 1) % len(self.image_files)
            elif key == ord('l'):  # 'L' key
                self.save_current_image()
            elif key == ord('c'):
                self.save_accepted_image()
            elif key == ord('e'):
                self.save_accepted_image_with_text()
            elif key == ord('n'):
                self.save_not_annotated_images()
            elif key == ord('t'):
                self.save_not_annotated_images_with_text()
            elif key == ord('r'):
                self.save_rejected_images()
            elif key == ord('m'):
                self.save_miscellaneous_images()
            

        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Yolo Annotation Viewer")

    parser.add_argument("input_dir", type=str, nargs='+', help="Path to the input directory containing images and labels in YOLO format")
    parser.add_argument("--obb", action="store_true", help="Enable Oriented Bounding Box (OBB) visualization")
    parser.add_argument("--seg", action="store_true", help="Enable Segmentation visualization")
    parser.add_argument("--json_dir", type=str, help="Path to the directory containing JSON files for segmentation")

    args = parser.parse_args()

    if args.seg and not args.json_dir:
        input_dir = input("Please provide the path to the JSON folder for segmentation: ").strip()
        if not input_dir:
            print("JSON folder path is required for segmentation visualization.")
            exit()
        args.json_dir = input_dir
    
    input_dir_array = args.input_dir

    for input_directory in input_dir_array:
        viewer = AnnotationViewer(input_directory, obb=args.obb, seg=args.seg, json_dir=args.json_dir)
        viewer.run()
        
        
        
        
    
