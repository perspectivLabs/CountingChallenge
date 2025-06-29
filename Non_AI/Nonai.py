import cv2
import numpy as np

class WatershedObjectCounter:
    def __init__(self, image, clip_limit=2.0, tile_grid_size=(8, 8), blur_kernel=(5, 5),
                 morph_kernel=(3, 3), morph_iterations=1, min_area=300):
        self.image = image
        self.gray = None
        self.binary = None
        self.opening = None
        self.dist_transform = None
        self.sure_fg = None
        self.sure_bg = None
        self.unknown = None
        self.markers = None
        self.output = None
        self.object_count = 0

        # Params
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.blur_kernel = blur_kernel
        self.morph_kernel = morph_kernel
        self.morph_iterations = morph_iterations
        self.min_area = min_area

    def preprocess(self):
        # Convert to grayscale
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        self.gray = clahe.apply(self.gray)

        # Gaussian blur
        blurred = cv2.GaussianBlur(self.gray, self.blur_kernel, 0)

        # Threshold (Otsu + Inverted)
        _, self.binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Morphological Opening
        kernel = np.ones(self.morph_kernel, np.uint8)
        self.opening = cv2.morphologyEx(self.binary, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations)

    def distance_transform(self):
        self.dist_transform = cv2.distanceTransform(self.opening, cv2.DIST_L2, 5)

    def sure_foreground_background(self):
        noise_std = np.std(self.gray)

        if noise_std > 30:
            factor = 0.2
        elif noise_std > 21:
            factor = 0.1
        elif noise_std > 20:
            factor = 0.05
        else:
            factor = 0.3

        _, self.sure_fg = cv2.threshold(self.dist_transform, factor * self.dist_transform.max(), 255, 0)
        self.sure_fg = np.uint8(self.sure_fg)

        kernel = np.ones(self.morph_kernel, np.uint8)
        self.sure_bg = cv2.dilate(self.opening, kernel, iterations=3)

        self.unknown = cv2.subtract(self.sure_bg, self.sure_fg)

    def apply_watershed(self):
        _, self.markers = cv2.connectedComponents(self.sure_fg)
        self.markers = self.markers + 1
        self.markers[self.unknown == 255] = 0

        image_copy = self.image.copy()
        self.markers = cv2.watershed(image_copy, self.markers)
        self.output = image_copy.copy()
        self.output[self.markers == -1] = [255, 0, 0]  # Mark boundaries in red

    def count_objects(self):
        object_count = 0
        for label in np.unique(self.markers):
            if label <= 1:
                continue

            mask = np.uint8(self.markers == label)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > self.min_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(self.output, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    object_count += 1

        self.object_count = object_count

    def process(self):
        self.preprocess()
        self.distance_transform()
        self.sure_foreground_background()
        self.apply_watershed()
        self.count_objects()
        return self.output, self.object_count



if __name__ == "__main__":
    # Load your image
    image_path = "D:/EDrivebackup/brocamp/DataScienceProjects/JIYAMARYJOSEPH/JIYAMARYJOSEPH/dataset/CountingChallenge/ScrewAndBolt_20240713/20240713_194606.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image at {image_path}")
    else:
        # Create and run the counter
        counter = WatershedObjectCounter(image)
        output_image, count = counter.process()

        print(f"Objects Detected: {count}")

       

   
