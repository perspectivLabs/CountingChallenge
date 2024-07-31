"""Script for counting the number of objects present in an image and creating mask for it."""

import cv2 as cv
import numpy as np
from pathlib import Path
from logzero import logger
from dataset_downloader import DownloadFiles
from multiprocessing import freeze_support


class CountObject:
    """Class for counting objects present in an image."""

    def __init__(self) -> None:
        """Initialise the required parameters."""
        self.min_threshold = 400
        Path.mkdir(Path.cwd() / "masks", exist_ok=True)
        self.mask_storage_dir = Path(Path.cwd() / "masks")
        self.dataset = Path(Path.cwd() / "dataset")

    def process_datasets(self):
        """
        Read the images which are to be processed along with the mask files and file names.

        Returns:
            images (list(np.ndarray)): List of images which are to processed
        """
        dataset_1 = self.dataset / "dataset_1"
        dataset_2 = self.dataset / "dataset_2"

        dataset_1_files = [f for f in dataset_1.iterdir() if f.is_file()]
        dataset_2_files = [f for f in dataset_2.iterdir() if f.is_file()]

        files = []
        final_images = []
        masks = []

        files.extend(self.get_first_two_files(dataset_1_files, "dataset 1"))
        files.extend(self.get_first_two_files(dataset_2_files, "dataset 2"))
        
        for file in files:
            image = cv.imread(str(file))
            final_images.append(image)

            mask = np.zeros_like(image)
            masks.append(mask)
        
        return (final_images, masks, files)

    @staticmethod
    def get_first_two_files(file_list, dataset_name):
        """
        Utility function for reading the first two files from the file list

        Returns:
            files (list): First two files according to the conditions applicable.
        """
        if len(file_list) >= 2:
            return file_list[:2]
        elif file_list:
            return [file_list[0]]
        else:
            logger.warning(f"WARNING: The {dataset_name} was empty")
            return []

    @staticmethod
    def preprocessing(image):
        """
        Preprocess the image for contour detection.

        Args:
            image (np.ndarray): Input image for preprocessing
        
        Returns:
            image (np.ndarray): Preprocessed image
        """
        # Convert the image to grayscale
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blurred_image = cv.GaussianBlur(gray_image, (5, 5), 0)

        # Apply thresholding to convert the image to binary format
        _, binary_image = cv.threshold(blurred_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # Apply dilation to the binary image
        dilated_image = cv.dilate(binary_image, None, iterations=3)

        return dilated_image
    
    def find_contours(self, image):
        """
        Find the contours for the input image.

        Args:
            image (np.ndarray): Input image which has been preprocessed
        
        Returns:
            filtered_contours (list): List of filtered contours
        """
        contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        filtered_contours = [contour for contour in contours if cv.contourArea(contour) > self.min_threshold]

        return filtered_contours
    
    def count_object_and_create_mask(self, filtered_contours, masks, file_name):
        """
        Count the number of objects present in the image.

        Args:
            filtered_contours (list): List of filtered contours
            masks (np.ndarray): Mask for the detected objects
            file_name (str): File name for the file being processed
        """
        mask_path = f"{str(self.mask_storage_dir)}/mask_{file_name}"
        for contour in filtered_contours:
            cv.drawContours(masks, [contour], -1, (0, 255, 0), thickness=cv.FILLED)
        cv.imwrite(mask_path, masks)

        return len(filtered_contours)
    
    def main(self):
        """Main processing logic for counting the objects in the image."""
        # Step 1: Download the dataset from the google drive
        freeze_support()
        obj = DownloadFiles()
        obj.main()

        # Step 2: Get first 2 images from each dataset
        final_images, masks, files = self.process_datasets()

        for image, mask, file in zip(final_images, masks, files):
            # Step 3: Perform preprocessing for contour detection 
            preprocessed_image = self.preprocessing(image)

            # Step 4: Get the contours from the preprocessed image
            contours = self.find_contours(preprocessed_image)
            
            # Step 5: Count the number of screws present in the image
            screw_count = self.count_object_and_create_mask(contours, mask, file.name)

            logger.info(f"INFO: Number of screws found in image {file} are {screw_count}")


if __name__ == "__main__":
    obj = CountObject()
    obj.main()
