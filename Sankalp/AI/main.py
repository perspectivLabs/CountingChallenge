import json
from pathlib import Path
import requests
import cv2 as cv
from logzero import logger
from multiprocessing import freeze_support
from dataset_downloader import DownloadFiles

class CountObjects:
    """Class for counting objects."""

    def __init__(self) -> None:
        """Initialize required parameters."""
        project_id = "screw-and-bolts/1"
        confidence = 0.1
        iou_thresh = 0.5

        api_key = "C8v68RnOpbPsqdARhvJW"
        self.model_endpoint = f"https://detect.roboflow.com/{project_id}?api_key={api_key}&confidence={confidence}&overlap={iou_thresh}"
        self.dataset = Path(Path.cwd() / "dataset")

    def process_datasets(self):
        """
        Read the images which are to be processed along with the mask files and file names.

        Returns:
            images (list(np.ndarray)): List of images which are to processed
            files (list): List of file names for the files being processed
        """
        dataset_1 = self.dataset / "dataset_1"
        dataset_2 = self.dataset / "dataset_2"

        dataset_1_files = [f for f in dataset_1.iterdir() if f.is_file()]
        dataset_2_files = [f for f in dataset_2.iterdir() if f.is_file()]

        files = []
        final_images = []

        files.extend(self.get_first_two_files(dataset_1_files, "dataset 1"))
        files.extend(self.get_first_two_files(dataset_2_files, "dataset 2"))

        final_images = [cv.imread(str(file)) for file in files]
        
        return (final_images, files)

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

    def infer_on_model(self, image):
        """
        Infer on the model hosted on Roboflow via the API call.

        Args:
            image (np.ndarray): Input image for inferencing
        
        Returns:
            result (json object): Inference result for the given image
        """
        _, img_encoded = cv.imencode('.jpg', image)

        # Make API request
        response = requests.post(self.model_endpoint, files={"file": img_encoded.tobytes()})
        result = response.json()

        return result

    @staticmethod
    def count_bolts_and_screws(data):
        # Parse the JSON data
        parsed_data = json.loads(data)
        
        # Extract the predictions
        predictions = parsed_data.get("predictions", [])
        
        # Count the number of "Bolt" and "Screw" class detections
        bolt_count = sum(1 for prediction in predictions if prediction.get('class') == 'Bolt')
        screw_count = sum(1 for prediction in predictions if prediction.get('class') == 'Screw')

        return bolt_count, screw_count

    def main(self):
        """Main logic for running the object detection code."""
        freeze_support()
        obj = DownloadFiles()
        obj.main()

        final_images, files = self.process_datasets()
        for image, file_path in zip(final_images, files):
            result = self.infer_on_model(image)
            bolt_count, screw_count = self.count_bolts_and_screws(json.dumps(result))
            logger.info(f"INFO: Number of bolts detected: {bolt_count}, Number of Screws detected: {screw_count} for image {file_path.name}")


if __name__ == "__main__":
    obj = CountObjects()
    obj.main()
