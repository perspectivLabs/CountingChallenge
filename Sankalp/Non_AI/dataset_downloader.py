"""Utility script for downloading files from google drive."""

from pathlib import Path
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from multiprocessing import Process, freeze_support


class DownloadFiles:
    """Class for downloading files from google drive"""

    def __init__(self) -> None:
        """Initialize the required parameters."""
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication.
        self.drive = GoogleDrive(gauth)

        # ID of the shared Google Drive folders
        self.dataset_1_id = "1E75YmFnVPmMPGARjq_O1X2dpidiGvgp6"
        self.dataset_2_id = "1Hp55bMW33A-e9LgXwvdZuiRzO6UKTGnZ"

        Path.mkdir(Path.cwd() / "dataset/dataset_1", exist_ok=True, parents=True)
        Path.mkdir(Path.cwd() / "dataset/dataset_2", exist_ok=True, parents=True)

    def get_list_of_files(self):
        """List all files in the folder"""
        dataset_1 = self.drive.ListFile({'q': f"'{self.dataset_1_id}' in parents and trashed=false"}).GetList()
        dataset_2 = self.drive.ListFile({'q': f"'{self.dataset_2_id}' in parents and trashed=false"}).GetList()

        return (dataset_1, dataset_2)

    @staticmethod
    def download_files(dataset, id):
        """Worker process for downloading the dataset."""
        for file in dataset:
            print(f'Downloading {file["title"]} to dataset/dataset_{id}')
            download_path = f"dataset/dataset_{id}/{file['title']}"
            file.GetContentFile(download_path)

    def main(self):
        """Main logic for downloading the files."""
        dataset_1, dataset_2 = self.get_list_of_files()
        dataset_1_process = Process(target=self.download_files, args=(dataset_1, 1))
        dataset_1_process.start()

        dataset_2_process = Process(target=self.download_files, args=(dataset_2, 2))
        dataset_2_process.start()

        dataset_1_process.join()
        dataset_2_process.join()


# For standalone execution
if __name__ == "__main__":
    freeze_support()
    obj = DownloadFiles()
    obj.main()
