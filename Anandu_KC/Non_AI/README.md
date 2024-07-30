# NON AI BASED COUNTING
## Description

This Streamlit application, titled **Non AI Based Counting**, allows users to upload an image and count the number of screws using non-AI techniques. The application leverages OpenCV for image processing and displays both the binary thresholded image and the detected screws with contours side by side. Users can adjust the minimum area threshold to fine-tune the detection process.

## Features

- Upload an image for screw counting.
- Display the binary thresholded image.
- Display the detected screws with contours.
- Adjustable minimum area threshold for screw detection.
- Real-time updating of results as the threshold is adjusted.

## How It Works

The application uses the following steps to count the screws:

1. **Image Upload**: The user uploads an image file.
2. **Image Processing**: The uploaded image is converted to grayscale and blurred to reduce noise.
3. **Binary Thresholding**: The image is thresholded to create a binary image with white screws on a black background.
4. **Contour Detection**: Contours are detected in the binary image.
5. **Screw Counting**: Contours are filtered based on area to count the screws.
6. **Results Display**: The binary thresholded image and the image with detected contours are displayed side by side. The screw count is shown to the right of the images.

## Example Code

In addition to the Streamlit application, an example code file named `example_code.py` is provided. This file contains the basic code to count objects using non-AI techniques with OpenCV. It demonstrates how to process an image, apply binary thresholding, detect contours, and count objects based on their area.

## Installation

- Create a virtual environment: (recommended)

Linux/macOS:

```bash
  # Create a virtual environment
    python3 -m venv venv

  # Activate the virtual environment
    source venv/bin/activate
  
```

 Windows:

```powershell
  # Create a virtual environment
    python -m venv venv

  # Activate the virtual environment
    venv\Scripts\activate
  
```

- install required libraries

```bash
  pip install -r requirements.txt
```

- Run the script:

```bash
  streamlit run app.py
```

- Run the basic script:

```bash
  python3 example_code.py
```

## output


![Output Example](https://github.com/Anandukc/CountingChallenge/blob/main/Anandu_KC/Non_AI/Screenshot_nonAI.png)






