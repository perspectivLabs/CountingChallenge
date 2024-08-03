# python demo_examples/generic_inference.py --token <your_token> 
from .config import token,vis_dir,box_threshold,DATA_ROOT
from trex import TRex2APIWrapper, visualize
import numpy as np
import os
from PIL import Image
import cv2
import time

def plot_boxes(image, boxes):
    """
    Plots boxes on the given image.
    
    Parameters:
    image (numpy array): The image on which to plot the boxes.
    boxes (list of lists): A list of boxes, where each box is represented as [startx, starty, endx, endy].
    
    Returns:
    numpy array: The image with boxes plotted.
    """
    # Make a copy of the image to avoid modifying the original one
    img_copy = image.copy()
    
    # Loop through each box and draw it on the image
    for box in boxes:
        startx, starty, endx, endy = box.astype(int)
        # Draw the rectangle on the image
        cv2.rectangle(img_copy, (startx, starty), (endx, endy), (0, 255, 0), 2)
    
    return img_copy


def infer(target_image):
    "take input target image path and use the prompt example"
    "output mask overlayed image and count of objects"
    "save the outputs to outputs directory"
    "images sub directory and counts"
    trex2 = TRex2APIWrapper(token)
    prompts = [
        {
            "prompt_image": str(DATA_ROOT / "data/raw/ScrewAndBolt_20240713/20240713_194551.jpg"),
            "rects": [[1424, 1847,1500, 1928],[1716, 2201,1794, 2283]],
        },
        {
            "prompt_image": str(DATA_ROOT / "data/raw/ScrewAndBolt_20240713/20240713_194206.jpg"),
            "rects": [[1757, 2233,1827, 2310],[1660, 2413,1719, 2507]],
        },
    ]
    result = trex2.generic_inference(target_image, prompts)
    # filter out the boxes with low score
    scores = np.array(result["scores"])
    labels = np.array(result["labels"])
    boxes = np.array(result["boxes"])
    filter_mask = scores > box_threshold
    filtered_result = {
        "scores": scores[filter_mask],
        "labels": labels[filter_mask],
        "boxes": boxes[filter_mask],
    }
    # visualize the results
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    image = cv2.imread(target_image)
    # image = visualize(image, filtered_result, draw_score=True)
    image = plot_boxes(image,filtered_result['boxes'])
    # image.save(os.path.join(vis_dir, f"generic.jpg"))
    cv2.imwrite(os.path.join(vis_dir, f"generic.jpg"),image)
    return image,filtered_result,boxes





def retry_function(func, max_retries=3, delay=10, *args, **kwargs):
    """
    Tries to execute a function and retries it if it fails.

    Parameters:
    - func: The function to be executed.
    - max_retries: Maximum number of retries if the function fails.
    - delay: Delay between retries in seconds.
    - *args: Arguments to pass to the function.
    - **kwargs: Keyword arguments to pass to the function.

    Returns:
    - The return value of the function if successful.
    
    Raises:
    - Exception if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            a,b,c = func(*args, **kwargs)
            return a,b,c
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise