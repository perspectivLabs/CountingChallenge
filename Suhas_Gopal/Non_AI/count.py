import argparse
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np


NUT_IMAGES = {'20240713_194541.jpg', '20240713_194551.jpg', '20240713_194606.jpg', '20240713_194621.jpg',
              '20240713_194630.jpg'}


def show_image(name: str, image: np.ndarray) -> None:
    '''
    Helper function to display a resized small version of the image
    '''
    max_size = 800

    orig_height, orig_width = image.shape[:2]
    aspect_ratio = orig_width / orig_height

    if orig_height > orig_width:
        new_height = max_size
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = max_size
        new_height = int(new_width / aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height))
    cv2.imshow(name, resized_image)


def remove_vignetting(binary: np.ndarray, contours: List[np.ndarray], 
                      hierarchy: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    '''
    Remove the effects of vignetting by removing the contours that are at the corners of the image
    This is a hacky way of removing only at the image edges, 
    but simpler than having to estimate the vignetting mask.
    '''
    rects = [cv2.boundingRect(contour) for contour in contours]
    to_remove = set([i for i in range(len(contours)) if is_corner(rects[i], binary.shape[1], binary.shape[0])])

    for i in to_remove:
        cv2.drawContours(binary, contours, i, 0, -1)
    
    contours = [contours[i] for i in range(len(contours)) if i not in to_remove]
    hierarchy = hierarchy[0]
    hierarchy = [hierarchy[i] for i in range(len(hierarchy)) if i not in to_remove]

    return binary, contours, hierarchy


def remove_noise(binary: np.ndarray, kernel_size:int=5) -> np.ndarray:
    '''
    Perform morphological opening and closing to remove small islands and holes in the binary image
    '''
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary

def is_corner(rect: List[int], img_width: int, img_height: int) -> bool:
    '''
    Check if the rectangle is touching any edge of the image
    '''
    x, y, w, h = rect
    return x == 0 or y == 0 or x + w == img_width or y + h == img_height

def detect_circles(image: np.ndarray, gray: np.ndarray, plot:bool=False):
    '''
    Detect small circles in the contoured image (gray) using Hough Transform.
    These circles are drawn on the original image (image) and returned, when plot is True
    '''
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1, 5, param1=100,
                            param2 = 20, minRadius = 1, maxRadius = 10)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        if plot:
            cimg = image.copy()
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
            return circles[0], cimg

    return circles

def filter_inner_contours(contours: List[np.ndarray], hierarchy: np.ndarray) -> List[np.ndarray]:
    '''
    Filter only the inner contours (contours with no children) from all the contours of the image, 
    using the hierarchy information from cv2.findContours.
    '''
    inner_contours = []
    for i in range(len(contours)):
        if hierarchy[i][2] < 0:
            inner_contours.append(i)

    return [contours[i] for i in inner_contours]

def process_nuts(image: np.ndarray, binary: np.ndarray, 
                 contours: List[np.ndarray], hierarchy: np.ndarray) -> Tuple[np.ndarray, int]:
    '''
    Process the image to detect nuts. 
    This is done by detecting small circles in the inner contours of the image, which are the holes of the nuts.
    The hough transform is used on a resized version, to make computation faster. 
    These circles are drawn on the original image and returned after resizing to original size.
    '''
    contours = filter_inner_contours(contours, hierarchy)
    contoured = cv2.drawContours(np.zeros_like(binary), contours, -1, 255, 3)
    circles, marked_image = detect_circles(cv2.resize(image, (0, 0), fx=0.5, fy=0.5),
                   cv2.resize(contoured, (0, 0), fx=0.5, fy=0.5), plot=True)
    marked_image = cv2.resize(marked_image, (0, 0), fx=2, fy=2)
    
    return marked_image, len(circles)

def process(image: np.ndarray, binary: np.ndarray, contours: List[np.ndarray]) -> Tuple[np.ndarray, int]:
    '''
    Process images to detect screws and bolts.
    The binary image is separated into background (by dilating the thresholded image) 
    and foreground (by finding the distance transform of the binary and thresholding it) to ideally get patches
    that are at the center of each object.
    Then contours of these patches gives the no of objects and their bounding boxes are drawn on the original image.
    '''
    # sure background area
    kernel = np.ones((3, 3), np.uint8)
    background = cv2.dilate(binary, kernel, iterations=3)

    #foreground area
    kernel = np.ones((3, 3), np.uint8)
    dist = cv2.distanceTransform(background, cv2.DIST_L2, 3)
    dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    # show_image('Before erosion Distance Transform', dist)
    dist = (dist * 255).astype(np.uint8)
    # 100 works in most cases.
    _, dist = cv2.threshold(dist, 100, 255, cv2.THRESH_BINARY)
    # show_image('Distance Transform', dist)

    contours, _ = cv2.findContours(dist, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(contour) for contour in contours]
    
    output = image.copy()
    for rect in rects:
        cv2.rectangle(output, rect, (0, 255, 0), 8)
    
    # show_image('Marked Image', output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return output, len(contours)



def detect_and_count(image_path: str, kernel_size:int=5, is_nut_image:bool=False):
    '''
    Process the image to detect screws and bolts or nuts.
    Method - Read the image, convert to grayscale, threshold using OTSU's method, remove noise and vignetting.
    Perform different processing for nuts and screws/bolts, to get approximate centers of objects and 
    use it to count no of objects present.
    '''
    image = cv2.imread(image_path)
    image = cv2.GaussianBlur(image,(kernel_size,kernel_size),0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = remove_noise(binary, kernel_size=kernel_size)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    binary, contours, hierarchy = remove_vignetting(binary, contours, hierarchy)

    if is_nut_image:
        marked_image, count = process_nuts(image, binary, contours, hierarchy)
    else:
        marked_image, count = process(image, binary, contours)

    # print(f'Number of segments: {count}')
    # show_image('Marked Image', marked_image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return marked_image, count

def main(args: argparse.Namespace):

    folder = args.input
    images = sorted(folder.glob('*.jpg'))

    output_folder = Path(f'{args.output}/Non_AI/{folder.name}')
    output_folder.mkdir(parents=True, exist_ok=True)

    # overwrite the results file if it exists
    with open(output_folder / 'results.txt', 'w') as f:
            pass

    for image in images:
        marked_image, count = detect_and_count(str(image), is_nut_image=image.name in NUT_IMAGES)
        cv2.imwrite(str(output_folder / image.name), marked_image)
        with open(output_folder / 'results.txt', 'a') as f:
            f.write(f'{image.name}: {count}\n')





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect screws, bolts and nuts in images')
    parser.add_argument('--input', '-i', type=Path, help='Path to the folder containing images', required=True)
    parser.add_argument('--output', '-o', type=Path, help='Path to the folder to save the results', required=True)
    args = parser.parse_args()
    main(args)