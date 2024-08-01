import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageEnhance
import supervision as sv
from ultralytics import YOLOWorld, YOLO



CONFIG = {
    'custom': {'iou': 0.4, 'conf': 0.05, 'imgsz': 1280, 'max_det': 400},
    'grounding_dino': {'box_threshold': 0.1, 'text_threshold': 0.1},
    'yoloworld': {'conf': 0.0001, 'max_det': 400, 'iou': 0.2},
}

def is_corner(xyxy: np.ndarray, img_width: int, img_height: int) -> bool:
    '''
    Check if the rectangle is touching any edge of the image
    '''
    x1, y1, x2, y2 = np.intp(xyxy)
    return x1 == 0 or y1 == 0 or x2 == img_width or y2 == img_height

def remove_vignette_effects(results: sv.Detections, img_width: int, img_height: int) -> sv.Detections:
    '''
    Check if any detected bounding box touches the image edges and remove them.
    '''
    xyxys = []
    confidences = []
    class_ids = []

    for (xyxy, _, confidence, class_id, _, _) in results:
        if not is_corner(xyxy, img_width, img_height):
            xyxys.append(xyxy)
            confidences.append(confidence)
            class_ids.append(class_id)

    return sv.Detections(xyxy = np.stack(xyxys, axis=0),
                         confidence = np.array(confidences),
                         class_id = np.array(class_ids))


def detect_using_yoloworld(image_files: List[Path], use_slicer: bool = False) -> List[Tuple[Image.Image, int]]:
    '''
    Predict the bounding box of objects using YoloWorld model - yolov8x-worldv2
    using a text prompt
    '''
    all_results = []
    model = YOLOWorld('checkpoints/yolov8x-worldv2.pt')
    model.set_classes(['small_part'])

    box_annotator = sv.BoxAnnotator(thickness=5)
    if use_slicer:
        def slicer_callback(image_slice: np.ndarray) -> sv.Detections:
            result = model.predict(image_slice, agnostic_nms=True, 
                    conf=CONFIG['yoloworld']['conf'], verbose=False)[0]
            return sv.Detections.from_ultralytics(result)
        slicer = sv.InferenceSlicer(callback=slicer_callback, slice_wh=(640, 640),
                                overlap_ratio_wh=(0.2, 0.2),
                                overlap_filter_strategy=sv.OverlapFilter.NON_MAX_SUPPRESSION)

    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')
        if use_slicer:
            results = slicer(np.array(image))
            results = results.with_nms(0.1, True)
        else:
            size = max(image.size)
            results = model.predict(image, conf=CONFIG['yoloworld']['conf'], 
                                    imgsz=size, iou=CONFIG['yoloworld']['iou'], 
                                    agnostic_nms=True, max_det=CONFIG['yoloworld']['max_det'],
                                    verbose=False)[0]
            results = sv.Detections.from_ultralytics(results)
            results = results.with_nmm(0.3, True)

        results.class_id = np.zeros_like(results.class_id)
        results = remove_vignette_effects(results, image.size[0], image.size[1])
        annotated_image = box_annotator.annotate(image, results)

        all_results.append((annotated_image, len(results)))

    return all_results


def detect_using_finetuned_yolo(image_files: List[Path], use_slicer: bool=False) -> List[Tuple[Image.Image, int]]:
    '''
    Predict using a custom trained model - yolov8n_allc
    Then perform non-maximum merging to remove overlapping bounding boxes.
    And then remove the detected bounding boxes that touch the image edges, 
    since they are false positives because of vignette.
    '''
    all_results = []
    model = YOLO('pretrained/yolov8n_allc/best.pt')


    box_annotator = sv.BoxAnnotator(thickness=5)
    if use_slicer:
        def slicer_callback(image_slice: np.ndarray) -> sv.Detections:
            result = model.predict(image_slice, agnostic_nms=True, iou=0.5, 
                                   conf=0.1, verbose=False, augment=True)[0]
            return sv.Detections.from_ultralytics(result)
        slicer = sv.InferenceSlicer(callback = slicer_callback, slice_wh=(1920, 1920),
                                    overlap_filter_strategy=sv.OverlapFilter.NON_MAX_SUPPRESSION)


    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)
        size = max(image.size)

        if use_slicer:
            results = slicer(np.array(image))
            results = results.with_nms(0.1, True)
        else:
            # The max detections is set to 400 since there are more than 300 objects in some images.
            results = model.predict(image, conf=CONFIG['custom']['conf'], imgsz=size, 
                                    max_det=CONFIG['custom']['max_det'], 
                                    agnostic_nms=True, iou=CONFIG['custom']['iou'], 
                                    augment=True, verbose=False)[0]
            results = sv.Detections.from_ultralytics(results)
            results = results.with_nmm(0.3, True)

        # Make all class ids same, since we only care about the objects presence.
        results.class_id = np.zeros_like(results.class_id)
        results = remove_vignette_effects(results, image.size[0], image.size[1])
        annotated_image = box_annotator.annotate(image, results)
        # annotated_image.show()

        all_results.append((annotated_image, len(results)))

    return all_results



def main(args: argparse.Namespace) -> None:
    '''
    Read images from the input folder and save the results in the output folder.
    '''
    folder = args.input
    image_files = sorted(folder.glob('*.jpg'))

    output_folder = Path(args.output, 'AI', folder.name)
    output_folder.mkdir(parents=True, exist_ok=True)

    all_results = detect_using_finetuned_yolo(image_files, use_slicer=False)

    with open(output_folder / 'results.txt', 'w') as f:
        pass

    for (image_file, (output, count)) in zip(image_files, all_results):
        output.save(output_folder / image_file.name)
        with open(output_folder / 'results.txt', 'a') as f:
            f.write(f'{image_file.name}: {count}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect screws, bolts and nuts in images')
    parser.add_argument('--input', '-i', type=Path, help='Path to the folder containing images', required=True)
    parser.add_argument('--output', '-o', type=Path, help='Path to the folder to save the results', required=True)
    args = parser.parse_args()
    main(args)