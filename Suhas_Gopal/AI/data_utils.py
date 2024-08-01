from pathlib import Path
import json
import shutil

import cv2
import numpy as np


def obb_to_aabb(x, y, w, h, r):
    rect = cv2.RotatedRect((x, y), (w, h), -r * 180 / np.pi)
    oriented_corners = cv2.boxPoints(rect)
    oriented_corners = np.intp(oriented_corners)

    min_x, min_y = oriented_corners[:, 0].min(), oriented_corners[:, 1].min()
    max_x, max_y = oriented_corners[:, 0].max(), oriented_corners[:, 1].max()

    return min_x, min_y, max_x, max_y


def visualise_dataset(data_folder: Path):
    images_folder = data_folder / 'images'

    train_json = data_folder / 'mvtec_screws_train.json'

    with open(train_json, 'r') as f:
        train_data = json.load(f)


    image_data = {}
    for image in train_data['images']:
        image_data[image['id']] = image

    for annotation in train_data['annotations'][:5]:
        image_id = annotation['image_id']
        image = image_data[image_id]

        image_path = images_folder / image['file_name']
        img = cv2.imread(str(image_path))

        y, x, w, h, r = annotation['bbox']
        rect = cv2.RotatedRect((x, y), (w, h), -r * 180 / np.pi)
        oriented_corners = cv2.boxPoints(rect)
        oriented_corners = np.intp(oriented_corners)
        
        min_x, min_y = oriented_corners[:, 0].min(), oriented_corners[:, 1].min()
        max_x, max_y = oriented_corners[:, 0].max(), oriented_corners[:, 1].max()
        img = cv2.drawContours(img, [oriented_corners], 0, (0, 255, 0), 2)
        img = cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

        cv2.imshow('image', cv2.resize(img, (640, 480)))
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def format_annotations(data: dict):
    all_image_data = {}
    for image in data['images']:
        all_image_data[image['id']] = image

    for annotation in data['annotations']:
        image_id = annotation['image_id']
        image_data = all_image_data[image_id]

        y, x, w, h, r = annotation['bbox']
        min_x, min_y, max_x, max_y = obb_to_aabb(x, y, w, h, r)

        x_center = (min_x + max_x) / 2
        y_center = (min_y + max_y) / 2
        width = max_x - min_x
        height = max_y - min_y

        image_data['annotations'] = image_data.get('annotations', [])
        image_data['annotations'].append({
            'bbox': [int(x_center), int(y_center), int(width), int(height)],
            'category_id': annotation['category_id']
        })

    
    return all_image_data

def write_image_and_annotations(images_folder: Path, all_image_data: dict, 
                                output_images_folder: Path, output_annotations_folder: Path):
    output_images_folder.mkdir(exist_ok=True, parents=True)
    output_annotations_folder.mkdir(exist_ok=True, parents=True)

    for image_id, image_data in all_image_data.items():
        image_path = images_folder / image_data['file_name']
        output_image_path = output_images_folder / image_data['file_name']
        shutil.copy(str(image_path), str(output_image_path))

        for annotation in image_data.get('annotations', []):
            x_center, y_center, width, height = annotation['bbox']
            category_id = annotation['category_id'] - 1

            x_center /= image_data['width']
            y_center /= image_data['height']
            width /= image_data['width']
            height /= image_data['height']

            output_label_path = output_annotations_folder / f'{Path(image_data["file_name"]).stem}.txt'

            with open(output_label_path, 'a') as f:
                f.write(f'{category_id} {x_center} {y_center} {width} {height}\n')




def mvtec_to_yolov8(data_folder: Path, output_folder: Path):
    images_folder = data_folder / 'images'

    train_json = data_folder / 'mvtec_screws_train.json'
    with open(train_json, 'r') as f:
        train_data = json.load(f)

    train_data = format_annotations(train_data)

    val_json = data_folder / 'mvtec_screws_val.json'
    with open(val_json, 'r') as f:
        val_data = json.load(f)
    val_data = format_annotations(val_data)

    output_folder.mkdir(exist_ok=True, parents=True)
    
    write_image_and_annotations(images_folder, train_data, 
                                output_folder / 'images' / 'train', 
                                output_folder / 'labels'/ 'train')
    write_image_and_annotations(images_folder, val_data,
                                output_folder / 'images' / 'val',
                                output_folder / 'labels' / 'val')

    

    


if __name__ == '__main__':
    mvtec_to_yolov8(Path('../data/MVTec_Screws'), Path('../data/MVTec_Screws/yolov8'))