# Object Counting — AI Engineer Assessment

## Project structure

- `AI/` — Grounding DINO zero-shot object detection.
- `Non_AI/` — OpenCV-based classical computer vision baseline.
- `evaluation/` — Model comparison and verified evaluation results.

## AI approach

Grounding DINO (`IDEA-Research/grounding-dino-tiny`) detects
screws and bolts using the text prompt `screw. bolt.`.

Inference configuration:
- Box threshold: 0.20
- Text threshold: 0.20

## Non-AI approach

The OpenCV baseline uses image preprocessing, foreground extraction,
distance transform, watershed segmentation, and region filtering.

## Evaluation

The verified evaluation covers four images with manually verified
ground-truth counts.

| Method | Aggregate count accuracy |
|---|---:|
| OpenCV V2 | 64.52% |
| Grounding DINO | 85.48% |

Grounding DINO reduced aggregate absolute count error by 59.1%
compared with the OpenCV baseline on this four-image sample.

The evaluation sample is limited to four verified images.
The reported accuracy must not be interpreted as full-dataset accuracy.
The 95% target was not achieved on this verified sample.

## Important implementation note

AI masks are rectangular detection-region masks derived from
bounding boxes. They are not pixel-level segmentation masks.

## Execution

Open the corresponding notebook in Google Colab and execute its
cells in order. The AI notebook downloads its pretrained model
when executed. The dataset is not included in this submission.

Install method-specific dependencies using the requirements file
inside the corresponding folder.
