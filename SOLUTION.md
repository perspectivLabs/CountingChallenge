# CountingChallenge

## Task definiton
* Count the number of items in the image and overlay masks for the same.

## Task list
1) Achieve the task definition using any Non-AI techniques (ex. OpenCV, etc)
2) Achieve the task definition using any AI techniques

## Setup

1. Clone the repository:
    ```bash
    git clone git@github.com:perspectivLabs/CountingChallenge.git
    cd CountingChallenge
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the counter script for the data:
    ```bash
    python src/count.py --data data/ --debug --non_ai
    ```

### Options:
- `--non-ai`: To run the Non-AI Solution
- `--ai`: To run the AI Solution
- `--data`: Specify path to dataset directory
- `--debug`: Enable debug mode to view outputs