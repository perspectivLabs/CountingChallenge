# CountingChallenge

## Task definiton
* Count the number of items in the image and overlay masks for the same.
* Accuracy should be > 95%

## Task list
1) Achieve the task definition using any Non-AI techniques (ex. OpenCV, etc)
2) Achieve the task definition using any AI techniques

## Note
* All tasks from the Task List are expected to be completed
* Requirement files are expected to be added.
* Colab/Kaggle notebooks can be used, but the ```ipynb``` file must be added.
* The dataset is available [here](https://drive.google.com/drive/folders/1TuM4CgGI3WBqOHNFjuzjaEzLOH5Yw43_?usp=sharing).
* Create a root folder with your name in the git.
* There has to be 2 folders viz. "AI" and "Non_AI" containing solutions to the respectiv tasks within the above root folder.

## Steps to submit the solutions
* Fork this repository.
* Push the solution into the folders sepecified.
* Raise a PR with your name as per resume to this repository from your forked repository.
* Cleanup unwanted files by putting the path in the .gitignore.
* No need of including the data files into GIT.
* Write or provide the training method used for AI training purpose if any. 

# My take on the challenge

The dataset is small with 25 images and no labels. And the images in the dataset have very favourable conditions to segment out the screws and nuts. 

## Favourable conditions:
* Black items against light plain background
* Each image has either only nuts or screws
* No overlapping of items over each other

It is just easy to separate the foreground from the background

## My attempt to solve it.
I have tried multiple techniques and methods to test and see the results. Since the images don't have ground truth label (i could have counted items for each image, but i did not consider it for time constraints) i am relying on visual inspection of the masks overlay over the image and the counts i printed on the image in the corner. 

In place of results, i will put my observation on the results for both AI and Non_AI techniques.

If i had to go for a production use case i would choose Non_AI technique as it performs pretty well with minimum compute requirements. My opinion is assuming the deployment environment will carry the same favourable conditions as in the images. If my assumption does not hold then I would use the AI technique, since i can simulate multiple scenarios in synthetic dataset and train an efficient small model to be effective against indeterministic scenarios. 

## Results and implementation
I have described the implementation of my approach and results in detail inside the AI and Non_AI folders with their own readme files.

I am also attaching the common requirements.txt in the root folder which will work against both AI and Non_AI code
