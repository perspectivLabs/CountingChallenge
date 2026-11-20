
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
