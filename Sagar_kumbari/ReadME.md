## Task 1 : Counting and detecting using non-AI technique ##
The steps used in exeuting the project are as follows:
 * Read the image through opencv
 * Converting the image to grayscale
 * Blurring the image to reomove the noise in the image. Applying the best guassian blur on the image.
 * Convertingt the image to binary image. where pixels with intensity above 60 are set to 0 and and pixels below 60 are set to be 255 
 * The next step is the morphological opertions to clear the small holes in the binary image.
 * The next step is to find the contours in the binary image.
 * Filetring the countours based on the size.
 * Draw contours on original image and the count the contours.
