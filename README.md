
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/window1.png
[image4]: ./output_images/window2.png
[image5]: ./output_images/window3.png
[image6]: ./output_images/snap1.png
[image7]: ./output_images/snap2.png
[image8]: ./output_images/test1.jpg
[image9]: ./output_images/frameheat.png
[image10]: ./output_images/heatmap.png
[image11]: ./output_images/labelbbox.png
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 4th code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are examples of `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then extracted HOG features for `vehicle` and `non-vehicle` to see the differences. Here is an example using the `Gray` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and fitted the LinearSVC model. In the first few iterations, I chose the parameters that strike a good balance between extract time and accuracy which was `orient = 12`, `pix_per_cell = 16`, `cell_per_block = 2` and `Color = YCrCb`. (NOTE: I also added raw color features with a spatial bin size of (16, 16) and 32 bins of histogram colors)

|   Orient |   Pix p/ Cell | Color   |   Time Train |   Accuracy |   Time Predict |   Time Extract |
|---------:|--------------:|:--------|-------------:|-----------:|---------------:|---------------:|
|        9 |            16 | RGB     |        42.6  |   0.967142 |       0.012137 |        130.713 |
|        9 |            16 | BGR     |        41.85 |   0.967243 |       0.020589 |        131.302 |
|       11 |            16 | YUV     |        33.37 |   0.983722 |       0.01314  |        131.399 |
|       11 |            16 | BGR     |        43.06 |   0.969051 |       0.012509 |        131.425 |
|        9 |            16 | YCrCb   |        33.06 |   0.981712 |       0.025503 |        133.672 |
|       12 |            16 | BGR     |        42.28 |   0.970659 |       0.019639 |        134.156 |
|       12 |            16 | YUV     |        33.45 |   0.984727 |       0.009258 |        134.423 |
|       11 |            16 | YCrCb   |        34.61 |   0.982918 |       0.014757 |        134.845 |
|        9 |            16 | YUV     |        33.06 |   0.982818 |       0.020851 |        135.855 |
|       11 |            16 | RGB     |        46.11 |   0.96875  |       0.006432 |        139.287 |
|       12 |            16 | YCrCb   |        35.08 |   0.987239 |       0.019432 |        140.27  |
|       11 |            16 | HSV     |        42.62 |   0.97297  |       0.007138 |        142.71  |
|       12 |            16 | RGB     |        54.55 |   0.970559 |       0.033296 |        145.178 |
|        9 |            16 | HSV     |        52.98 |   0.971764 |       0.008253 |        179.634 |
|       11 |            16 | HLS     |        56    |   0.968147 |       0.012969 |        180.886 |
|        9 |            16 | HLS     |        54.6  |   0.966941 |       0.011595 |        181.395 |
|       12 |            16 | HLS     |        55.85 |   0.973774 |       0.019097 |        185.866 |
|       12 |            16 | HSV     |        53.91 |   0.975482 |       0.008957 |        189.933 |
|        9 |             8 | YCrCb   |        70.57 |   0.986636 |       0.010155 |        246.019 |
|        9 |             8 | BGR     |        90.71 |   0.970458 |       0.022895 |        246.63  |
|        9 |             8 | YUV     |        48.15 |   0.98533  |       0.01017  |        248.322 |
|        9 |             8 | RGB     |        90.61 |   0.970659 |       0.015565 |        251.581 |
|       11 |             8 | BGR     |       102.54 |   0.967645 |       0.017035 |        257.226 |
|       11 |             8 | YCrCb   |        82.69 |   0.985229 |       0.01852  |        257.289 |
|       11 |             8 | YUV     |        81.81 |   0.984827 |       0.019124 |        259.366 |
|       12 |             8 | BGR     |       107.34 |   0.971564 |       0.016271 |        262.786 |
|       12 |             8 | YUV     |        87.54 |   0.986937 |       0.012293 |        264.437 |
|       11 |             8 | RGB     |       107.18 |   0.967645 |       0.022777 |        266.863 |
|       12 |             8 | HSV     |       123.43 |   0.981411 |       0.023489 |        268.37  |
|       12 |             8 | RGB     |       114.3  |   0.971463 |       0.013907 |        274.335 |
|       12 |             8 | YCrCb   |        89.26 |   0.987741 |       0.015327 |        292.278 |
|        9 |             8 | HSV     |       103.54 |   0.979803 |       0.015345 |        349.329 |
|        9 |             8 | HLS     |       108.07 |   0.977391 |       0.021115 |        349.538 |
|       11 |             8 | HSV     |        99.19 |   0.978195 |       0.012065 |        364.053 |
|       11 |             8 | HLS     |       126.23 |   0.977391 |       0.018405 |        368.231 |
|       12 |             8 | HLS     |       130.8  |   0.980004 |       0.023446 |        370.87  |

However, I found the accuracy not adequate for processing the video. Then, I noticed that with increasing number of `orient`, accuracy increase. So I settled on `orient = 15`, `pix_per_cell = 16`, `cell_per_block = 2` and `Color = YcrCb` which achieves accuracy of .992. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `YCrCb` color space and HOG features with parameters `orientations=15`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`, and raw color with `spatial_size = (16, 16)` and histogram of color features with `nbins = 32`. The code is in the 9th code cell. I also augmented the data by flipping all images, and I also tried to handle the time series in `vehicle` by performing a histogram comparison and assigning different classes for different cars, and using `sklearn.model_selection.StratifiedShuffleSplit` to get a good split that distribute the classes evenly.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The method to search by sliding window is located in the 10th and 11th code cells. The method `find_cars` is duplicated from the course to do a sub sampling of the HOG features within a x, y region of interest. However, I found that subsampling HOG features produced much worse classifying performance compared to a sliding window search, so I opted to use sliding window search without subsampling for HOG features.

I also used different scales of window to search for cars. I used smaller scales - 0.75, 0.8, 1 - to mainly search in the middle of the image for small cars and for cars right in front of us. Then I used bigger scales - 1, 1.5, 2 - to search on left and right sides for bigger cars. The scale of 1 tends to work well for most sizes of car so I tried to cover most spaces with scale of 1. For the scale of 2, I triple weighted the boxes because the bigger the car the less boxes can overlap.

I let the window overlap a lot 75% on x axis and 50% on y axis to help me phase out false positives with the heatmap approach.

The 12th code cell shows images of searching windows at different scales to optimize for depth of field, for example:

![alt text][image3]
![alt text][image4]
![alt text][image5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 5 different scales with a 64 x 64 window, and 0.75, 0.8, 1, 1.5, 2. Here are some example images:

![alt text][image6]
![alt text][image7]
![alt text][image8]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

To smooth the boxes, I remembered 10 contingous frames and used all the positive detections from all the frames for the pipeline. This is also hoping to reduce false positive since we will overlay more true positive over many frames.

In order to help the pipeline run faster on video, I only sampled 1 out of 5 frames

This code is located in the 14th and 15th cells.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are 10 frames and their corresponding heatmaps:

![alt text][image9]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all 10 frames:
![alt text][image10]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image11]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had a really hard time finding the right parameters for the classifier to produce satisfactory performance. Despite achieving .98+ accuracy on the classifier, when using it on the video, the results were terrible. However, I then realized that using sliding window search without subsampling HOG features would yield better performance, I managed to get satisfactory performance.

But then, there were still a lot of problems with finding the right window scales and overlap to reduce false positives when thresholding. I had to tune the windows carefully to get satisfactory performance on the video.

In the video, the pipeline performs fairly well to detect the cars, but it struggles with oncoming traffic on the far left side of the video, which resulted in orphaned bounding boxes because the pipeline remembered true detections from previous frames and compounded with false positives in later frames but the car has already moved out of the bounding box. However, the pipeline recovered nicely.

As seen in the overlay in the top right corner, the pipeline actually detects a lot of false positives, and evident from the first frame of the video which has a colorful tree on the left, the pipeline fails to detect false positives and draw bounding boxes where cars are not. A better classifier, or more comprehensive training data, will help us in this respect.

The pipeline also runs really slow since we are not subsampling HOG features and search over a big number of windows. A better classifier or a better set of windows can help us here. Right now the pipeline cannot go into production because it cannot practically perform in real time.

The pipeline also fails to detect when cars overlap each other. It is a hard problem, but perhaps we could simulate the environment and use existing information about the cars like speed, previous position and lane position to juxtapose the positions when cars overlap.
