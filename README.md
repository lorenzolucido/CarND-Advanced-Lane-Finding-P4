
# **Advanced Lane Finding Project**
#### _Lorenzo's version_
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calib]: ./calibration.png "Calibration Step"
[undistorted]: ./undistorted.png "Undistorted Example"
[perspective]: ./perspective_transform.png "Perspective Transform Example"
[colgrad]: ./color_gradient.png "Color and Gradient Example"
[full_preprocessing_pipeline]: ./full_pipeline.png "Full Preprocessing Example"
[lanes_warped]: ./lanes_warped.png "Lane Finding Example (warped)"
[lanes_unwarped]: ./lanes_unwarped.png "Lane Finding Example (unwarped)"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

#### Link to the [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points.


---

### General notes

For this project, I have decided to try and have all the code fit on a single Jupyter notebook called [P4-Advanced-Lane-Finding](./P4-Advanced-Lane-Finding.ipynb) (+ [HTML](./P4-Advanced-Lane-Finding.html) version). I tried my best to write concise code so that the notebook remains compact. I believe this actually makes it more readable and easy to understand than a code base scattered across several `.py` files, which in my opinion is slightly _overkill_ for a project of this size.  
For this, I have used a functional programming approach, using functions such as `compose` and `pipe` from the [`toolz`](http://toolz.readthedocs.io/en/latest) library, this library comes by default with Anaconda.

### Camera Calibration

The code for this step is contained in the **Step 1: Calibrating the camera** of the Jupyter notebook.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][calib]

### Pipeline (single images)

Let's first look at an example undistorted image:
![alt text][undistorted]

From here, in order to better identify the lanes on the road, I used a combination of color and gradient thresholds to generate a binary image. This is described in _Part 2 & 3_ of the **Step 2: A clean bird's eye view on the situation** within the Jupyter notebook. Here's an example of my output for this step.  
- _for gradient_: I used a combination of `x` and `y` gradient magnitude thresholding
- _for color_: I used both the saturation (S) of the HLS image and the value (V) of the HSV image

![alt text][colgrad]

Then, in order to convert this _cleaned up_ thresholded image to a bird's eye view, I used perspective transform. The code for this perspective transform is defined in the `perspective_pipeline`, in the **Step 2: A clean bird's eye view on the situation** | _part 1: Perspective Transform_.  I chose the hardcode the source and destination points in the following manner:

```python
height = 720
width = 1280
lower_width_offset = 420
higher_width_offset = 50
lower_height = 680
higher_height = 450

src_vertices = [
    (width/2 - lower_width_offset,           lower_height),
    (width/2 - higher_width_offset,          higher_height),
    (width/2 + higher_width_offset,          higher_height),
    (width/2 + lower_width_offset,           lower_height)
]
src = np.float32(src_vertices)

dst = np.float32([
  (500,700),
  (500,80),
  (900,80),
  (900,700)
])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 220, 680      | 500, 700        |
| 590, 450      | 500, 80      |
| 690, 450     | 900, 80      |
| 1060, 680      | 900, 700        |

I verified that my perspective transform was working as expected by drawing the `src` points onto a test image and verified that the lines appear parallel in the warped image
(by construction, we know that our warped `dst` points are forming a rectangle).

![alt text][perspective]

I combined the color thresholding, gradient thresholding and perspective transform to finally get the `full_preprocessing_pipeline`. We can now clearly see the lanes, here is an example of the **binary images** that are output by this pipeline:
![alt text][full_preprocessing_pipeline]

##### Lane Finding

From here, I identified the lanes using a _sliding window polynomial fit_, i.e. in order to get a point for my second order polynomial, I split each image into 9 horizontal slices and counted the number of **1**'s on the **binary image** vertical axis. I repeat that process twice, once with the left part of the image in order to get the left lane, and once on the right part of the image for the right lane.
The code for this quadratic fit is in **Step 3: Finding the lanes** in the notebook.

We can see below how this polynomial fit looks on the warped image:
![alt text][lanes_warped]
And here on the unwarped (original) image:
![alt text][lanes_unwarped]

##### Curvature and safety checks

Then, in **Step 4: Curvature of the road and sanity checks**, I defined the functions `curvature()` and `fits_are_valid()`.
Details on how I computed the curvature are in the notebook.
For the safety checks, I mainly compared the two fitted lines (left and right) and input some constraints:
- they should not be separated by more than a specific distance (450 pixels, or 3.7m)
- they should have roughly the same slope (i.e. be parallel)
- they should have roughly the same curvature or second-order coefficient
If those three criteria are not all matched, I filter out the 2 fitted lines.

---

### Pipeline (video)

Here's a [link to my video result](./project_video_output.mp4)
_Note: I have tried to optimize the code for readability, not performance, so computing the output video takes roughly 30 minutes on my computer._

---

### Discussion

While the project walked me through all the main components of lane finding, almost all parameters (gradient/color thresholding + safety checks) have been fitted in-sample, i.e. the images and the video from which the images have been taken serve for both training and testing set. Right now, I do not know if my pipeline would work in different environments such as very intense sunlight, or during nighttime, or rain.

In order to have a more robust pipeline, we should **define a loss function** and try to optimize the parameters, given some data. This might not be convex though, in which case a technique like [Sebastian's twiddle](https://www.youtube.com/watch?v=2uQ2BSzDvXs) might help find some local minimum.
