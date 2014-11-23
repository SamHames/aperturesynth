aperturesynth
=============

A command line tool for registering and fusing a series of handheld photographs into a single image. Doing so can reduce the impact of noise, allow for the simulation of longer exposures and allow for control of the focal plane through the image.


Usage
-----

1. Take a series of photos in burst mode.

2. If shooting in raw format, convert the raw files into tiff, png or jpeg format. Note that it's best to avoid noise reduction and sharpening at this point. You might have a series of photos 1.png, 2.png, 3.png.

3. Run the command line application on the series of photographs

   ::

    aperturesynth --out fused.tiff 1.png 2.png 3.png

   This will fuse the 3 .png images and save the result to fused.tiff. The file 1.png will be the baseline image used to register 2 and 3.

4. The baseline image (1.png) will appear in a window. Indicate the in focus regions by selecting the top left and bottom right of each rectangular focal patch. Consecutive pairs of points define each rectangular window, the last point of an odd number of points will be ignored. You can right click to undo a selection.

5. Press enter when done to begin the fusion process.


Tips
----

Reducing Noise
^^^^^^^^^^^^^^

If your aim is to reduce noise try a short high speed burst first as it is least likely to have problems in registration. The shutter speed will need to be high enough to avoid camera shake, otherwise the output will just look the average of all the blur. Even combining 3 or 4 photos can make a noticable improvement in noise in the registered output.

Long Exposures
^^^^^^^^^^^^^^

Try taking a very long series of jpeg images, staying as still as possible. Note that if the registration patch is occluded during the series of images, it won't be possible to register that frame, and you will get strange outliers.


Tilt Shift
^^^^^^^^^^

The windows you choose to be in focus define the focal plane that will be in focus in the final fused image. Note that this means there has to be some distinguishable feature in the photo to define the focal plane.

To get the best out of focus simulation, you need a large number of images (20+) and some camera motion from frame to frame. The more the camera moves from photo to photo the narrower the depth of field you can simulate. Be careful though: if you move the camera too much the out of focus areas will look very rough. Try moving the camera in a circle around 1 cm as a starting point.

