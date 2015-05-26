ApertureSynth is a command line tool for registering and combining multiple photographs to form a single photograph with better properties. It can be used to:

1. Reduce noise by averaging, even in handheld photos.
2. Simulate a longer exposure than would otherwise be possible using a handheld camera.
3. Control the location of the focal plane, including tilt shift effects (though further optimisation is needed to handle this nicely.)

Note that aperturesynth is currently experimental and is definitely to be used at your own risk.


# Usage

1. Take a series of photos in burst mode.

2. If shooting in raw format, convert the raw files into a format handled by PIL, such as png, tiff or jpeg. Note that it's best to avoid noise reduction and sharpening at this point. You might have a series of photos 1.png, 2.png, 3.png.

3. Run the command line application on the series of photographs

        aperturesynth combine --out fused.tiff 1.png 2.png 3.png

    This will fuse the three png images and save the result to fused.tiff. The file 1.png will be the baseline image used to register 2.png and 3.png.

4. The baseline image (1.png) will appear in a window. Indicate the in focus regions by selecting the top left and bottom right of each rectangular focal patch. Consecutive pairs of points define each rectangular window, the last point of an odd number of points will be ignored. You can right click to undo a selection.

5. Press enter when done to begin the fusion process.

For additional help and to see all the options run:

    aperturesynth --help


# Installation

This software is written in Python, and currently requires a working Python implementation to run. If you have a working Python installation you should be able to install aperturesynth by running:

    pip install aperturesynth

Alternatively you can install from source. First clone the project from [the website](https://hames.id.au/software/aperturesynth) using fossil , then run the setup.py in the root directory to install the file.

    fossil clone https://hames.id.au/software/aperturesynth aperturesynth.fossil
    mkdir aperturesynth; cd aperturesynth
    fossil open ../aperturesynth.fossil
    python setup.py install

Either of these options should result in the installation of a commandline tool called aperturesynth.


# Changes

## 0.0.2
- The type of transform is now chosen based on the number of focal regions specified. This allows registration of only one or two points, instead of always requiring three or more.
- The commandline syntax has changed to allow multiple subcommands, including a new way of saving and loading the focal region locations to/from a file.

## 0.0.1
- Initial proof of concept.
