

OpenFrameworks based machine vision application for tracking a human hand on a contrasting background, and to train a Support Vector Machine (SVM) to classify static hand poses.

This application requires the following addons:

 - ofxConvexHull
 - ofxCv
 - ofxGrt
 - ofxGui
 - ofx.JSON
 - ofxOpenCv
 - ofxOSC
 - ofxPerPixelSegment
 
To install them for your environment, Google their names, find their GitHub repository, and clone those repositories in .../of_v0.9.3_osx_release/addons

Steps for getting this project running:

 - install OpenFrameworks 0.93
 - install all the addons listed above
 
Then you have two options:
 
 - Clone this repository into .../of/apps/myApps/
 - Open 'ofxRGBHandTracK.xcodeproj'
 - Use the [OFXCodeMenu](https://github.com/openframeworks/OFXcodeMenu) to add all the addons to a the project
 - Build (make sure to build 'ofxRGBHandTracK' not 'openFrameworks')
 
or
 - use the openFrameworks Project builder to make a new project; include all the necessary addons
 - replace the three files inside the generated projects .../src folder with the ones from this repository
 
 
