#include "ofMain.h"
#include "ofApp.h"

//========================================================================
int main( ){
#ifdef _USE_PER_PIXEL_SEGMENTATION
	ofSetupOpenGL(800,400,OF_WINDOW);			// <-------- setup the GL context
#else
    ofSetupOpenGL(1720,400,OF_WINDOW);			// <-------- setup the GL context
#endif
	// this kicks off the running of my app
	// can be OF_WINDOW or OF_FULLSCREEN
	// pass in width and height too:
	ofRunApp(new ofApp());

}
