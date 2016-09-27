#pragma once

#include "ofMain.h"

#include "ofxOpenCv.h"
#include "ofxCv.h"
#include "ofxGui.h"
#include "ofxConvexHull.h"
#include "ofxOsc.h"
#include "ofxGrt.h"

#define HOST "localhost"
#define PORT 6448
#define ADDRESS "/test"

#define _USE_LIVE_VIDEO		// uncomment this to use a live camera
								// otherwise, we'll use a movie file

#define VID_WIDTH 160
#define VID_HEIGHT 120
#define VID_SPACE  20

//const float  VID_WIDTH  = 320;
//const float  VID_HEIGHT = 240;
//const float  VID_SPACE  = 20;

const int GRT_FEATURE_NUM = 5;
const int FEATURE_NUM = 5;

const int infoTextX = 20;
const int infoTextY = VID_HEIGHT + 20;



class ofApp : public ofBaseApp{

	public:
    
        void setup();
		void update();
		void draw();
		
		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
    
    
        void sendOSC(float param[FEATURE_NUM]);
        void wekinatorControl(int command, int arg);

        //GUI
    
        ofxPanel gui;

        //Video source

        #ifdef _USE_LIVE_VIDEO
		  ofVideoGrabber 		vidGrabber;
		#else
		  ofVideoPlayer 		vidPlayer;
		#endif

        ofxCvColorImage			colorImg;

        ofxCvGrayscaleImage 	grayImage;
		ofxCvGrayscaleImage 	grayBg;
		ofxCvGrayscaleImage 	grayDiff;

    
        //Contour
        ofxCvContourFinder 	contourFinder;
        float area;
        float length;
    
        //Convex Hull
        ofxConvexHull convexHull;
        vector<ofPoint> points;
        vector<ofPoint> hull;
        float hullArea;
    
    
        //for background subtraction
		bool    bLearnBakground;
        ofParameter<int> threshold;

        //for OSC
        float features[FEATURE_NUM];
        ofxOscSender sender;
    
    
        //for GRT
    

    
        //Create some variables for the demo
        ClassificationData trainingData;      		//This will store our training data
        GestureRecognitionPipeline pipeline;        //This is a wrapper for our classifier and any pre/post processing modules
        bool record;                                //This is a flag that keeps track of when we should record training data
        bool drawInfo;
        int classifierType;
        UINT trainingClassLabel;                    //This will hold the current label for when we are training the classifier
        string infoText;                            //This string will be used to draw some info messages to the main app window
        ofTrueTypeFont largeFont;
        ofTrueTypeFont smallFont;
    
        /* for Plotting
        ofxGrtTimeseriesPlot feature1Plot;
        ofxGrtTimeseriesPlot feature2Plot;
        ofxGrtTimeseriesPlot feature3Plot;
       
        */
    
        GRT::VectorFloat GRTfeatures;
    
    


    
};

