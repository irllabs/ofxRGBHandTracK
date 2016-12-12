#include "ofApp.h"

using namespace ofxCv;
using namespace cv;

//State that we want to use the GRT namespace
using namespace GRT;

ofstream ftPosFile("fingerPos.txt");

//--------------------------------------------------------------
void ofApp::setup(){
    
#ifdef _USE_LIVE_VIDEO
    vidGrabber.setVerbose(true);
    vidGrabber.setup(VID_WIDTH,VID_HEIGHT);
#else
    vidPlayer.load("hand.mp4");
    vidPlayer.play();
    vidPlayer.setLoopState(OF_LOOP_NONE);
#endif
    
#ifdef _USE_PER_PIXEL_SEGMENTATION
    segmenter.loadModels();
#endif
    
    //for GUI
    gui.setup();
    gui.add(threshold.set("Threshold ", 10, 0, 255));
    
    
    //for OSC: open an outgoing connection to HOST:PORT
    sender.setup(HOST, PORT);
    
#ifdef _USE_PER_PIXEL_SEGMENTATION
    colorImg.allocate(VID_WIDTH,VID_HEIGHT);
    grayDiff.allocate(VID_WIDTH,VID_HEIGHT);
#else
    //for background subtraction
    colorImg.allocate(VID_WIDTH,VID_HEIGHT);
    grayImage.allocate(VID_WIDTH,VID_HEIGHT);
    grayBg.allocate(VID_WIDTH,VID_HEIGHT);
    grayDiff.allocate(VID_WIDTH,VID_HEIGHT);
    
    bLearnBakground = true;
#endif
    
    
    
    //for GRT
    
    //Initialize the training and info variables
    infoText = "";
    trainingClassLabel = 1;
    record = false;
    drawInfo = true;
    
    
    
    
    //The input to the training data will be the [x y] from the mouse, so we set the number of dimensions to 2
    trainingData.setNumDimensions( GRT_FEATURE_NUM );
    GRTfeatures.resize(GRT_FEATURE_NUM);
    
    //set the default classifier
    /*
     ANBC naiveBayes;
     naiveBayes.enableNullRejection( false );
     naiveBayes.setNullRejectionCoeff( 3 );
     pipeline.setClassifier( naiveBayes );
     */
    
    GRT::SVM svm;
    svm.enableNullRejection( false );
    svm.setNullRejectionCoeff( 3 );
    pipeline.setClassifier( GRT::SVM(GRT::SVM::RBF_KERNEL) );
    
    
    /* for plotting
     feature1Plot.setup( 500, 3, "area" );
     feature1Plot.setDrawGrid( true );
     feature1Plot.setDrawInfoText( true );
     feature1Plot.setFont( smallFont );
     */
}

void ofApp::sendOSC(float param[FEATURE_NUM]) {
    ofxOscMessage m;
    m.setAddress("/test");
    for (int k=0;k<FEATURE_NUM;k++){
        m.addFloatArg(param[k]);
    }
    
    sender.sendMessage(m, false);
    
}

void ofApp::wekinatorControl(int command, int arg) {
    ofxOscMessage m;
    
    switch ( command )
    {
        case 1:
            m.setAddress("/wekinator/control/startRecording");
            sender.sendMessage(m, false);
            break;
        case 0:
            m.setAddress("/wekinator/control/stopRecording");
            sender.sendMessage(m, false);
            break;
        case 2:
            m.setAddress("/wekinator/control/outputs");
            m.addFloatArg(arg);
            sender.sendMessage(m, false);
            break;
            
    }
    
    
    
}

//--------------------------------------------------------------
void ofApp::update(){
    ofBackground(100,100,100);
    
    bool bNewFrame = false;
    
#ifdef _USE_LIVE_VIDEO
    vidGrabber.update();
	   bNewFrame = vidGrabber.isFrameNew();
#else
    vidPlayer.update();
    bNewFrame = vidPlayer.isFrameNew();
#endif
    
    if (bNewFrame){
        
#ifdef _USE_LIVE_VIDEO
        colorImg.setFromPixels(vidGrabber.getPixels());
#else
        colorImg.setFromPixels(vidPlayer.getPixels());
#endif
        
#ifdef _USE_PER_PIXEL_SEGMENTATION
        segmenter.detect(colorImg);
        segmenter.getForegroundMask(foregroundMask);
        foregroundMask.update();
        grayDiff.setFromPixels(foregroundMask.getPixels());
#else
        grayImage = colorImg;
        if (bLearnBakground == true){
            grayBg = grayImage;		// the = sign copys the pixels from grayImage into grayBg (operator overloading)
            bLearnBakground = false;
        }
        
        // take the abs value of the difference between background and incoming and then threshold:
        grayDiff.absDiff(grayBg, grayImage);
        grayDiff.threshold(threshold);
#endif
        
        // find contours which are between the size of 20 pixels and 1/3 the w*h pixels.
        // also, find holes is set to true so we will get interior contours as well....
        contourFinder.findContours(grayDiff, 20, (VID_WIDTH*VID_HEIGHT)*.6, 1, false);	// find holes
        
        
        //calculate features
        for (int i = 0; i < contourFinder.nBlobs; i++){
            hull = convexHull.getConvexHull(contourFinder.blobs[i].pts);
            area = contourFinder.blobs[i].area;
            length    = contourFinder.blobs[i].length;
            hullArea = convexHull.getArea(hull);
        }
        
        
        // assemble and send out feature vector to Wekinator for analysis
        
        features[0]=area;
        features[1]=length;
        features[2]=hullArea;
        features[3]=hullArea/length;
        features[4]=length/area;
        sendOSC(features);
        
        
        // assemble feature vector for GRT
        
        GRTfeatures[0]=area;
        GRTfeatures[1]=length;
        GRTfeatures[2]=hullArea;
        GRTfeatures[3]=hullArea/length;
        GRTfeatures[4]=length/area;
        
        
        
        // for GRT
        //If we are recording training data, then add the current sample to the training data set
        if( record ){
            trainingData.addSample( trainingClassLabel, GRTfeatures );
        }
        
        //If the pipeline has been trained, then run the prediction
        if( pipeline.getTrained() ){
            pipeline.predict( GRTfeatures );
            //predictionPlot.update( pipeline.getClassLikelihoods() );
        }
        
    }
    
}

//--------------------------------------------------------------
void ofApp::draw(){
    
    // draw the incoming, the grayscale, the bg and the thresholded difference
#ifdef _USE_PER_PIXEL_SEGMENTATION
    ofSetHexColor(0xffffff);
    colorImg.draw(VID_SPACE,VID_SPACE);
    grayDiff.draw(VID_SPACE*2+VID_WIDTH,20);
#else
    ofSetHexColor(0xffffff);
    colorImg.draw(VID_SPACE,VID_SPACE);
    grayImage.draw(VID_SPACE*2+VID_WIDTH,VID_SPACE);
    grayBg.draw(VID_SPACE*3+VID_WIDTH*2,20);
    grayDiff.draw(VID_SPACE*4+VID_WIDTH*3,20);
#endif
    
    // then draw the contours:
    
    ofFill();
    ofSetHexColor(0x333333);
    ofDrawRectangle(VID_SPACE*5+VID_WIDTH*4,VID_SPACE,VID_WIDTH,VID_HEIGHT);
    ofSetHexColor(0xffffff);
    
    ofNoFill();
    // we could draw the whole contour finder
    //contourFinder.draw(360,540);
    
    // or, instead we can draw each blob individually from the blobs vector,
    // this is how to get access to them:
    for (int i = 0; i < contourFinder.nBlobs; i++){
        
        // draw hull
        ofBeginShape();
        for (int i=0; i<hull.size(); i++) {
            ofVertex(VID_SPACE*5+VID_WIDTH*4+hull[i].x,VID_SPACE+hull[i].y);
        }
        ofEndShape();
        
        
        //draw countour and features
        
        contourFinder.blobs[i].draw(VID_SPACE*5+VID_WIDTH*4,VID_SPACE);
        
        //Sort the points in the hull by y-coordinate to get the finger tips
        struct myClass {
            bool operator() (ofPoint pt1, ofPoint pt2) {return (pt1.y < pt2.y);}
        } myObject;
        
        sort(hull.begin(), hull.end(), myObject);
        
        
        //Draw circles on the contour for the fingertips, and store the fingertip positions in the text file
        ofSetColor(0, 0, 255);
        ofFill();
        int numFingers = 5;
        int count = 0;
        float prevX = 0;
        float prevY = 0;
        for (int m=0; m < 15; m++) {
            if (abs(hull[m].x - prevX) > 10 && abs(hull[m].y - prevY)) {
                //TODO: Never draw circle if coordinate is below the center of mass
                ofDrawCircle(VID_SPACE*5+VID_WIDTH*4+hull[m].x, VID_SPACE+hull[m].y, 5);
                cout << "The x coordinate is: " << VID_SPACE*5+VID_WIDTH*4+hull[m].x << endl;
                cout << "The y coordinate is: " << VID_SPACE+hull[m].y << endl;
                if (ftPosFile.is_open()) {
                    ftPosFile << VID_SPACE*5+VID_WIDTH*4+hull[m].x;
                    ftPosFile << ",";
                    ftPosFile << VID_SPACE+hull[m].y << endl;
                }
                count++;
            }
            if (count == numFingers) {
                break;
            }
            prevX = hull[m].x;
            prevY = hull[m].y;
        }
        
        //ftPosFile.close();  //close the file
        
        
        //Find the approximate center of the palm
        ofPoint hullCenter;
        hullCenter.x = 0;
        hullCenter.y = 0;
        for (int m=0; m < hull.size(); m++) {
            hullCenter.x += hull[m].x;
            hullCenter.y += hull[m].y;
        }
        
        hullCenter.x /= hull.size();
        hullCenter.y /= hull.size();
        
        //Draw a circle for the center of the palm
        ofSetColor(255, 0, 0);
        ofFill();
        ofDrawCircle(VID_SPACE*5+VID_WIDTH*4+hullCenter.x, VID_SPACE+hullCenter.y, 3);
        
        //Calculate the distance from center of palm to the five fingertips
        float first = hullCenter.distance(hull[0]);
        float second = hullCenter.distance(hull[1]);
        float third = hullCenter.distance(hull[2]);
        float fourth = hullCenter.distance(hull[3]);
        float fifth = hullCenter.distance(hull[4]);
        
        ofDrawBitmapString("A: "+ofToString(area),
                           contourFinder.blobs[i].boundingRect.getCenter().x + VID_SPACE*5+VID_WIDTH*4,
                           contourFinder.blobs[i].boundingRect.getCenter().y + VID_SPACE);
        
        ofDrawBitmapString("P: "+ofToString(length),
                           contourFinder.blobs[i].boundingRect.getCenter().x + VID_SPACE*5+VID_WIDTH*4,
                           contourFinder.blobs[i].boundingRect.getCenter().y + VID_SPACE + 10);
        
        ofDrawBitmapString("H: "+ofToString(hullArea),
                           contourFinder.blobs[i].boundingRect.getCenter().x + VID_SPACE*5+VID_WIDTH*4,
                           contourFinder.blobs[i].boundingRect.getCenter().y + VID_SPACE + 20);
        
        
        // draw over the centroid if the blob is a hole
        ofSetColor(255);
        
        if(contourFinder.blobs[i].hole){
            ofDrawBitmapString("hole",
                               contourFinder.blobs[i].boundingRect.getCenter().x + VID_SPACE*5+VID_WIDTH*4,
                               contourFinder.blobs[i].boundingRect.getCenter().y + VID_SPACE);
        }
        
        
        
        
        
    }
    
    
    
    ofSetColor(255);
    // for GRT
    
    if (record) {
        ofSetColor(255,0,0);
        
        string featureStr = "";
        for (int i=0;i<FEATURE_NUM;i++){
            featureStr = featureStr + ofToString(features[i]);
        }
        
        infoText="GRT record " + ofToString(trainingClassLabel)+ ": " + featureStr;
        
    }
    
    
    //If the model has been trained, then draw the texture
    if( pipeline.getTrained() ){
        ofSetColor(0,255,0);
        infoText="GRT Predict: " + ofToString(pipeline.getPredictedClassLabel());
    }
    
    
    ofDrawBitmapString(infoText,infoTextX,infoTextY);
    
    // instructions
    ofSetHexColor(0xffffff);
    ofDrawBitmapString("press SPACE-BAR to reset background", infoTextX,infoTextY + 20);
    ofDrawBitmapString("press and hold 1-5 to record training samples for the corresponding class", infoTextX,infoTextY + 40);
    ofDrawBitmapString("press 't' to train (with SVM)", infoTextX,infoTextY + 60);
    ofDrawBitmapString("press 'c' to to clear training", infoTextX,infoTextY + 80);
    
    gui.draw();
}



//--------------------------------------------------------------
void ofApp::keyPressed(int key){
    
    switch (key){
        case ' ':
            bLearnBakground = true;
            break;
        case '+':
            threshold ++;
            if (threshold > 255) threshold = 255;
            break;
        case '-':
            threshold --;
            if (threshold < 0) threshold = 0;
            break;
        case 'r':
            wekinatorControl(1,0);
            break;
        case 's':
            wekinatorControl(0,0);;
            if (threshold < 0) threshold = 0;
            break;
        case '1':
            trainingClassLabel = 1;
            wekinatorControl(2,trainingClassLabel);
            wekinatorControl(1,0);
            record = true;
            break;
        case '2':
            trainingClassLabel = 2;
            wekinatorControl(2,trainingClassLabel);
            wekinatorControl(1,0);
            
            record = true;
            break;
        case '3':
            trainingClassLabel = 3;
            wekinatorControl(2,trainingClassLabel);
            wekinatorControl(1,0);
            
            record = true;
            break;
        case '4':
            trainingClassLabel = 4;
            wekinatorControl(2,trainingClassLabel);
            wekinatorControl(1,0);
            
            record = true;
            break;
        case '5':
            trainingClassLabel = 5;
            wekinatorControl(2,trainingClassLabel);
            wekinatorControl(1,0);
            
            record = true;
            break;
            
        case 't':
            if( pipeline.train( trainingData ) ){
                infoText = "Pipeline Trained";
                //predictionPlot.setup( 500, pipeline.getNumClasses(), "prediction likelihoods" );
                //predictionPlot.setDrawGrid( true );
                //predictionPlot.setDrawInfoText( true );
                //predictionPlot.setFont( smallFont );
            }
            
            else infoText = "WARNING: Failed to train pipeline";
            break;
        case 'c':
            trainingData.clear();
            
            pipeline.clear();
            pipeline.setClassifier( GRT::SVM(GRT::SVM::RBF_KERNEL) );
            infoText = "Training data cleared";
            break;
    }
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){
    switch (key){
        case '1':
            record = false;
            wekinatorControl(0,0);;
            break;
        case '2':
            record = false;
            wekinatorControl(0,0);;
            break;
        case '3':
            record = false;
            wekinatorControl(0,0);;
            break;
        case '4':
            record = false;
            wekinatorControl(0,0);;
            break;
        case '5':
            record = false;
            wekinatorControl(0,0);;
            break;
    }
    
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){
    
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
    
}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){
    
}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){
    
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){
    
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){
    
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 
    
}
