#!/usr/bin/python
"""
This program is demonstration for face and object detection using haar-like features.
The program finds faces in a camera image or video stream and displays a red box around them.

Original C implementation by:  ?
Python implementation by: Roman Stanchak
"""
import sys
from opencv.cv import *
from opencv.highgui import *


# Global Variables
cascade = None
storage = cvCreateMemStorage(0)
cascade_name = "./haarcascade_frontalface_alt.xml"
input_name = "../c/lena.jpg"
lm_name = "../gits-lm-withwhite.png"
MPEG1VIDEO = 0x314D4950


# Parameters for haar detection
# From the API:
# The default parameters (scale_factor=1.1, min_neighbors=3, flags=0) are tuned 
# for accurate yet slow object detection. For a faster operation on real video 
# images the settings are: 
# scale_factor=1.2, min_neighbors=2, flags=CV_HAAR_DO_CANNY_PRUNING, 
# min_size=<minimum possible face size
min_size = cvSize(20,20)
image_scale = 1.3
haar_scale = 1.2
min_neighbors = 2
#haar_flags = CV_HAAR_DO_CANNY_PRUNING,
haar_flags = CV_HAAR_DO_CANNY_PRUNING


def detect_and_draw( img ):
    # allocate temporary images
    gray = cvCreateImage( cvSize(img.width,img.height), 8, 1 )
    small_img = cvCreateImage((cvRound(img.width/image_scale),
			       cvRound (img.height/image_scale)), 8, 1 )

    # convert color input image to grayscale
    cvCvtColor( img, gray, CV_BGR2GRAY )

    # scale input image for faster processing
    cvResize( gray, small_img, CV_INTER_LINEAR )

    cvEqualizeHist( small_img, small_img )
    
    cvClearMemStorage( storage )

    if( cascade ):
        t = cvGetTickCount()
        faces = cvHaarDetectObjects( small_img, cascade, storage, 
				haar_scale, min_neighbors, haar_flags, min_size )
        #t = cvGetTickCount() - t
        #print "detection time = %gms" % (t/(cvGetTickFrequency()*1000.))
        if faces:
            for face_rect in faces:
                # the input to cvHaarDetectObjects was resized, so scale the 
                # bounding box of each face and convert it to two CvPoints
                pt1 = cvPoint( int(face_rect.x*image_scale), int(face_rect.y*image_scale))
                pt2 = cvPoint( int((face_rect.x+face_rect.width)*image_scale),
                               int((face_rect.y+face_rect.height)*image_scale) )
                #cvRectangle( img, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0 )
                lman = cvLoadImage( lm_name, 1 )
                lman_small = cvCreateImage ( cvSize(int((face_rect.width*image_scale)),int((face_rect.height*image_scale))),
                                             IPL_DEPTH_8U, img.nChannels )
		cvResize(lman,lman_small)
                x_offset=int(face_rect.x*image_scale)
                y_offset=int(face_rect.y*image_scale)
                subarea=cvGetSubRect(img, cvRect( x_offset , y_offset, int(face_rect.width*image_scale), int(face_rect.height*image_scale)))
                smallx = 0
                for xrange in range(0,lman_small.height):
                  smally = 0
                  for yrange in range(0,lman_small.width):
                    overlaycolor = cvGet2D(lman_small,smallx,smally)
                    sourcecolor = cvGet2D(img,xrange+y_offset,smally+x_offset)
                    if ( overlaycolor[0] == 0.0 and overlaycolor[1] == 0.0 and overlaycolor[2] == 0.0 ):
                      cvSet2D(lman_small,smallx,smally,sourcecolor)
                      True
                    smally = smally + 1
                  smallx = smallx + 1
                #paste
		cvCopy(lman_small,subarea)

    cvShowImage( "result", img )
#    cvWriteFrame( writer, img )


if __name__ == '__main__':

    if len(sys.argv) > 1:

        if sys.argv[1].startswith("--cascade="):
            cascade_name = sys.argv[1][ len("--cascade="): ]
            if len(sys.argv) > 2:
                input_name = sys.argv[2]

        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print "Usage: facedetect --cascade=\"<cascade_path>\" [filename|camera_index]\n" 
            sys.exit(-1)

        else:
            input_name = sys.argv[1]
    
    # the OpenCV API says this function is obsolete, but we can't
    # cast the output of cvLoad to a HaarClassifierCascade, so use this anyways
    # the size parameter is ignored
    cascade = cvLoadHaarClassifierCascade( cascade_name, cvSize(1,1) )
    
    if not cascade:
        print "ERROR: Could not load classifier cascade"
        sys.exit(-1)
    

    if input_name.isdigit():
        capture = cvCreateCameraCapture( int(input_name) )
    else:
        capture = cvCreateFileCapture( input_name ) 

## SNIP OF WRITER
##    # capture the 1st frame to get some propertie on it
##    frame = cvQueryFrame (capture)
##
##    # get size of the frame
##    frame_size = cvGetSize (frame)
##
##    # get the frame rate of the capture device
##    #fps = cvGetCaptureProperty (capture, CV_CAP_PROP_FPS)
##    fps = 0
##    if fps == 0:
##        # no fps getted, so set it to 30 by default
##        fps = 24
##
##    # create the writer
##    writer = cvCreateVideoWriter ("captured.mpg", MPEG1VIDEO,
##                                          fps, frame_size, True)
##
##    # check the writer is OK
##    if not writer:
##        print "Error opening writer"
##        sys.exit (1)
##
##/ SNIP OF WRITER

    cvNamedWindow( "result", 1 )

    if capture:
        frame_copy = None
        while True:
            frame = cvQueryFrame( capture )
            if not frame:
                cvWaitKey(0)
                break
            if not frame_copy:
                frame_copy = cvCreateImage( cvSize(frame.width,frame.height),
                                            IPL_DEPTH_8U, frame.nChannels )
            if frame.origin == IPL_ORIGIN_TL:
                cvCopy( frame, frame_copy )
            else:
                cvFlip( frame, frame_copy, 0 )
            
            detect_and_draw( frame_copy )

            if( cvWaitKey( 10 ) >= 0 ):
                break

    else:
        image = cvLoadImage( input_name, 1 )

        if image:
            detect_and_draw( image )
            cvWaitKey(0)

    cvDestroyWindow("result")
