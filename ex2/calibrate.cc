#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

enum { INTERACTIVE_MODE, PRESPECIFIED_MODE };

#define KEY_ESCAPE 1048603
#define KEY_SPACE 1048608
#define KEY_CLOSE_WINDOW -1

#define COUNT_SQUARES_X 4
#define COUNT_SQUARES_Y 6

// global container variables
double middle_x, middle_y, cx, cy, projected_point_x, projected_point_y;

/**
 * Method for detecting pattern in current frame. 
 * Returns world and image coordinates of detected corners via out parameters.
 */
bool detectPattern(Mat frame,
			    vector< vector<Point3f> >& object_points,
			    vector<vector<Point2f> >& image_points)
{
  // number of squares in the pattern, a.k.a, interior number of corners
  Size pattern_size(COUNT_SQUARES_X, COUNT_SQUARES_Y);
  // storage for the detected corners in findChessboardConrners
  vector<Point2f> corners;

  bool pattern_found = findChessboardCorners(
            frame, pattern_size, corners,
            CALIB_CB_ADAPTIVE_THRESH +
            CALIB_CB_NORMALIZE_IMAGE +
            CALIB_CB_FAST_CHECK);

  if (!pattern_found)
    return false;

  // if corners are detected, they are further refined by
  // calculating subpixel corners from the grayscale image
  // this iterative process terminates after the given number
  // of iterations and error epsilon
  cornerSubPix(frame, corners, Size(11, 11), Size(-1, -1),
               TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100,
               0.15));
  // draw the detected corners as sanity check
  drawChessboardCorners(frame, pattern_size, Mat(corners),
                        pattern_found);

  // show detected corners in a different window
  imshow("Detected pattern", frame);

  // build a grid of 3D points (z component is 0 because the pattern is
  // in one plane) to fit the square pattern area
  // (COUNT_SQUARES_X * COUNT_SQUARES_Y)
  object_points.clear();
  image_points.clear();
  vector<Point3f> pattern_points;
  for (int j = 0; j < COUNT_SQUARES_X * COUNT_SQUARES_Y; ++j)
    pattern_points.push_back(Point3f(j / COUNT_SQUARES_X,
							  j % COUNT_SQUARES_X,
							  0.0f));


  // get middle of checkerboard coordinates
  // this already is in image coordinates? pixels?
  double min_x = 1000.0;
  double min_y = 1000.0;
  double max_x = 0.0;
  double max_y = 0.0;
  for (int k = 0; k<corners.size(); k++) {
    cout << "Checkerboard Corner " << k << ": " << corners[k].x << "/" << corners[k].y << endl;
    if(corners[k].x < min_x) min_x = corners[k].x;
    if(corners[k].x > max_x) max_x = corners[k].x;
    if(corners[k].y < min_y) min_y = corners[k].y;
    if(corners[k].y > max_y) max_y = corners[k].y;
  }

  middle_x = 0.5*(max_x-min_x) + min_x;
  middle_y = 0.5*(max_y-min_y) + min_y;

  cout << "Checkerboard Middle: " << middle_x << "/" << middle_y << endl;

  // populate image points with corners and object points with grid points
  object_points.push_back(pattern_points);
  image_points.push_back(corners);

  return true;
}

/**
 * Main method for interactive behavior. 
 * Requires user to present calibration pattern in front of camera. 
 * By pressing any key an image is grabbed from the camera stream,
 * pressing ESC finishes calibration.
 */
int runInteractive(int iCap = 0)
{
  cout << "Camera calibration using interactive behavior. "
	  << " Press any key to grab frame, ESC to perform calibration.\n";
  // show camera image in a separate window
  namedWindow("Camera Image", CV_WINDOW_KEEPRATIO);

  // current camera frame and first captured frame 
  Mat frame, undistorted_frame;
  // storage for object points (world coords)
  // and image points (image coords) for use in calibrateCamera 
  vector< vector< Point3f > > object_points;
  vector< vector< Point2f > > image_points;
  // calibration parameters
  Mat intrinsic = Mat(3, 3, CV_32FC1);
  Mat distCoeffs;
  vector<Mat> rvecs, tvecs;

  // open the default camera
  VideoCapture capture(0);      // changed: this was capture(iCap), but did not work. Needs to be manually set to 0 to work.

  // check if opening camera stream succeeded
  if (!capture.isOpened()) 
  {
    cerr << "Camera could not be found. Exiting.\n";
    return -1;
  }

  // set frame width and height by hand, defaults to 160x120
  capture.set (CV_CAP_PROP_FRAME_WIDTH, 640);
  capture.set (CV_CAP_PROP_FRAME_HEIGHT, 480);

  // flag for determining whether pattern was detected in at
  // least one of the camera grabs
  bool calibration_ready = false;
  int count_frames = 0;
  
  for (; ; count_frames++) {
    cout << "Frame: " << count_frames << endl;

    capture >> frame; // get a new frame from camera
    Mat gray_frame;
    // convert current frame to grayscale
    cvtColor(frame, gray_frame, CV_BGR2GRAY); 
    imshow("Camera Image", frame); // update camera image

    int key_pressed = waitKey(25); // get user key press
    if (key_pressed == 27)
	 break;
    if (detectPattern(gray_frame, object_points, image_points)) {
	 calibration_ready = true;
	 if (key_pressed == 32)
	   break;
    } 
  }
  
  // if at least one video capture contains the pattern, perform calibration
  if (calibration_ready) 
  {
    // perform calibration, obtain instrinsic parameters
    // and distortion coefficients
    cout << "Calibrating..." << endl;
    calibrateCamera(object_points,
				image_points,
				frame.size(),
				intrinsic,
                    distCoeffs,
                                rvecs,          // dtype: OutputArrayOfArrays, output vector of rotation vectors
                                tvecs);         // dtype: OutputArrayOfArrays, output vector of translation vectors

    cout << "Intrinsic parameters:" << endl << intrinsic << endl;
    cout << "Distortion coefficients:" << endl << distCoeffs << endl;

    FileStorage fs("myCalibration.yml", FileStorage::WRITE);
    FileStorage fs_ext("extrinisic.yml",FileStorage::WRITE);

    // get rotation matrix from Rodriguez vector returned by calibrateCamera
    Mat rot_mat = Mat::zeros(3,3,CV_64F);
    Rodrigues(rvecs[0], rot_mat);

    cx = intrinsic.at<double>(0,2);
    cy = intrinsic.at<double>(1,2);

    // create 3D point
    Point3d center = Point3d(cx,cy,250.0);      // z is height component of checkerboard middle in mm
    vector<Point3d> threed_points;
    threed_points.push_back(center);

    // create container for returned 2D projection
    vector<Point2d> projectedPoints;

    // project 3D point into 2D scene
    projectPoints(threed_points, rvecs[0], tvecs[0], intrinsic, distCoeffs, projectedPoints);


    for(unsigned int i=0; i<projectedPoints.size(); i++){
       std::cout << "Image point: " << threed_points[i] << " Projected to " << projectedPoints[i] << "\n";
     }

    projected_point_x = projectedPoints[0].x;
    projected_point_y = projectedPoints[0].y;

    fs << "intrinsic" << intrinsic;
    fs << "distCoeffs" << distCoeffs;
    fs_ext << "rotation matrix" << rot_mat;
    fs_ext << "transation vector" << tvecs;
  }
  else 
  {
    cerr << "No pattern found in any video capture. Exiting." << endl;
  }

  for (; ; count_frames++) {
    cout << "Frame: " << count_frames << " (calibrated) " << endl;
    capture >> frame; // get a new frame from camera
    undistort(frame, undistorted_frame, intrinsic, distCoeffs);

    // frame size: 640x480
    // project 2d point into undistorted frame
    circle(undistorted_frame, Point(cx, cy), 10, Scalar(255,0,0), CV_FILLED);       // blue
    circle(undistorted_frame, Point(projected_point_x, projected_point_y), 10, Scalar(0,255,0), CV_FILLED); // green
    circle(undistorted_frame, Point(middle_x, middle_y), 10, Scalar(0,0,255), CV_FILLED);   // red
    imshow("Camera Image", undistorted_frame); // update camera image
    int key_pressed = waitKey(25); // get user key press
    if (key_pressed == 27)
	 break;
  }
  
  // release camera
  capture.release();
}


int main(int argc, char **argv)
{
  int user_mode;
  int specified_boards;

  /* no arguments means interactive mode
   * one or more arguments are image filenames */
  if (argc == 1) {
      runInteractive(1);
  } else {
    cout << argv[1] << endl;
    runInteractive(atoi(argv[1]));
  }

  return 0;
}
