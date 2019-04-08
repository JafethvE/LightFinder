import java.util.List;
import java.util.ArrayList;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

class ObjectDetector {
	
	private int lightSources = 0;
	
    public void detectAndDisplay(Mat frame, CascadeClassifier faceCascade, CascadeClassifier eyesCascade) {
    	
    	//Make a seperate frame to do editing on without disturbing the original.
        Mat frameGray = new Mat();
        
        //Gray and blur the frame.
        Imgproc.cvtColor(frame, frameGray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(frameGray, frameGray);
        Imgproc.GaussianBlur(frameGray, frameGray, new Size(11, 11), 11);
        
        //Apply a threshold to the frame
        Imgproc.threshold(frameGray, frameGray, 250, 255, Imgproc.THRESH_BINARY);
        
        //init
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        
        Imgproc.findContours(frameGray, contours, hierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
        
        if (hierarchy.size().height > 0 && hierarchy.size().width > 0)
        {
                // for each contour, display it in blue
                for (int idx = 0; idx >= 0; idx = (int) hierarchy.get(0, idx)[0])
                {
                        Imgproc.drawContours(frame, contours, idx, new Scalar(250, 0, 0));
                        Imgproc.putText(frame, String.valueOf(idx + 1), contours.get(idx).toArray()[0], 0, 0.65, new Scalar(0, 255, 0));
                }
        }

        //-- Show the original frame and the edited frame in two seperate Windows.
        HighGui.imshow("Capture - changed", frameGray );
        HighGui.imshow("Capture", frame);
    }

    public void run(String[] args) {
        String filenameFaceCascade = args.length > 2 ? args[0] : "F:\\Programming\\Java\\libraries\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
        String filenameEyesCascade = args.length > 2 ? args[1] : "F:\\Programming\\Java\\libraries\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml";
        int cameraDevice = args.length > 2 ? Integer.parseInt(args[2]) : 0;

        CascadeClassifier faceCascade = new CascadeClassifier();
        CascadeClassifier eyesCascade = new CascadeClassifier();

        if (!faceCascade.load(filenameFaceCascade)) {
            System.err.println("--(!)Error loading face cascade: " + filenameFaceCascade);
            System.exit(0);
        }
        if (!eyesCascade.load(filenameEyesCascade)) {
            System.err.println("--(!)Error loading eyes cascade: " + filenameEyesCascade);
            System.exit(0);
        }

        VideoCapture capture = new VideoCapture(cameraDevice);
        if (!capture.isOpened()) {
            System.err.println("--(!)Error opening video capture");
            System.exit(0);
        }

        Mat frame = new Mat();
        while (capture.read(frame)) {
            if (frame.empty()) {
                System.err.println("--(!) No captured frame -- Break!");
                break;
            }

            //-- 3. Apply the classifier to the frame
            detectAndDisplay(frame, faceCascade, eyesCascade);

            if (HighGui.waitKey(10) == 27) {
                break;// escape
            }
        }

        System.exit(0);
    }
}