package org.firstinspires.ftc.teamcode;

import android.graphics.Canvas;

import org.firstinspires.ftc.robotcore.internal.camera.calibration.CameraCalibration;
import org.firstinspires.ftc.vision.VisionProcessor;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.List;

public class ElementVisionProcessor implements VisionProcessor {
    private int screenWidth;
    private int elementPosX;

    private Mat hsvFrame;
    private Mat maskFrame;
    private Mat hierarchy;

    private static final Scalar LOWER_BLUE = new Scalar(90, 50, 50);
    private static final Scalar HIGHER_BLUE = new Scalar(130, 255, 255);

    @Override
    public void init(int width, int height, CameraCalibration calibration) {
        this.screenWidth = width;

        this.hsvFrame = new Mat();
        this.maskFrame = new Mat();
        this.hierarchy = new Mat();
    }

    @Override
    public Object processFrame(Mat frame, long captureTimeNanos) {
        // Convert the frame from RGBA values to HSV
        Imgproc.cvtColor(frame, hsvFrame, Imgproc.COLOR_RGBA2RGB);
        Imgproc.cvtColor(hsvFrame, hsvFrame, Imgproc.COLOR_RGB2HSV);

        // Get the blue colors from the frame
        Core.inRange(hsvFrame, LOWER_BLUE, HIGHER_BLUE, maskFrame);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(maskFrame, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        double maxArea = 0;
        int maxContourIndex = -1;

        for (int i = 0; i < contours.size(); i++) {
            double contourArea = Imgproc.contourArea(contours.get(i));

            if (contourArea > maxArea) {
                maxArea = contourArea;
                maxContourIndex = i;
            }
        }

        if (maxContourIndex != -1) {
            Imgproc.drawContours(frame, contours, maxContourIndex, new Scalar(0, 255, 0), 2);
            Moments moments = Imgproc.moments(contours.get(maxContourIndex));
            this.elementPosX = (int) (moments.get_m10() / moments.get_m00());
        } else {
            this.elementPosX = -1;
        }

        return frame;
    }

    @Override
    public void onDrawFrame(Canvas canvas, int onscreenWidth, int onscreenHeight, float scaleBmpPxToCanvasPx, float scaleCanvasDensity, Object userContext) {}

    public int getElementRegion() {
        if (this.elementPosX < (screenWidth / 3))
            return 1; // Left or not found
        else if (this.elementPosX < 2 * screenWidth / 3)
            return 2; // Center
        else
            return 3; // Right
    }
}