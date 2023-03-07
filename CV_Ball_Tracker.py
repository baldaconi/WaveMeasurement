# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 08:41:06 2022

@author: ibald
This version uses in built opencv object tracking
"""

from __future__ import print_function
import cv2 as cv
import pandas as pd
import numpy as np
import os

def findCircle(dFrame, targMeanRd, lsc, usc):

    hsv = cv.cvtColor(dFrame, cv.COLOR_BGR2HSV)
    if hsv is None:
        circle = None
    else:
        # preparing the mask to overlay
        mask = cv.inRange(hsv, lsc, usc)
        # The black region in the mask has the value of 0,
        # so when multiplied with original image removes all non-orange regions
        maskedFrame = cv.bitwise_and(dFrame, dFrame, mask=mask)
        if maskedFrame is None:
            circle = None
        else:
            # cv.imshow("Masked", maskedFrame)
            # cv.waitKey(0)
            grayFrame = cv.cvtColor(maskedFrame, cv.COLOR_BGR2GRAY)
            if grayFrame is None:
                circle = None
            else:
                blurFrame = cv.GaussianBlur(grayFrame, (7, 7), 0)
                if blurFrame is None:
                    circle = None
                else:
                    # cv.imshow("Blur", blurFrame)
                    # cv.waitKey(0)
                    templateCircle = cv.HoughCircles(blurFrame, cv.HOUGH_GRADIENT, 1, targMeanRd//2,
                                                      param1=25, param2=5,
                                                      minRadius=np.intc(np.round(0.95 * targMeanRd)),
                                                      maxRadius=np.intc(np.round(1.05 * targMeanRd)))
                    if not(templateCircle is None):
                        circle = np.uint16(np.around(templateCircle))
                        circle = circle[0, :]
                    else:
                        circle = None
                    # cv.imshow("Targets found", dFrame)
                    # cv.waitKey(0)
                    # cv.destroyAllWindows()
    return circle


###################################################################
numTargets = 20
showVideoOnScreen = True
recordVideoOutput = True
loadPreSelectedROIs = False

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
tracker_type = tracker_types[2]

# tracker = np.empty(numTargets)
if tracker_type == 'BOOSTING':
    tracker = [cv.legacy.TrackerBoosting_create()]
if tracker_type == 'MIL':
    tracker = [cv.legacy.TrackerMIL_create()]
if tracker_type == 'KCF':
    tracker = [cv.legacy.TrackerKCF_create()]
if tracker_type == 'TLD':
    tracker = [cv.legacy.TrackerTLD_create()]
if tracker_type == 'MEDIANFLOW':
    tracker = [cv.legacy.TrackerMedianFlow_create()]
if tracker_type == 'CSRT':
    tracker = [cv.legacy.TrackerCSRT_create()]
if tracker_type == 'MOSSE':
    tracker = [cv.legacy.TrackerMOSSE_create()]

for i in range(1, numTargets):
    if tracker_type == 'BOOSTING':
        tracker.append(cv.legacy.TrackerBoosting_create())
    if tracker_type == 'MIL':
        tracker.append(cv.legacy.TrackerMIL_create())
    if tracker_type == 'KCF':
        tracker.append(cv.legacy.TrackerKCF_create())
    if tracker_type == 'TLD':
        tracker.append(cv.legacy.TrackerTLD_create())
    if tracker_type == 'MEDIANFLOW':
        tracker.append(cv.legacy.TrackerMedianFlow_create())
    if tracker_type == 'CSRT':
        tracker.append(cv.legacy.TrackerCSRT_create())
    if tracker_type == 'MOSSE':
        tracker.append(cv.legacy.TrackerMOSSE_create())

videoFileName = "MVI_0013_converted.mp4"
videoFilePath = "D:\Computer Vision Experiments\Articulated Structure"
videoFolderCase = "no mooring"
ballRealSize = 30  # ball size in mm
fxVideoRsz = 1
fyVideoRsz = 1
# for ping pong balls
# lowerShadeColor = np.array([0, 133, 126])  # darker shades
# upperShadeColor = np.array([63, 255, 255])  # lighter shades
# for 20 targets bright targets
lowerShadeColor = np.array([0, 98, 177])  # darker shades
upperShadeColor = np.array([180, 255, 255])  # lighter shades

fullPath = os.path.join(videoFilePath, videoFolderCase, videoFileName)

print(fullPath)
print('This version is defined for {} targets'.format(numTargets))

# Opening video file
videoCapture = cv.VideoCapture(fullPath)

ret, frame = videoCapture.read()
kFrame = 0
if not ret:
    if not videoCapture.isOpened():
        print("Error opening video stream or file")

# Counting video duration and # of frames
frameCount = int(videoCapture.get(cv.CAP_PROP_FRAME_COUNT))
FPS = videoCapture.get(cv.CAP_PROP_FPS)

videoDuration = frameCount / FPS
videoMinutes = int(videoDuration / 60)
videoSeconds = videoDuration % 60

print('Total duration of video: {}:{}'.format(videoMinutes, videoSeconds))

vidCodec = videoCapture.get(cv.CAP_PROP_FOURCC)
frameSz = (int(videoCapture.get(3)), int(videoCapture.get(4)))
if recordVideoOutput:
    outputPath = os.path.join(videoFilePath, videoFolderCase, 'output videos')
    try:
        os.mkdir(outputPath)
    except OSError as error:
        print(error)
    outputVideo = videoFileName.replace('_converted.mp4', '_read.mp4')
    videoRecording = cv.VideoWriter(os.path.join(outputPath, outputVideo), cv.VideoWriter_fourcc(*'XVID'), FPS, frameSz)

# Defining variables sizes
ROI = np.empty((numTargets, 4))
targetTemplate = np.empty((numTargets, 4))
circlePosition = np.empty((numTargets, frameCount-1, 3))
bboxPosition = np.empty((numTargets, frameCount-1, 4))
targetSz = np.empty((numTargets, 4))
targetRd = np.empty(numTargets)
# first column is the mean measured value, second column is the improved
# radius found by circle identification

# Reading video
ret, originalFrame = videoCapture.read()
frame = originalFrame
kFrame = 0
if not ret:
    if not videoCapture.isOpened():
        print("Error opening video stream or file")

elif loadPreSelectedROIs:
    for i in range(0, numTargets):
        # Select Region of Interest for each target
        dialogTxt = 'Select ROI #{}'.format(i+1)
        resizedFrame = cv.resize(originalFrame, None, fx=fxVideoRsz, fy=fyVideoRsz, interpolation=cv.INTER_LINEAR)
        ROI[i, :] = cv.selectROI(dialogTxt, resizedFrame)
        # ROI receives top-left x, top-left y, width and height of selection
        croppedFrame = frame[int((1./fyVideoRsz)*ROI[i, 1]):int((1./fyVideoRsz)*(ROI[i, 1] + ROI[i, 3])),
                             int((1./fxVideoRsz)*ROI[i, 0]):int((1./fxVideoRsz)*(ROI[i, 0] + ROI[i, 2]))].copy()
        cv.destroyWindow(dialogTxt)

        # Select area around target to be tracked
        dialogTxt = 'Select area #{} to be tracked'.format(i+1)
        targetTemplate[i, :] = cv.selectROI(dialogTxt, cv.resize(croppedFrame, None, fx=2, fy=2,
                                                                 interpolation=cv.INTER_LINEAR))
        targetTemplate[i, :] = targetTemplate[i, :] // 2
        ret = tracker[i].init(croppedFrame, targetTemplate[i, :])
        cv.destroyWindow(dialogTxt)

        # Select each target to act as scale
        dialogTxt = 'Select target #{}'.format(i+1)
        targetSz[i, :] = cv.selectROI(dialogTxt, cv.resize(croppedFrame, None, fx=2, fy=2,
                                                           interpolation=cv.INTER_LINEAR))
        targetSz[i, :] = targetSz[i, :] // 2
        targetRd[i] = np.average(targetSz[i, 2:4]/2) # radius
        cv.destroyWindow(dialogTxt)

        # Find the tracked area in the cropped frame
        ret, bbox = tracker[i].update(croppedFrame)
        if ret:
            bboxFrame = croppedFrame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
            # Find a circle with the defined radius within the tracked area
            circles = findCircle(bboxFrame, targetRd[i], lowerShadeColor, upperShadeColor)
            if not(circles is None):
                # Recalculate the target radius
                targetRd[i] = np.average([targetRd[i], circles[0, 2]])  # radius

                p1 = (int(bbox[0] + ROI[i, 0]), int(bbox[1] + ROI[i, 1]))
                p2 = (int(p1[0] + bbox[2]), int(p1[1] + bbox[3]))
                bboxPosition[i, kFrame, :] = bbox
                # circlePosition[i, kFrame, :] = circles[0, :]
                # Circles positions are relative to the bounding boxes
                # They need to be added to bbox and ROI top left corners so they are in the global axes
                circlePosition[i, kFrame, 0] = int(round((circles[0, 0] + targetRd[i] + targetSz[i, 0] - bbox[0])/2))
                circlePosition[i, kFrame, 1] = int(round((circles[0, 1] + targetRd[i] + targetSz[i, 1] - bbox[1])/2))
                circlePosition[i, kFrame, 2] = int(round(targetRd[i]))
                cv.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                cv.circle(frame, (int(circlePosition[i, kFrame, 0] + p1[0]), int(circlePosition[i, kFrame, 1] + p1[1])),
                          int(circlePosition[i, kFrame, 2]), (255, 0, 255), 2)
                printTxt = 'Target #{}/{}'.format(i+1, numTargets)
                cv.putText(frame, printTxt, (120, 80),
                           cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 50), 2)
            else:
                break
            printTxt = 'Frame #{}/{}'.format(kFrame+1, frameCount)
            cv.putText(frame, printTxt, (120, 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 50), 2)
            # cv.imshow("Tracking", frame)
            # cv.waitKey(0)
            # It is only necessary to measure the position of the ball inside the tracking area, because that is fixed.
            # After that, the tracking will give the waveform and the program should store only the x,y and r of the circle
        else:
            cv.putText(frame, "Failed", (100, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            break

    pd.DataFrame(ROI).to_csv('saved_ROI.csv')
    pd.DataFrame(targetSz).to_csv('saved_TARGET_SZ.csv')
    pd.DataFrame(targetTemplate).to_csv('saved_TARGETS_BXS.csv')
else:
    ROI = pd.read_csv('saved_ROI.csv')
    targetSz = pd.read_csv('saved_TARGET_SZ.csv')
    targetTemplate = pd.read_csv('saved_TARGETS_BXS.csv')
    for i in range(0, numTargets):
        croppedFrame = frame[int((1./fyVideoRsz)*ROI[i, 1]):int((1./fyVideoRsz)*(ROI[i, 1] + ROI[i, 3])),
                             int((1./fxVideoRsz)*ROI[i, 0]):int((1./fxVideoRsz)*(ROI[i, 0] + ROI[i, 2]))].copy()
        ret = tracker[i].init(croppedFrame, targetTemplate[i, :])
        targetRd[i] = np.average(targetSz[i, 2:4]/2) # radius

        # Find the tracked area in the cropped frame
        ret, bbox = tracker[i].update(croppedFrame)
        if ret:
            bboxFrame = croppedFrame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
            # Find a circle with the defined radius within the tracked area
            circles = findCircle(bboxFrame, targetRd[i], lowerShadeColor, upperShadeColor)
            if not(circles is None):
                # Recalculate the target radius
                targetRd[i] = np.average([targetRd[i], circles[0, 2]])  # radius

                p1 = (int(bbox[0] + ROI[i, 0]), int(bbox[1] + ROI[i, 1]))
                p2 = (int(p1[0] + bbox[2]), int(p1[1] + bbox[3]))
                bboxPosition[i, kFrame, :] = bbox
                # circlePosition[i, kFrame, :] = circles[0, :]
                # Circles positions are relative to the bounding boxes
                # They need to be added to bbox and ROI top left corners so they are in the global axes
                circlePosition[i, kFrame, 0] = int(round((circles[0, 0] + targetRd[i] + targetSz[i, 0] - bbox[0])/2))
                circlePosition[i, kFrame, 1] = int(round((circles[0, 1] + targetRd[i] + targetSz[i, 1] - bbox[1])/2))
                circlePosition[i, kFrame, 2] = int(round(targetRd[i]))
                cv.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                cv.circle(frame, (int(circlePosition[i, kFrame, 0] + p1[0]), int(circlePosition[i, kFrame, 1] + p1[1])),
                          int(circlePosition[i, kFrame, 2]), (255, 0, 255), 2)

    if len(targetRd) == numTargets:
        while True:
            # # Reading the video frame
            ret, originalFrame = videoCapture.read()
            if not ret: break

            frame = originalFrame
            kFrame += 1
            dialogTxt = 'Frame #{} / {}'.format(kFrame+1, frameCount)
            print(dialogTxt)

            for i in range(0, numTargets):
                # Select Region of Interest for each target
                croppedFrame = originalFrame[
                               int((1. / fyVideoRsz) * ROI[i, 1]):int((1. / fyVideoRsz) * (ROI[i, 1] + ROI[i, 3])),
                               int((1. / fxVideoRsz) * ROI[i, 0]):int((1. / fxVideoRsz) * (ROI[i, 0] + ROI[i, 2]))].copy()
                # ret = tracker[i].init(croppedFrame, bboxPosition[i, kFrame-1, :])
                # if not ret:
                #    ret = tracker[i].init(croppedFrame, targetTemplate[i, :] // 2)
                # Check if performance improves by doing the .init with the bbox found in the previous step

                # Find the tracked area in the cropped frame
                ret, bbox = tracker[i].update(croppedFrame)
                if ret:
                    p1 = (int(bbox[0] + ROI[i, 0]), int(bbox[1] + ROI[i, 1]))
                    p2 = (int(p1[0] + bbox[2]), int(p1[1] + bbox[3]))
                    bboxFrame = croppedFrame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                    circles = findCircle(bboxFrame, targetRd[i], lowerShadeColor, upperShadeColor)
                    if circles is None:
                        circlePosition[i, kFrame, 0] = int(-99999)#int(circlePosition[i, kFrame-1, 0])
                        circlePosition[i, kFrame, 1] = int(-99999)#int(circlePosition[i, kFrame-1, 1])
                        circlePosition[i, kFrame, 2] = int(round(targetRd[i]))
                    else:
                        bboxPosition[i, kFrame, :] = bbox
                        circlePosition[i, kFrame, 0] = int(round(circles[0, 0] + p1[0]))
                        circlePosition[i, kFrame, 1] = int(round(circles[0, 1] + p1[1]))
                        circlePosition[i, kFrame, 2] = int(round(targetRd[i]))

                    # circlePosition[i, kFrame, 0] = int(round(circlePosition[i, 0, 0] + p1[0]))
                    # circlePosition[i, kFrame, 1] = int(round(circlePosition[i, 0, 1] + p1[1]))
                    # circlePosition[i, kFrame, 2] = int(round(targetRd[i]))
                    if showVideoOnScreen or recordVideoOutput:
                        cv.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                        cv.circle(frame, (int(circlePosition[i, kFrame, 0]),
                                          int(circlePosition[i, kFrame, 1])),
                                  int(circlePosition[i, kFrame, 2]), (255, 0, 255), 2)
                else:
                    cv.putText(frame, "Failed", (100, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    break
            if showVideoOnScreen or recordVideoOutput:
                printTxt = 'Frame #{}/{}'.format(kFrame+1, frameCount)
                cv.putText(frame, printTxt, (120, 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 50), 2)
                if showVideoOnScreen:
                    cv.imshow("Tracking", frame)
                    if cv.waitKey(1) & 0xFF == ord('q'): break
                if recordVideoOutput:
                    videoRecording.write(frame)

        for i in range(0, numTargets):
            circlePosition[i, 0, 0] = circlePosition[i, 0, 0] + bboxPosition[i, 0, 0] + ROI[i, 0]
            circlePosition[i, 0, 1] = circlePosition[i, 0, 1] + bboxPosition[i, 0, 1] + ROI[i, 1]

        pixelFct = ballRealSize/np.average(2*targetRd)
        dataFileName = videoFileName.replace('_converted.mp4', '_Surge.csv')
        pd.DataFrame(pixelFct*circlePosition[:, :, 0].transpose()).to_csv(dataFileName) #, index_label="Index", header=['T1', 'T2']
        dataFileName = videoFileName.replace('_converted.mp4', '_Heave.csv')
        pd.DataFrame(pixelFct*circlePosition[:, :, 1].transpose()).to_csv(dataFileName)


videoCapture.release()
if recordVideoOutput:
    videoRecording.release()
cv.destroyAllWindows()

# figH = plt.figure()
# plt.plot(range(0, frameCount), circlePosition[0, :, 1])
# plt.show()
# # plt.plot(range(0, frameCount), circleCenterPosition[1, :, 1])
# # plt.plot(range(0, frameCount), circleCenterPosition[2, :, 1])
# # plt.plot(range(0, frameCount), circleCenterPosition[3, :, 1])
# # plt.plot(range(0, frameCount), circleCenterPosition[4, :, 1])
# # plt.plot(range(0, frameCount), circleCenterPosition[5, :, 1])
