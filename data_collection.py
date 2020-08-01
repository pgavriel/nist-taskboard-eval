import os, sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from PIL import Image

def nothing(x):
    pass

# Assumes the same orientation is sent every time. Reordered to clockwise.
# Likely needs a better solution later on
def sortpoints(pts):
    if pts.shape == (4,2):
        return np.array([pts[2],pts[1],pts[3],pts[0]])
    else:
        print('ERR: Wrong shape, send (4,2) points.')
    pass

def getLargestContour(image, contours, minsize = 0):
    # print('Contours:', contours)
    maxarea = 0
    focus = 0
    for c in range(len(contours)):
        area = cv2.contourArea(contours[c])
        if (area > maxarea):
            maxarea = area
            focus = c
    if (maxarea != 0):
        if (maxarea > minsize) or True:
            print('Focus is contour',focus,'with area',maxarea)
            rect = cv2.minAreaRect(contours[focus])
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            return box
        else:
            print('Contour found but it looks too small. ( Area =',maxarea,'Minsize =',minsize,')')
    pass

def saveTaskboard(dir,image,box,component,state,size):
    saveoriginal = True
    # denoise = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    if saveoriginal:
        original = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    pts1 = np.float32(box)
    print('PTS1:',pts1)
    pts2 = np.float32([[0,0],[size,0],[0,size],[size,size]])
    print('PTS2:',pts2)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    warped = cv2.warpPerspective(image,M,(size,size))
    rgb_im = Image.fromarray(cv2.cvtColor(warped,cv2.COLOR_BGR2RGB))

    # Create save directory
    savedir = os.path.join(dir,str('{:02}'.format(component)))
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    files = len([name for name in os.listdir(savedir) if os.path.isfile(os.path.join(savedir, name))])
    filename = str(state) + '-' + str('{:03}'.format(files)) + '-RGBWarped.png'
    path = os.path.join(savedir,filename)
    rgb_im = rgb_im.transpose(Image.ROTATE_270)
    rgb_im = rgb_im.transpose(Image.FLIP_LEFT_RIGHT)
    rgb_im.save(path)
    if saveoriginal:
        # Save original
        filename = str(state) + '-' + str('{:03}'.format(files)) + '-RGBOriginal.png'
        path = os.path.join(savedir,filename)
        original.save(path)
    print("Snapshot for Component ",component," in State ",state)
    print("Image \"",filename,"\" saved to ",path)
    pass


def main(DIR,step_through):



    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Main loop
    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue



            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Blur to smooth images
            blur = cv2.bilateralFilter(gray,9,75,75)
            blur = cv2.medianBlur(blur,5)
            th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
            # Find edges
            edges = cv2.Canny(blur,50,150,apertureSize = 3)
            kernel = np.ones((5,5),np.uint8)
            edges = cv2.dilate(edges,kernel,iterations = 1)
            # Find contours and try to pick out task board
            contours , heir = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            taskboard = getLargestContour(edges,contours,60000)
            print(type(taskboard))
            print('TASKBOARD: \n',taskboard)
            print('------')

            color_cont = color_image.copy()
            color_tb = color_image.copy()
            if taskboard is not None:
                print('Drawing contours.')
                cv2.drawContours(color_cont, contours, -1, (0,0,255), 3)
                cv2.drawContours(color_tb, [taskboard], 0, (0,255,0), 3)
                taskboard = sortpoints(taskboard)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Merge images into a 2x2 square to display
            blur_3chan = cv2.cvtColor(blur,cv2.COLOR_GRAY2BGR)
            edges_3chan = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
            images = np.hstack((blur_3chan, edges_3chan))
            images2 = np.hstack((color_cont,color_tb))
            stack = np.vstack((images,images2))

            # Show images and trackbars
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', stack)
            cv2.createTrackbar('componentTracker','RealSense',1,20,nothing)
            cv2.createTrackbar('stateTracker','RealSense',0,2,nothing)

            # Get user key input
            wait = 0 if step_through else 1
            key = cv2.waitKey(wait) & 0xFF
            # Save screenshot (S pressed)
            if key == ord('s'):
                if taskboard is not None:
                    component = cv2.getTrackbarPos('componentTracker','RealSense')
                    state = cv2.getTrackbarPos('stateTracker','RealSense')
                    saveTaskboard(DIR, color_image, taskboard, component, state, 400)
                else:
                    print('Attempted screenshot but nothing saved (No taskboard found)')
            # Toggle frame step mode (P key)
            elif key == ord('p'):
                step_through = not step_through
            # Quit program (Q pressed)
            elif key == ord('q'):
                break
    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == '__main__':
    # Set directory to save images
    cwd = os.getcwd()
    datadir = os.path.join(cwd,'data')
    print(datadir)
    # Play stream or step frame by frame with keyboard
    step_through = False
    main(datadir,step_through)
