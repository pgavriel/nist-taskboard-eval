import os, sys
import torch
import math
import numpy as np
import apriltag
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from PIL import Image

def nothing(x):
    pass

def info(x, str = 'INFO:'):
    print(str)
    print(type(x))
    print(len(x))
    print(x)

def draw_poly(image, pts, highlight=1, c1=(0,255,0), c2=(0,0,255), center=True):
    for i in range(0,len(pts)):
        if i == highlight:
            color = c2
        else:
            color = c1
        image = cv2.line(image,(pts[i][0],pts[i][1]),(pts[i-1][0],pts[i-1][1]),color,2)
    if center:
        ctr_pt = [int(sum(x)/len(x)) for x in zip(*pts)]
        image = cv2.circle(image,(ctr_pt[0],ctr_pt[1]),2,c1,-1)

    return image

def draw_points(image, pts, color=(0,255,0), radius=2, thickness=-1):
    for p in pts:
        image = cv2.circle(image,(p[0],p[1]),radius,color,thickness)

    return image

def sortpoints(board_corners,tag_corners):
    print("\nSORTING POINTS:")
    tag = tag_corners.copy()
    board = board_corners.copy()
    ordered = np.zeros((4,2))
    print("Tag Corners:\n",tag_corners)
    print("Unordered Board Corners:\n",board_corners)
    for i in range(0,len(board)):
        dists = []
        for p in tag:
            dists.append(math.sqrt( ((board[i][0]-p[0])**2)+((board[i][1]-p[1])**2) ))
        minpos = dists.index(min(dists))
        ordered[minpos] = board[i]
        print(i,"-\tDists -",dists,"\tMinPos -",minpos)

    print("Ordered Board Corners:\n",ordered)
    ordered = np.array(ordered, dtype=np.uint32)
    return ordered


def getLargestContourCenter(mat):
    gray = cv2.cvtColor(mat,cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        area = cv2.contourArea(c)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(mat, [box], 0,(0,255,0),1)
        center = [int(sum(x)/len(x)) for x in zip(*box)]
        mat[center[0],center[1]] = (255,0,0)
        print("BOX: ",box)
        print("CENTER: ",center)

    else:
        return mat, None

    return mat, center


def isolate_board(image,thresh=120,area_lb=0,area_ub=100000):
    print("\nISOLATING TASKBOARD:")
    area = 0
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray,9,75,75)
    ret,th1 = cv2.threshold(smooth,thresh,255,cv2.THRESH_BINARY)
    stage1_img = cv2.cvtColor(th1,cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        area = cv2.contourArea(c)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(stage1_img, [box], 0,(0,255,0),2)
        if area < area_lb or area > area_ub:
            print("Found contour was not within specified bounds, not returning corners.")
            print("Contour Area:",area,"\tLBound:",area_lb," UBound:",area_ub)
            box = None

    else:
        box = None

    print("Bounding Box Found:\n",box)
    print("Region Area:",area)
    return stage1_img, smooth, box

def refine_corners(image,corners,box_size=50,line_thresh=110):
    print("\nREFINING CORNERS:")
    image_bgr = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    edges = cv2.Canny(image,100,200)
    stage2_img = edges.copy()
    stage2_img = cv2.cvtColor(stage2_img,cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, line_thresh, None, 0, 0)
    line_mat = edges.copy()
    line_mat[:,:] = 0
    if lines is not None:
        print("Lines found: ",len(lines))
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(line_mat, pt1, pt2, 255, 1, cv2.LINE_AA)
            cv2.line(stage2_img, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
    harris = cv2.cornerHarris(line_mat,2,3,0.04)
    harris = cv2.dilate(harris,None)
    ret, harris = cv2.threshold(harris,0.01*harris.max(),255,0)
    harris = np.uint8(harris)
    harris_bgr = cv2.cvtColor(harris,cv2.COLOR_GRAY2BGR)
    line_mat_bgr = cv2.cvtColor(line_mat,cv2.COLOR_GRAY2BGR)
    c_gray = []
    c_line = []
    c_line_mat = []
    c_harris = []
    print(corners.shape,"CORNERS")
    for c in corners:
        print("Corner Shape", c.shape)
        x, y = int(c[0]-(box_size/2)), int(c[1]-(box_size/2))
        #x2, y2 = int(x+c.shape[0]), int(y+c.shape[1])
        stage2_img = cv2.rectangle(stage2_img,(x,y),(x+box_size,y+box_size),(0,255,255),2)
        print("x", x, "\ty", y)
        found_corner = image_bgr[y:y+box_size,x:x+box_size]
        c_gray.append(found_corner)
        found_corner = stage2_img[y:y+box_size,x:x+box_size]
        c_line.append(found_corner)
        found_corner = line_mat_bgr[y:y+box_size,x:x+box_size]
        c_line_mat.append(found_corner)
        found_corner = harris_bgr[y:y+box_size,x:x+box_size]
        c_harris.append(found_corner)
    # info(c_gray,'GRAY')
    # info(c_line,"LINE")
    # info(c_harris,'HARRIS')
    c_subpix = []
    #c_subpix = corners
    for c, b in zip(c_harris,corners):
        xpos, ypos = int(b[0]-(c.shape[0]/2)), int(b[1]-(c.shape[1]/2))

        if not np.any(c):
            c_subpix.append((b[0],b[1]))
            print("Corner",b,"region appears empty, defaulting to rough corner.")
            continue

        c, ctr_pt = getLargestContourCenter(c)
        if ctr_pt is None:
            c_subpix.append((b[0],b[1]))
            print("ERR: No center point returned")
            continue

        xpos = int(xpos + ctr_pt[0])
        ypos = int(ypos + ctr_pt[1])
        c_subpix.append((xpos,ypos))

    try:
        stack1 = np.hstack(c_gray)
        stack2 = np.hstack(c_line)
        stack3 = np.hstack(c_line_mat)
        stack4 = np.hstack(c_harris)
        vstack = np.vstack((stack1,stack2,stack3,stack4))

        return stage2_img, c_subpix, vstack
    except Exception as e:
        print("Error with corner mats")
        print(e)
        return stage2_img, c_subpix, None


def saveTaskboard(dir,image,box,component,state,size):
    saveoriginal = False
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

def saveImage(dir,image,state,descriptor):
    if isinstance(state, int):
        savedir = os.path.join(dir,str('{:02}'.format(state)))
    else:
        savedir = os.path.join(dir,state)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    files = len([name for name in os.listdir(savedir) if os.path.isfile(os.path.join(savedir, name))])
    #filename = str('{:02}'.format(state)) + '-' + str('{:03}'.format(files)) + '-' + descriptor + '.png'
    filename = str('{:03}'.format(files)) + '-' + descriptor + '.png'
    rgb_im = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    rgb_im = Image.fromarray(rgb_im)
    path = os.path.join(savedir,filename)
    print("Saving '",filename,"' to '",savedir,"'")
    rgb_im.save(path)
    pass

def find_homography(image,corners,size=400):
    print("\nWARPING PERSPECTIVE:")
    print("Output Size:",size)
    pts1 = np.float32(corners)
    print('PTS1:',pts1)
    pts2 = np.float32([[0,0],[size,0],[size,size],[0,size]])
    print('PTS2:',pts2)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    warped = cv2.warpPerspective(image,M,(size,size))
    return warped


def main(DIR,step_through,img_file_desc,img_file_loc=None):

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
            original = color_image.copy()
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            #CV Winows
            cv2.namedWindow('StageOne', cv2.WINDOW_AUTOSIZE)
            cv2.createTrackbar('speed','StageOne',1000,2000,nothing)
            cv2.createTrackbar('thresh','StageOne',120,255,nothing)
            cv2.createTrackbar('corner_size','StageOne',50,200,nothing)
            cv2.createTrackbar('line_thresh','StageOne',110,255,nothing)
            cv2.namedWindow('StageTwo', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('Warped', cv2.WINDOW_AUTOSIZE)
            if img_file_loc is None:
                cv2.createTrackbar('board_state','Warped',5,20,nothing)
            cv2.namedWindow('Corners', cv2.WINDOW_AUTOSIZE)
            delay = cv2.getTrackbarPos('speed','StageOne')
            wait = 0 if step_through else delay

            # Detect apriltag
            detector = apriltag.Detector()
            result = detector.detect(gray)

            warped = np.zeros((gray.shape[0],gray.shape[1],3),np.uint8)
            err_img = np.zeros((gray.shape[0],gray.shape[1],3),np.uint8)
            y = int(warped.shape[0]/4)
            x = int(warped.shape[1]/4)
            err_img = cv2.line(err_img,(x,y),(3*x,3*y),(0,0,255),5)
            err_img = cv2.line(err_img,(3*x,y),(x,3*y),(0,0,255),5)
            if (len(result) == 0):
                print("No apriltag detected.")
                #warped = np.zeros((gray.shape[0],gray.shape[1],3),np.uint8)
                # Draw an X to indicate tag not detected
                y = int(warped.shape[0]/4)
                x = int(warped.shape[1]/4)
                warped = cv2.line(warped,(x,y),(3*x,3*y),(0,0,255),5)
                warped = cv2.line(warped,(3*x,y),(x,3*y),(0,0,255),5)
            else:
                #APRIL TAG DETECTED
                try:
                    #Stage One
                    thresh = cv2.getTrackbarPos('thresh','StageOne')
                    stage1_img, smooth_img, rough_corners = isolate_board(color_image,thresh,55000,75000)
                    if rough_corners is None:
                        stack_1 = np.hstack((stage1_img,err_img))
                        cv2.imshow('StageOne',stack_1)
                        key = cv2.waitKey(wait)
                        if key == ord('q'):
                            break
                        continue

                    #Sort corners to align with Apriltag
                    tag_corners = result[0].corners.astype('int32')
                    rough_corners = sortpoints(rough_corners,tag_corners)

                    #Stage Two
                    corner_size = cv2.getTrackbarPos('corner_size','StageOne')
                    line_thresh = cv2.getTrackbarPos('line_thresh','StageOne')
                    stage2_img, tb_corners, crops = refine_corners(smooth_img,rough_corners,corner_size,line_thresh)

                    #Warp Image Perspective
                    warped = find_homography(color_image,tb_corners)

                    #Draw found apriltag and taskboard on camera image
                    corner_img = color_image.copy()
                    corner_img = draw_points(corner_img,rough_corners,(0,0,255))
                    corner_img = draw_points(corner_img,tb_corners,(0,255,0))
                    color_image = draw_poly(color_image,tag_corners)
                    color_image = draw_poly(color_image,tb_corners)

                    #Display images
                    stack_1 = np.hstack((stage1_img,stage2_img))
                    cv2.imshow('StageOne',stack_1)
                    stack_2 = np.hstack((corner_img,color_image))
                    cv2.imshow('StageTwo',stack_2)
                    cv2.imshow('Warped', warped)
                    if crops is not None:
                        cv2.imshow('Corners',crops)
                except Exception as e:
                    print("Something went wrong")
                    print(e)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)




            # Get user key input
            key = cv2.waitKey(wait) & 0xFF
            # Save screenshot (S pressed)
            if key == ord('s'):
                if img_file_loc is None:
                    state = cv2.getTrackbarPos('board_state','Warped')
                    saveImage(DIR,warped,state,img_file_desc)
                else:
                    saveImage(DIR,warped,img_file_loc,img_file_desc)
                #saveImage(DIR,warped,state,"Warped")
                #saveImage(DIR,original,state,"Original")
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
    print("Data Directory:", datadir)
    print("Arguments Given: ",len(sys.argv))
    print("Args: ", sys.argv)
    img_file_loc = None
    if (len(sys.argv)==1):
        img_file_desc = 'warped'
    if (len(sys.argv)>=2):
        img_file_desc = sys.argv[1]
    if (len(sys.argv)==3):
        img_file_loc = sys.argv[2]
    # Play stream or step frame by frame with keyboard
    step_through = False
    main(datadir,step_through,img_file_desc,img_file_loc)
