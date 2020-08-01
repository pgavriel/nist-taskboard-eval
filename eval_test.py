import os, sys
import csv
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from PIL import Image

def nothing(x):
    pass

def evalComponent(diff_image,eval_image,comp):
    # s = Crop Size
    s = int(25 * comp[2])
    mask = np.zeros(diff_image.shape[:2], np.uint8)
    mask[comp[1]-s:comp[1]+s, comp[0]-s:comp[0]+s] = 255
    #cv2.imshow('mask',mask)
    #diff_crop = diff_image[comp[1]-s:comp[1]+s, comp[0]-s:comp[0]+s]
    #eval_crop = eval_image[comp[1]-s:comp[1]+s, comp[0]-s:comp[0]+s]
    #diff_hist = cv2.calcHist([diff_crop],[0],None,[256],[0,256])
    #eval_hist = cv2.calcHist([eval_crop],[0],None,[16],[0,256])
    diff_hist = cv2.calcHist([diff_image],[0],mask,[256],[0,256])
    eval_hist = cv2.calcHist([eval_image],[0],mask,[256],[0,256])
    plt.plot(diff_hist)
    plt.plot(eval_hist)

    hist_diff = cv2.compareHist(diff_hist,eval_hist,cv2.HISTCMP_BHATTACHARYYA)
    print("HIST DIFF: ",hist_diff)

    #print("DIFF HIST\n",diff_hist)
    #print("EVAL HIST\n",eval_hist)
    #cv2.imshow('diff',diff_crop)
    #cv2.imshow('eval',eval_crop)
    print("Component:", comp)

    plt.show()
    pass

def evalAllComponents(diff_image,eval_image,comp_list):
    font = cv2.FONT_HERSHEY_SIMPLEX
    print("Evaluating Components with Diff and Diff2")
    diff_types = ['Correlation','Chi-Square','Intersection','Bhattacharyya']
    eval_type = 3
    eval_thresh = 0.3
    print("Evaluating using",diff_types[eval_type],"Histogram Comparison with threshold",str(eval_thresh))

    comp_number = 1
    scores = []
    for comp in comp_list:
        s = int(25 * comp[2])
        mask = np.zeros(diff_image.shape[:2], np.uint8)
        mask[comp[1]-s:comp[1]+s, comp[0]-s:comp[0]+s] = 255
        diff_hist = cv2.calcHist([diff_image],[0],mask,[256],[0,256])
        eval_hist = cv2.calcHist([eval_image],[0],mask,[256],[0,256])

        # Plot Histograms for focused component, and save it
        plt.clf()
        plt.plot(diff_hist)
        plt.plot(eval_hist)
        plt.legend(['Left','Right'],loc=9)
        title = 'Component '+str(comp_number)+' Histogram'
        plt.title(title)
        plt.xlabel('Pixel Value')
        file = 'plots/comp'+str(comp_number)+'plot.png'
        plt.savefig(file)

        # Compare the Histograms using all methods
        diff_correl = cv2.compareHist(diff_hist,eval_hist,cv2.HISTCMP_CORREL)
        diff_chi = cv2.compareHist(diff_hist,eval_hist,cv2.HISTCMP_CHISQR)
        diff_inter = cv2.compareHist(diff_hist,eval_hist,cv2.HISTCMP_INTERSECT)
        diff_bhatta = cv2.compareHist(diff_hist,eval_hist,cv2.HISTCMP_BHATTACHARYYA)
        diffs = [diff_correl,diff_chi,diff_inter,diff_bhatta]

        # Load saved plot, add crop images and compared values
        plot_img = cv2.imread(file)
        #Attach crop
        stack = np.hstack((diff_image[comp[1]-s:comp[1]+s, comp[0]-s:comp[0]+s],eval_image[comp[1]-s:comp[1]+s, comp[0]-s:comp[0]+s]))
        stack_3ch = cv2.cvtColor(stack,cv2.COLOR_GRAY2BGR)
        plot_img[59:59+stack.shape[0],576-stack.shape[1]:576,:] = stack_3ch

        #Attach text
        y_off = 18
        ypos = 140
        cv2.putText(plot_img,"Histogram Comparisons:",(285,123), font, 0.3,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(plot_img,"CORR: "+str(round(diffs[0],2)),(285,ypos), font, 0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(plot_img,"CHSQ: "+str(round(diffs[1],2)),(285,ypos+y_off), font, 0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(plot_img,"INTR: "+str(round(diffs[2],2)),(285,ypos+(y_off*2)), font, 0.5,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(plot_img,"BHAT: "+str(round(diffs[3],2)),(285,ypos+(y_off*3)), font, 0.5,(0,0,0),1,cv2.LINE_AA)
        combined = Image.fromarray(plot_img)
        combined.save(file)


        # Attempt to Evaluate Component
        if (diffs[eval_type] < eval_thresh):
            state = 0
        else:
            state = 2

        print("Component:",comp_number,"\tCORR:",round(diff_correl,2),\
                                        "\tCHSQ:",round(diff_chi,2),\
                                        "\tINTR:",round(diff_inter,2),\
                                        "\tBHAT:",round(diff_bhatta,2),\
                                        "\tState:",state)
        scores.append((comp_number,diffs[eval_type],state))
        comp_number = comp_number + 1
    pass

def getROI(csv_file,width,height):
    im_width = width
    im_height = height
    roi_list = []
    with open(csv_file, newline='') as roi_csv:
        reader = csv.DictReader(roi_csv)
        for row in reader:
            x = float(row['xval']) * im_width
            y = float(row['yval']) * im_height
            region_scale = float(row['scale'])
            roi_list.append((int(x),int(y),region_scale))
    return roi_list

def main(DIR,eval_image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    step_through = True
    cv2.namedWindow('eval_test', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('thresh','eval_test',25,255,nothing)

    emptyboard = cv2.imread(os.path.join(DIR,'00/00-001-Warped.png'))
    emptyboard = cv2.bilateralFilter(emptyboard,9,75,75)
    emptyboard_g = cv2.cvtColor(emptyboard, cv2.COLOR_BGR2GRAY)
    emptyboard_g = cv2.adaptiveThreshold(emptyboard_g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    #emptyboard_g3 = cv2.cvtColor(emptyboard_g,cv2.COLOR_GRAY2BGR)
    fullboard = cv2.imread(os.path.join(DIR,'20/20-009-Warped.png'))
    fullboard = cv2.bilateralFilter(fullboard,9,75,75)
    fullboard_g = cv2.cvtColor(fullboard, cv2.COLOR_BGR2GRAY)
    fullboard_g = cv2.adaptiveThreshold(fullboard_g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imshow('emptyboard',emptyboard_g)
    cv2.imshow('full',fullboard_g)
    ad = cv2.absdiff(emptyboard_g,fullboard_g)
    cv2.imshow('ad',ad)
    #fullboard_g3 = cv2.cvtColor(fullboard_g,cv2.COLOR_GRAY2BGR)

    test_img = cv2.imread(eval_image)
    test_img = cv2.bilateralFilter(test_img,9,75,75)
    #test_img = cv2.bilateralFilter(test_img,9,75,75)
    test_g = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    #eth1 = cv2.adaptiveThreshold(emptyboard_g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    roi = getROI('tb_roi.csv',emptyboard.shape[0],emptyboard.shape[1])
    print("ROI LIST:\n",roi)
    while True:
        threshval = cv2.getTrackbarPos('thresh','eval_test')
        diff = cv2.absdiff(emptyboard,fullboard)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        #_, diff = cv2.threshold(diff,threshval,255,cv2.THRESH_BINARY)

        diff2 = cv2.absdiff(test_img,fullboard)
        diff2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2GRAY)
        #_, diff2 = cv2.threshold(diff2,threshval,255,cv2.THRESH_BINARY)

        diff3 = cv2.absdiff(diff,diff2)

        diff4 = cv2.absdiff(test_img,emptyboard)
        diff5 = cv2.absdiff(test_img,fullboard)
        #diff2 = cv2.absdiff(fth1,eth1)
        #diff3 = cv2.absdiff(emptyboard_g,fullboard_g)
        #diff2_3ch = cv2.cvtColor(diff2,cv2.COLOR_GRAY2BGR)
        cv2.putText(diff,"Diff: Fullboard - EmptyBoard",(5,diff.shape[1]-10), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(diff2,"Diff2: Fullboard - Eval_Image",(5,diff2.shape[1]-10), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(diff3,"Diff3: Diff - Diff2",(5,diff3.shape[1]-10), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        stack = np.hstack((diff,diff2,diff3  ))
        cv2.imshow('eval_test', stack)

        scores = evalAllComponents(diff,diff2,roi)
        wait = 0 if step_through else 1
        key = cv2.waitKey(wait) & 0xFF
        if key == ord('p'):
            step_through = not step_through
        elif key == ord('q'):
            break


if __name__ == '__main__':
    # Set directory to save images
    cwd = os.getcwd()
    datadir = os.path.join(cwd,'data')
    print("Data Directory:", datadir)
    # Play stream or step frame by frame with keyboard
    step_through = False
    eval_image = os.path.join(datadir,'09/09-000-Warped.png')
    main(datadir,eval_image)
