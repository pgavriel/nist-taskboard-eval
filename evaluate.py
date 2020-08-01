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

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def processTB(image,k3=25,debug=False):
    #Kernels for various operations
    kernel = np.ones((2,2),np.uint8)
    kernel2 = np.ones((4,4),np.uint8)
    kernel3 = np.ones((k3,k3),np.uint8)

    #Copy, smooth, and greyscale given image, and copied for later use
    tb = image.copy()
    tb = cv2.bilateralFilter(tb,9,75,75)
    tb = cv2.cvtColor(tb, cv2.COLOR_BGR2GRAY)
    board = tb.copy()
    mask = tb.copy()
    mask[:] = 0

    #Further processing to remove background leaving only the components
    tb = cv2.adaptiveThreshold(tb,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    tb = cv2.bitwise_not(tb)
    tb = cv2.morphologyEx(tb, cv2.MORPH_OPEN, kernel)
    tb = cv2.dilate(tb,kernel2,iterations = 2)
    #cv2.imshow('processTB/tb',tb)

    #Find contours within thresholded taskboard
    contours, hierarchy = cv2.findContours(tb,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print("CONTOURS", len(contours))
    #Draw contours filled in, onto blank mask
    mask = cv2.drawContours(mask, contours, -1, 255, -1)
    #Attempt to close holes in the mask
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel3)
    #Invert mask and fill in small contours, then invert again to remove holes
    mask = cv2.bitwise_not(mask)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print("INVMASK Contours", len(contours))
    holes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 2000:
            holes.append(c)
    mask = cv2.drawContours(mask, holes, -1, 0, -1)
    mask = cv2.bitwise_not(mask)
    mask_cp = mask.copy()
    cv2.imshow('maskcp',mask_cp)
    cv2.imshow('processTB/mask',mask)

    #bitwise_and between smooth gray image and mask to remove background
    masked = cv2.bitwise_and(mask,board,mask)
    #cv2.imshow('processTB/masked',masked)

    if debug:
        image_g = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        stack = np.hstack( (image_g,mask_cp,masked) )
        cv2.imshow('processTB DEBUG',stack)
    return masked

def processTB2(image):
    tb = image.copy()
    tb = cv2.bilateralFilter(tb,9,75,75)
    tb = cv2.cvtColor(tb, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('pre',tb)
    tb = cv2.normalize(tb,  tb, 0, 255, cv2.NORM_MINMAX)
    #cv2.imshow('post',tb)
    return tb
    pass

def evalAllComponents(ref,eval,comp_list,original,thresh):
    print("EVALUATING COMPONENTS:")
    print(" # Component\tMSE\tMSE/Size\tPresent?")
    comp_number = 1
    scores = []
    for comp in comp_list:
        name = comp[0]
        x = comp[1]
        y = comp[2]
        scale = int(25 * comp[3])
        ref_crop = ref[y-scale:y+scale,x-scale:x+scale]
        eval_crop = eval[y-scale:y+scale,x-scale:x+scale]
        error = mse(ref_crop,eval_crop)
        error_f = str('{:08.2f}'.format(error))
        serror = error/ref_crop.shape[0]
        serror_f = str('{:08.2f}'.format(serror))
        present = False if (serror>thresh) else True
        if not present:
            cv2.rectangle(original,(x-scale,y-scale),(x+scale,y+scale),(0,0,255),2)
        else:
            cv2.rectangle(original,(x-scale,y-scale),(x+scale,y+scale),(0,255,0),2)
        #stack = np.hstack((ref_crop,eval_crop))
        #print(name, x, y, scale)
        #cv2.imshow(name,stack)
        #cv2.waitKey(0)
        print(str('{:02}'.format(comp_number)),name,"\t",error_f,serror_f,present)
        comp_number = comp_number + 1
    cv2.imshow('Evaluation',original)
    pass

def evalAllComponents2(ref,vs_empty,vs_full,comp_list,original,thresh):
    print("EVALUATING COMPONENTS:")
    print(" # Component\tEmpty MSE\tFull MSE\tScore\t\tPresent?")
    comp_number = 1
    scores = []
    for comp in comp_list:
        name = comp[0]
        x = comp[1]
        y = comp[2]
        scale = int(25 * comp[3])
        ref_crop = ref[y-scale:y+scale,x-scale:x+scale]
        vemp_crop = vs_empty[y-scale:y+scale,x-scale:x+scale]
        vfull_crop = vs_full[y-scale:y+scale,x-scale:x+scale]
        e_error = mse(ref_crop,vemp_crop)
        f_error = mse(ref_crop,vfull_crop)
        score = f_error - e_error
        eerror_f = str('{:07.1f}'.format(e_error))
        ferror_f = str('{:07.1f}'.format(f_error))
        score_f = str('{:+05.1f}'.format(score))
        present = False if (score<0) else True
        if not present:
            cv2.rectangle(original,(x-scale,y-scale),(x+scale,y+scale),(0,0,255),2)
        else:
            cv2.rectangle(original,(x-scale,y-scale),(x+scale,y+scale),(0,255,0),2)
        #stack = np.hstack((ref_crop,eval_crop))
        #print(name, x, y, scale)
        #cv2.imshow(name,stack)
        #cv2.waitKey(0)
        print(str('{:02}'.format(comp_number)),name,"\t",eerror_f,"\t",ferror_f,"\t",score_f,"\t",present)
        comp_number = comp_number + 1
    cv2.imshow('Evaluation',original)
    pass

def evalAllComponents3(eval,empty,full,comp_list,original,thresh):
    debug = False
    print("EVALUATING COMPONENTS:")
    if debug: print(" # Component\tEmpty MSE\tFull MSE\tScore\t\tPresent?")
    comp_number = 1
    scores = []
    for comp in comp_list:
        name = comp[0]
        x = comp[1]
        y = comp[2]
        scale = int(25 * comp[3])
        eval_crop = eval[y-scale:y+scale,x-scale:x+scale]
        emp_crop = empty[y-scale:y+scale,x-scale:x+scale]
        full_crop = full[y-scale:y+scale,x-scale:x+scale]
        #stack = np.hstack( (emp_crop,eval_crop,full_crop) )
        #cv2.imshow(name,stack)
        e_error = mse(eval_crop,emp_crop)
        f_error = mse(eval_crop,full_crop)
        score = e_error - f_error
        scores.append(int(score))
        eerror_f = str('{:07.1f}'.format(e_error))
        ferror_f = str('{:07.1f}'.format(f_error))
        score_f = str('{:=+6d}'.format(int(score)))
        present = False if (score<0) else True
        if not present:
            cv2.rectangle(original,(x-scale,y-scale),(x+scale,y+scale),(0,0,255),2)
        else:
            cv2.rectangle(original,(x-scale,y-scale),(x+scale,y+scale),(0,255,0),2)

        if debug: print(str('{:02}'.format(comp_number)),name,"\t",eerror_f,"\t",ferror_f,"\t",score_f,"\t",present)
        comp_number = comp_number + 1
    cv2.imshow('Evaluation',original)
    print("SCORES\n",scores)
    return scores,original

def drawScores(image,roi,scores):
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r, score in zip(roi,scores):
        x = r[1]
        y = r[2]
        scale = int(25 * r[3])
        if score < 0:
            color = (0,0,255)
        else:
            color = (0,255,0)
        cv2.rectangle(img,(x-scale,y-scale),(x+scale,y+scale),color,2)
        cv2.putText(img,str(int(score)),(x-scale,y-scale-2), font, 0.4,color,1,cv2.LINE_AA)
    return img

def getROI(csv_file,width,height):
    im_width = width
    im_height = height
    roi_list = []
    with open(csv_file, newline='') as roi_csv:
        reader = csv.DictReader(roi_csv)
        for row in reader:
            name = row['name']
            x = float(row['xval']) * im_width
            y = float(row['yval']) * im_height
            region_scale = float(row['scale'])
            roi_list.append((name,int(x),int(y),region_scale))
    return roi_list

def main(DIR, image):
    step_through = True
    cv2.namedWindow('Testing', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('close_kern','Testing',8,35,nothing)
    cv2.namedWindow('Evaluation', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('thresh','Evaluation',100,500,nothing)
    emptylist = os.listdir(os.path.join(DIR,'empty'))
    fulllist = os.listdir(os.path.join(DIR,'full'))
    print("EMPTY\n",emptylist)
    print("FULL\n",fulllist)
    emptyboard = cv2.imread(os.path.join(DIR,'testing/empty.png'))
    fullboard = cv2.imread(os.path.join(DIR,'testing/full.png'))
    queryboard = cv2.imread(image)
    roi = getROI('tb_roi.csv',queryboard.shape[0],queryboard.shape[1])
    original = queryboard.copy()
    cv2.imshow('orig',original)

    while True:
        thresh = cv2.getTrackbarPos('thresh','Evaluation')
        iter = cv2.getTrackbarPos('close_kern','Testing')
        # e = processTB(emptyboard,iter)
        # f = processTB(fullboard,iter)
        # q = processTB(queryboard,iter)
        #e = processTB2(emptyboard)
        #f = processTB2(fullboard)
        q = processTB2(queryboard)
        sum_score = np.zeros(20)
        for e, f in zip(emptylist,fulllist):
            e = 'empty/'+e
            f = 'full/'+f
            emptyboard = cv2.imread(os.path.join(DIR,e))
            fullboard = cv2.imread(os.path.join(DIR,f))
            empty = processTB2(emptyboard)
            full = processTB2(fullboard)
            scores, score_img = evalAllComponents3(q,empty,full,roi,original,thresh)
            sum_score = [a + b for a, b in zip(scores, sum_score)]
            #cv2.imshow(e,score_img)
            print("SUM SCORES\n",sum_score)
        print("FINAL\n",sum_score)
        final_img = drawScores(original,roi,sum_score)
        cv2.imshow('FINAL',final_img)
        #ref = cv2.absdiff(e,f)
        #eval = cv2.absdiff(e,q)
        #vs_emp = cv2.absdiff(e,q)
        #vs_full = cv2.absdiff(f,q)


        #evalAllComponents(ref,eval,roi,original,thresh)
        #evalAllComponents2(ref,vs_emp,vs_full,roi,original,thresh)
        #evalAllComponents3(q,e,f,roi,original,thresh)

        #stack1 = np.hstack( (e,f,q) )
        #cv2.imshow('Testing',stack1)

        #stack2 = np.hstack( (ref,eval) )
        #stack2 = np.hstack( (ref,vs_emp,vs_full) )
        #cv2.imshow('ref/eval',stack2)


        wait = 0 if step_through else 1000
        key = cv2.waitKey(wait) & 0xFF
        if key == ord('p'):
            step_through = not step_through
        elif key == ord('q'):
            break
    pass

if __name__ == '__main__':
    # Set directory to save images
    cwd = os.getcwd()
    datadir = os.path.join(cwd,'data')
    print("Data Directory:", datadir)
    print("Arguments Given: ",len(sys.argv))
    print("Args: ", sys.argv)
    if (len(sys.argv)==1):
        eval_image = os.path.join(datadir,'09/09-000-Warped.png')
    else:
        eval_image = os.path.join(datadir,sys.argv[1])
    print("EVAL IMAGE: ",eval_image)
    main(datadir,eval_image)
