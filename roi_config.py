import os, sys
import csv
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from PIL import Image
from functools import partial

def nothing(x):
    pass

def drawROI(image,roi,focus):
    img = image.copy()
    size = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    component = 1
    for r in roi:
        if component == focus:
            color = (0,255,0)
            #color = (0,0,255)
        else:
            color = (0,0,255)
        x = int(r[1] * image.shape[0])
        y = int(r[2] * image.shape[1])
        s = int(size * r[3])
        img = cv2.rectangle(img,(x-s,y-s),(x+s,y+s),color,1)
        cv2.putText(img,str(component),(x-s+5,y+s-5), font, 0.4,color,1,cv2.LINE_AA)
        component += 1
    return img

def setFocus(x,roi,img):
    focus = x
    if focus == 0:
        cv2.setTrackbarPos('component','ROI_Config',1)
        focus = 1
    region = roi[focus-1]
    x = int(region[1]*1000)
    y = int(region[2]*1000)
    s = int(region[3]*10)
    cv2.setTrackbarPos('xval','ROI_Config',x)
    cv2.setTrackbarPos('yval','ROI_Config',y)
    cv2.setTrackbarPos('scale','ROI_Config',s)
    print("FOCUS",focus,"-",region)
    print("Trackbar Positions:",x,y,s)
    pass

def getROI(csv_file,width,height):
    im_width = width
    im_height = height
    roi_list = []
    with open(csv_file, newline='') as roi_csv:
        reader = csv.DictReader(roi_csv)
        component = 1
        for row in reader:
            name = row['name']
            x = float(row['xval'])
            y = float(row['yval'])
            scale = float(row['scale'])
            roi_list.append((name,x,y,scale))
            print(str('{:02}'.format(component)),name,"\t",x,y,scale)
            component += 1
    return roi_list

def saveROI(roi,out_csv):
    with open(out_csv, 'w', newline='') as newcsv:
        writer = csv.writer(newcsv,delimiter=',')
        writer.writerow(["name","component","xval","yval","scale"])
        component = 1
        for r in roi:
            name = r[0]
            comp = str('{:02}'.format(component))
            x = r[1]
            y = r[2]
            scale = r[3]
            writer.writerow([name,comp,x,y,scale])
            component += 1

    print("Saving new CSV to",out_csv)

def main(img,roi_csv,out_csv):
    step_through = False
    image = cv2.imread(img)
    print("IMAGE SHAPE:\n",image.shape)
    print("ROI LIST:")
    roi = getROI(roi_csv,image.shape[0],image.shape[1])
    print("\nPRESS 'Q' TO QUIT OR 'S' TO SAVE CONFIGURATION\n")

    cv2.namedWindow('ROI_Config', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('component','ROI_Config',1,20,partial(setFocus, roi=roi, img=image))
    cv2.createTrackbar('xval','ROI_Config',1,1000,nothing)
    cv2.createTrackbar('yval','ROI_Config',1,1000,nothing)
    cv2.createTrackbar('scale','ROI_Config',1,20,nothing)
    setFocus(cv2.getTrackbarPos('component','ROI_Config'),roi,image)

    while True:
        focus = cv2.getTrackbarPos('component','ROI_Config')
        x = float(cv2.getTrackbarPos('xval','ROI_Config')/1000)
        y = float(cv2.getTrackbarPos('yval','ROI_Config')/1000)
        s = float(cv2.getTrackbarPos('scale','ROI_Config')/10)
        new_data = (roi[focus-1][0],x,y,s)
        #print("NEWDATA",new_data)
        #print("ROI PRE",roi)
        roi[focus-1] = new_data
        #print("ROI POST",roi)
        roi_img = drawROI(image,roi,focus)

        cv2.imshow('ROI_Config', roi_img)
        wait = 0 if step_through else 1000
        key = cv2.waitKey(wait) & 0xFF
        if key == ord('p'):
            step_through = not step_through
        elif key == ord('s'):
            saveROI(roi,out_csv)
            break
        elif key == ord('q'):
            print("Quitting without saving!")
            break

if __name__ == '__main__':
    roi_csv = 'tb_roi.csv'
    out_csv = roi_csv
    img = 'data/q.png'
    main(img,roi_csv,out_csv)
