import numpy as np
import apriltag
import cv2
from matplotlib import pyplot as plt

def nothing(x):
    pass

def main():
    running = True
    step_through = True

    cv2.namedWindow('Homography', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('h11','Homography',500,1000,nothing)
    cv2.createTrackbar('h12','Homography',0,1000,nothing)
    cv2.createTrackbar('h13','Homography',0,1000,nothing)
    cv2.createTrackbar('h21','Homography',0,1000,nothing)
    cv2.createTrackbar('h22','Homography',500,1000,nothing)
    cv2.createTrackbar('h23','Homography',0,1000,nothing)
    cv2.createTrackbar('h31','Homography',0,1000,nothing)
    cv2.createTrackbar('h32','Homography',0,1000,nothing)
    cv2.createTrackbar('h33','Homography',500,1000,nothing)

    img = cv2.imread('TB_emptyreference.jpg',1)
    w = img.shape[1]
    h = img.shape[0]
    print("W",w,"\tH",h)

    while running:
        h11 = cv2.getTrackbarPos('h11','Homography')
        h12 = cv2.getTrackbarPos('h12','Homography')
        h13 = cv2.getTrackbarPos('h13','Homography')
        h21 = cv2.getTrackbarPos('h21','Homography')
        h22 = cv2.getTrackbarPos('h22','Homography')
        h23 = cv2.getTrackbarPos('h23','Homography')
        h31 = cv2.getTrackbarPos('h31','Homography')
        h32 = cv2.getTrackbarPos('h32','Homography')
        h33 = cv2.getTrackbarPos('h33','Homography')

        # construct homography matrix
        scale = 1000
        H = [[h11,h12,h13],[h21,h22,h23],[h31,h32,h33]]
        H = np.array(H).astype(np.float32)
        H = H / scale
        H[0][2] = H[0][2] * scale
        H[1][2] = H[1][2] * scale
        H[2][0] = H[2][0] / scale
        H[2][1] = H[2][1] / scale
        print("H:\n",H)

        warped = cv2.warpPerspective(img,H,(w,h))
        stack = np.hstack((img, warped))

        cv2.imshow('Homography',stack)

        wait = 0 if step_through else 500
        key = cv2.waitKey(wait) & 0xFF
        # Toggle frame step mode (P key)
        if key == ord('p'):
            step_through = not step_through
        # Quit program (Q pressed)
        elif key == ord('q'):
            running = False
            break

    print("Quitting.")


if __name__ == '__main__':
    main()
