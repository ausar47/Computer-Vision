import cv2
import numpy as np
import os

Window_Size = 3                     # Window size of Harris Corner Detect
Harris_Corner_Constant = 0.04       # Harris corner constant
Thresh = 10000                      # Threshold


def gcd(a, b):
    if a < b:
        a, b = b, a
    while b != 0:
        temp = a % b
        a = b
        b = temp
    return a


def HarrisCornerDetection(img, color_img, window_size, k, thresh):
    # 求梯度
    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = img.shape[0]
    width = img.shape[1]
    minEigenvalueImg = np.zeros((height, width, 3), np.uint8)
    minEigenvalueImg[:] = [255, 255, 255]
    maxEigenvalueImg = np.zeros((height, width, 3), np.uint8)
    maxEigenvalueImg[:] = [255,255,255]
    offset = int(window_size/2)
    cornerList = np.zeros((width, height), dtype=int)

    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            R = det - k * (trace**2)
            cornerList[x][y] = int(R)
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            if cornerList[x][y] > thresh:
                if cornerList[x-offset:x+offset+1,y-offset:y+offset+1].max() == cornerList[x][y]:
                    color_img.itemset((y, x, 0), 0)
                    color_img.itemset((y, x, 1), 0)
                    color_img.itemset((y, x, 2), 255)
                    minEigenvalueImg.itemset((y, x, 0), 0)
                    minEigenvalueImg.itemset((y, x, 1), 0)
                    minEigenvalueImg.itemset((y, x, 2), 255)
            elif cornerList[x][y]<0:
                maxEigenvalueImg.itemset((y, x, 0), 0)
                maxEigenvalueImg.itemset((y, x, 1), 0)
                maxEigenvalueImg.itemset((y, x, 2), 255)
    return color_img, minEigenvalueImg, maxEigenvalueImg, cornerList


if __name__ == '__main__':
    if not os.path.isdir("./Output/"):
        os.makedirs("./Output/")
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        cv2.imshow("Camera Capture", frame)
        if cv2.waitKey(100) & 0xff == ord(' '):
            imgname = "frame%d.jpg" % count

            path = os.path.join("./Output/", imgname)
            cv2.imwrite(path, frame)
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            finalImg, minEigenvalueImg, maxEigenvalueImg, cornerList = HarrisCornerDetection(img, frame, int(Window_Size), float(Harris_Corner_Constant), int(Thresh))
            cv2.imwrite(os.path.join("./Output/", "result%d.jpg" % count), finalImg)
            cv2.imwrite(os.path.join("./Output/", "minEigenvalue%d.jpg" % count), minEigenvalueImg)
            cv2.imwrite(os.path.join("./Output/", "maxEigenvalue%d.jpg" % count), maxEigenvalueImg)
            cv2.imshow("result", finalImg)
            cv2.imshow("minEigenvalue", minEigenvalueImg)
            cv2.imshow("maxEigenvalue", maxEigenvalueImg)
            count += 1
            cv2.waitKey(0)
