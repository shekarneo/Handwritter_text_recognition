import numpy as np
from PIL import Image
import cv2
import re


def roi_image(imagei):
    image = cv2.imread(imagei)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 5,5 0
    # thresh = cv2.threshold(blur,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 50)  # 115,50

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    ROI_number = 1
    im_list = ['Sign: ', 'Tick: ', 'DOB: ', 'Name: ']
    val_list = []
    res_img = []
    for c in cnts:
        area = cv2.contourArea(c, True)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        if len(approx) == 4 and (area > 1000) and (area < 800000):
            ROI = image[y + 10:y + h - 15, x + 15:x + w - 15]
            ROI1 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            #cv2.imwrite(im_path + '/ROI_{}.png'.format(ROI_number), ROI1)
            # im_list.append(ROI_number)
            res_img.append(ROI1)
            # img = image[y+10:y+h-10, x+10:x+w-10]
            img1 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('opening', img1)
            pi = np.sum(img1 < 150)
            if pi >= 10:
                # print("ROI_{} is non-empty".format(ROI_number))
                val_list.append("valid")
            else:
                # print("ROI_{} is empty or invalid".format(ROI_number))
                val_list.append("invalid")
            ROI_number += 1
    #if res_img > 1:
     #   res_3 = res_img[-1]
    #else:
    res_3 = res_img[-1]
    verification = list(zip(im_list, val_list))
    #print(verification)
    return(res_3,verification)
