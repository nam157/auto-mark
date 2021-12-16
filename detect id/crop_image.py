
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import cv2
from math import ceil

class crop_image:
    def get_x_ver1(self,s):
        s = cv2.boundingRect(s)
        return s[0] * s[1]
    def pre_processing_input(self,img):
        img = cv2.resize(img,(1056,1500))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
        img_canny = cv2.Canny(blurred, 100, 200)
        cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        mask = np.zeros(img.shape[:2], dtype="uint8")
        return cnts,img 
    def crop_image_sbd(self,img):
        cnts,gray_img = self.pre_processing_input(img)
        info_blocks = []
        x_old, y_old, w_old, h_old = 0, 0, 0, 0
        if len(cnts) > 0:
          cnts = sorted(cnts, key=self.get_x_ver1)
          for i, c in enumerate(cnts):
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 2000 and h < w and h == 34:
              check_xy_min = x * y - x_old * y_old
              check_xy_max = (x + w) * (y + h) - (x_old + w_old) * (y_old + h_old)
              if check_xy_min > 70000 and check_xy_max > 110000:
                print(check_xy_min,check_xy_max)
                info_blocks.append((gray_img[y:y + h, x:x + w],[x,y,w,h]))
                x_old,y_old,w_old,h_old= x,y,w,h
          
        return info_blocks
    def crop_image_md(self,img):
        cnts,gray_img = self.pre_processing_input(img)
        info_blocks = []
        x_old, y_old, w_old, h_old = 0, 0, 0, 0
        if len(cnts) > 0:
          cnts = sorted(cnts, key=self.get_x_ver1)
          for i, c in enumerate(cnts):
            x, y, w, h = cv2.boundingRect(c)
            if w * h > 2000 and h < w and h == 34:
              check_xy_min = x * y - x_old * y_old
              check_xy_max = (x + w) * (y + h) - (x_old + w_old) * (y_old + h_old)
              if check_xy_min > 90000 and check_xy_max > 131000:
                print(check_xy_min,check_xy_max)
                info_blocks.append((gray_img[y:y + h, x:x + w],[x,y,w,h]))
                x_old,y_old,w_old,h_old= x,y,w,h

        return info_blocks
    def split_blocks_sbd(self,a):
        list_answers = []
        for ans_block in a:
          ans_block_img = np.array(ans_block[0])
          offset1 = ceil(ans_block_img.shape[1] / 6)
          h = ans_block_img.shape[0]
          for i in range(6):
            box_img = np.array(ans_block_img[4:h-4,i * offset1+2:(i + 1) * offset1-2])
            list_answers.append(box_img)
        return list_answers
    def split_blocks_md(self,a):
        list_answers = []
        for ans_block in a:
            ans_block_img = np.array(ans_block[0])
            offset1 = ceil(ans_block_img.shape[1] / 3)
            h = ans_block_img.shape[0]
            for i in range(3):
                box_img = np.array(ans_block_img[4:h-4,i * offset1+2:(i + 1) * offset1-2])
                list_answers.append(box_img)
        return list_answers