#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


import imutils
import numpy as np
import cv2
from math import ceil
from collections import defaultdict
import matplotlib.pyplot as plt


# In[6]:


class answer:
    def get_x_ver1(self,s):
        s = cv2.boundingRect(s)
        return s[0] * s[1]

    def get_x(self,s):
          return s[1][0]
    def pre_processing_img(self,img):
       # convert image from BGR to GRAY to apply canny edge detection algorithm
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # remove noise by blur image
        blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)

        # apply canny edge detection algorithm
        img_canny = cv2.Canny(blurred, 100, 200)

        # find contours
        cnts = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return cnts,gray_img
    def crop_image(self,img):
        cnts,gray_img = self.pre_processing_img(img)

        ans_blocks = []
        x_old, y_old, w_old, h_old = 0, 0, 0, 0

        # ensure that at least one contour was found
        if len(cnts) > 0:
            # sort the contours according to their size in descending order
            cnts = sorted(cnts, key=self.get_x_ver1)

            # loop over the sorted contours
            for i, c in enumerate(cnts):
                x_curr, y_curr, w_curr, h_curr = cv2.boundingRect(c)

                if w_curr * h_curr > 100000 and w_curr < h_curr:
                    # check overlap contours
                    check_xy_min = x_curr * y_curr - x_old * y_old
                    check_xy_max = (x_curr + w_curr) * (y_curr + h_curr) - (x_old + w_old) * (y_old + h_old)

                    # if list answer box is empty
                    if len(ans_blocks) == 0:
                        ans_blocks.append(
                            (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr],[x_curr,y_curr,w_curr,h_curr]))
                        # update coordinates (x, y) and (height, width) of added contours
                        x_old,y_old,w_old,h_old = x_curr,y_curr,w_curr,h_curr

                    elif check_xy_min > 20000 and check_xy_max > 20000:
                        ans_blocks.append(
                            (gray_img[y_curr:y_curr + h_curr, x_curr:x_curr + w_curr],[x_curr,y_curr,w_curr,h_curr]))
                        # update coordinates (x, y) and (height, width) of added contours
                        x_old,y_old,w_old,h_old = x_curr,y_curr,w_curr,h_curr

            # sort ans_blocks according to x coordinate
            sorted_ans_blocks = sorted(ans_blocks, key=self.get_x)
            return sorted_ans_blocks
    def divide_ans_blocks(self,ans_blocks):
        """
          Mỗi blocks đáp án có 6 ô nhỏ mỗi ô nhỏ sẽ có 5 câu 
          Do 4 blocks có độ dài bằng nhau sẽ chia ra 6 ô nhỏ 

          """
        list_answers = []
        for ans_block in ans_blocks:
            ans_block_img = np.array(ans_block[0])
            offset1 = ceil(ans_block_img.shape[0] / 6)
            for i in range(6):
                    box_img = np.array(ans_block_img[i * offset1:(i + 1) * offset1, :])
                    height_box = box_img.shape[0]
                    box_img = box_img[14:height_box-14, :]
                    offset2 = ceil(box_img.shape[0] / 5)
                    for j in range(5):
                          list_answers.append(box_img[j * offset2:(j + 1) * offset2, :])
        return list_answers
    def list_ans(self,list_answers):
        """
            - có 120 câu thì sẽ có 4 đáp án thì tổng sẽ có 120 * 4 = 480
            - Để crop mỗi lựa chọn thì lặp qua 4 rồi chọn khoảng crop ra (start lấy tự vị trí đầu cho mỗi đáp án bỏ qua số thứ tự và offset là ví trị kq dừng)

          """
        list_choices = []
        for answer_img in list_answers:
            start = 40
            offset = 40
            for i in range(4):
                    bubble_choice = answer_img[:,start + i * offset:start + (i + 1) * offset]
                    bubble_choice = cv2.threshold(bubble_choice, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    bubble_choice = cv2.resize(bubble_choice, (28, 28), cv2.INTER_AREA)
                    bubble_choice = bubble_choice.reshape((28, 28, 1))
                    list_choices.append(bubble_choice)
        return list_choices
    def map_answer(self,idx):
        if idx % 4 == 0:
            answer_circle = "A"
        elif idx % 4 == 1:
            answer_circle = "B"
        elif idx % 4 == 2:
            answer_circle = "C"
        else:
            answer_circle = "D"
        return answer_circle
    def get_answers(self,list_answers,model):
        results = defaultdict(list)
        list_answers = np.array(list_answers)
        scores = model.predict_on_batch(list_answers / 255.0)
        for idx, score in enumerate(scores):
            question = idx // 4
            if score[1] > 0.9:
                chosed_answer = self.map_answer(idx)
                results[question + 1].append(chosed_answer)

        return results


# In[9]:


def main():
    img = cv2.imread('a1.jpg')
    img = cv2.resize(img,(1100,1500))
    model  = tf.keras.models.load_model('weight11.h5')
    crop =  answer()
    ans_blocks = crop.crop_image(img)
    list_answer = crop.divide_ans_blocks(ans_blocks)
    list_answer = crop.list_ans(list_answer)
    result = crop.get_answers(list_answer,model)
    return result


# In[10]:


main()


# In[ ]:




