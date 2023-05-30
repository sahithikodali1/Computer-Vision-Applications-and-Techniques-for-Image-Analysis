#!/usr/bin/env python
# coding: utf-8

# <h2><center>ECE661 COMPUTER VISION</center></h2>
# <h3><center>Homework - 4</center></h3>
# <h3><center>Sahithi Kodali - 34789866</center></h3>
# <h3><center>kodali1@purdue.edu</center></h3>

# ## Task 1
# ### 1.1 Harris Corner Detector

# In[1]:


# Import libraries needed

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as sk
import cv2
import PIL
import math


# In[2]:


# Read the images from uplaoded files/directly by giving path in the computer

books1_img = sk.imread('books_1.jpeg')
books2_img = sk.imread('books_2.jpeg')
fountain1_img = sk.imread('fountain_1.jpg')
fountain2_img = sk.imread('fountain_2.jpg')


# In[3]:


#find harr wavelets filters which are based on teh smallest even number above 4 * sigma based on teh scale value
def haar_wavelets (sigma):
    
    #determine kernel size
    k_size = int(math.ceil(sigma * 4))
    k_size = k_size + k_size%2
    
    #initiate kernels
    haar_x = np.ones((k_size, k_size))
    haar_y = np.ones((k_size, k_size))

    #modify as haar wavelets
    haar_x[:, : int(k_size//2)] = -1
    haar_y[int(k_size//2) :, :] = -1
    
    return haar_x, haar_y


# In[4]:


#normalize RGB image by converting to grey scale image i.e channel of 1 and dividing with 255.

def normalize_img(image):
    if len (image.shape) == 3:
        norm_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        norm_img = image
    return norm_img/255


# In[5]:


def harris(image, sigma):
    
    #normalize image
    img = normalize_img(image)
    
    #obtain haar wavelets
    hx, hy = haar_wavelets(sigma)
    
    #define width and height of the image
    h, w = img.shape 
    
    #Find the gradients
    Ix = cv2.filter2D(img, -1, hx)
    Iy = cv2.filter2D(img, -1, hy)
    
    #determine the window which is (5*sigma, 5*sigma) size kernel to find the correlation at each pixel
    window_size = int(math.ceil(5 * sigma))
    window_size = window_size + window_size%2
    sum_window = np.ones((window_size, window_size))
    
    #compute the matrix C by finding the respective matrix values
    sum_dx2 = cv2.filter2D( Ix * Ix, -1, sum_window)
    sum_dy2 = cv2.filter2D( Iy * Iy, -1, sum_window)
    sum_dxy = cv2.filter2D( Ix * Iy, -1, sum_window)
    
    det_c = (sum_dx2 * sum_dy2) - (sum_dxy * sum_dxy)    
    trace_c = sum_dx2 + sum_dy2 

    #find the ratio k value
    k = det_c / (trace_c ** 2 + 0.000001)
    k = np.sum(k)/ (h*w)
    
    #compute response and give a threshold fo first 500 values
    response = det_c - (k * trace_c ** 2)
    response_thresholded = np.sort(response.flatten())[-500]
    
    #perform non-max supression on window aroudn each pixel
    nms_corner_points = []
    nms_window = int(window_size//2)
    print(nms_window)
    
    for x in range (nms_window, w - nms_window, 1):
        for y in range (nms_window, h - nms_window, 1):
            
            window = response[ y - nms_window : y + nms_window, x - nms_window : x + nms_window]
            
            if response[y,x] == window.max() and window.max() >= response_thresholded:
                nms_corner_points.append([x,y])
                
                #mark circles using random colors at each corner point determined
                rand_color = (list(np.random.choice(range(256), size=3)))  
                color =[int(rand_color[0]), int(rand_color[1]), int(rand_color[2])] 
                opt_img = cv2.circle (image, (x,y), 4, color , 3)

    
    return nms_corner_points, opt_img, sigma


# In[6]:


#initiate sigma values
sigma_vals = [0.3, 0.6, 1.2, 2.4]

#make copy of images to not override the original images required for further tasks
books1_copy = books1_img.copy()
books2_copy = books2_img.copy()
fountain1_copy = fountain1_img.copy()
fountain2_copy = fountain2_img.copy()


# In[7]:


#Compute the distance metrics SSD and NCC to compute distance required to match corresponsing interest points in the image pair

def distance_metric(img1, points1, img2, points2, metric = '', k = 21):
#     k_win = int(k_win//2)
#     print(k_win)

    #give the 21 x 21 window for neighbourhood around each pixel
    w1 = img1[ points1[1] : points1[1] + k, points1[0] : points1[0] + k ].flatten()
    w2 = img2[ points2[1] : points2[1] + k, points2[0] : points2[0] + k ].flatten()
    
    #compute SSD and NCC accoording to the formula in the theory concepts
    if metric == 'SSD':
        distance = np.sum( (w1 - w2) ** 2)
    elif metric == 'NCC':
        mean1 = w1.mean()
        mean2 = w2.mean()
        
        ncc_num = np.sum ((w1 - mean1) * (w2 - mean2))
        ncc_den = np.sqrt (np.sum((w1 - mean1) ** 2) * np.sum((w2 - mean2) ** 2))
        
        distance = ncc_num/ (ncc_den + 0.000001)
    
    return distance


# In[8]:


#The corresponding interest point pairs are matched by computing the distance accoording to chosen metric
#The points with shortest distacne are chosen as the pair

def harris_correspondences(image1, points1, image2, points2, metric, pad):
    
    #normalize images
    img1 = normalize_img(image1)
    img2 = normalize_img(image2)
    
    #pad the images
    img1 = cv2.copyMakeBorder(img1, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value = 0)
    img2 = cv2.copyMakeBorder(img2, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value = 0)
    
    #compute distance between every point in an image with every oterh point
    #sort distances acocordingly and return the points with best matching pairs
    #a circle for the interest points deduced and the corresponding matching lines are drawn
    if len(points1) <= len (points2):
        w = image1.shape[1]
        images_concat = np.concatenate((image1, image2), axis = 1)
        
        for p1 in points1:
            dist_tmp = []
            for p2 in points2:
                distance = distance_metric(img1, p1, img2, p2, metric = metric, k = int(2*pad + 1))
                dist_tmp.append(distance)
            
            dist_sorted = points2[np.argsort(dist_tmp)[0]]
            
            if np.min(dist_tmp) < 30:
                pp1 = tuple(p1)
                pp2 = (dist_sorted[0] + w, dist_sorted[1])
                
                cv2.circle (images_concat, pp1 , 4, (0,0,255) , 3)
                cv2.circle (images_concat, pp2 , 4, (0,0,255) , 3)
                cv2.line (images_concat, pp1 , pp2, (0,255,0) , 3)

        
    else:
        w = image2.shape[1]
        images_concat = np.concatenate((image1, image2), axis = 1)
        
        for p2 in points2:
            dist_tmp = []
            for p1 in points1:
                distance = distance_metric(img1, p1, img2, p2, metric = metric, k = int(2*pad + 1))
                dist_tmp.append(distance)
        
            dist_sorted = points1[np.argsort(dist_tmp)[0]]

            if np.min(dist_tmp) < 30:
                pp1 = (dist_sorted[0], dist_sorted[1])
                pp2 = (p2[0] + w, p2[1])

                cv2.circle (images_concat, pp1 , 4, (0,0,255) , 3)
                cv2.circle (images_concat, pp2 , 4, (0,0,255) , 3)
                cv2.line (images_concat, pp1 , pp2, (0,255,0) , 3)
    
    return images_concat


# In[9]:


points_b1, opt_img_b1, sigma = harris(books1_copy, 0.3)
points_b2, opt_img_b2, sigma = harris(books2_copy, 0.3)

filename1 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'b1' +'.jpg'
cv2.imwrite(filename1, opt_img_b1)

filename2 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'b2' +'.jpg'
cv2.imwrite(filename2, opt_img_b2)

SSDbooks_12 = harris_correspondences(books1_img, points_b1, books2_img, points_b2, 'SSD', 10 )
filename3 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'SSD books' +'.jpg'
cv2.imwrite(filename3, SSDbooks_12)

NCCbooks_12 = harris_correspondences(books1_img, points_b1, books2_img, points_b2, 'NCC', 10 )
filename4 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'NCC books' +'.jpg'
cv2.imwrite(filename4, NCCbooks_12)


# In[10]:


points_b1, opt_img_b1, sigma = harris(books1_copy, 0.6)
points_b2, opt_img_b2, sigma = harris(books2_copy, 0.6)

filename1 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'b1' +'.jpg'
cv2.imwrite(filename1, opt_img_b1)

filename2 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'b2' +'.jpg'
cv2.imwrite(filename2, opt_img_b2)

SSDbooks_12 = harris_correspondences(books1_img, points_b1, books2_img, points_b2, 'SSD', 10 )
filename3 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'SSD books' +'.jpg'
cv2.imwrite(filename3, SSDbooks_12)

NCCbooks_12 = harris_correspondences(books1_img, points_b1, books2_img, points_b2, 'NCC', 10 )
filename4 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'NCC books' +'.jpg'
cv2.imwrite(filename4, NCCbooks_12)


# In[11]:


points_b1, opt_img_b1, sigma = harris(books1_copy, 1.2)
points_b2, opt_img_b2, sigma = harris(books2_copy, 1.2)

filename1 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'b1' +'.jpg'
cv2.imwrite(filename1, opt_img_b1)

filename2 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'b2' +'.jpg'
cv2.imwrite(filename2, opt_img_b2)

SSDbooks_12 = harris_correspondences(books1_img, points_b1, books2_img, points_b2, 'SSD', 10 )
filename3 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'SSD books' +'.jpg'
cv2.imwrite(filename3, SSDbooks_12)

NCCbooks_12 = harris_correspondences(books1_img, points_b1, books2_img, points_b2, 'NCC', 10 )
filename4 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'NCC books' +'.jpg'
cv2.imwrite(filename4, NCCbooks_12)


# In[12]:


points_b1, opt_img_b1, sigma = harris(books1_copy, 2.4)
points_b2, opt_img_b2, sigma = harris(books2_copy, 2.4)

filename1 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'b1' +'.jpg'
cv2.imwrite(filename1, opt_img_b1)

filename2 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'b2' +'.jpg'
cv2.imwrite(filename2, opt_img_b2)

SSDbooks_12 = harris_correspondences(books1_img, points_b1, books2_img, points_b2, 'SSD', 10 )
filename3 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'SSD books' +'.jpg'
cv2.imwrite(filename3, SSDbooks_12)

NCCbooks_12 = harris_correspondences(books1_img, points_b1, books2_img, points_b2, 'NCC', 10 )
filename4 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'NCC books' +'.jpg'
cv2.imwrite(filename4, NCCbooks_12)


# ##### Fountain imgs - Harris

# In[13]:


points_f1, opt_img_f1, sigma = harris(fountain1_copy, 0.3)
points_f2, opt_img_f2, sigma = harris(fountain2_copy, 0.3)

filename1 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'f1' +'.jpg'
cv2.imwrite(filename1, opt_img_f1)

filename2 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'f2' +'.jpg'
cv2.imwrite(filename2, opt_img_f2)

SSDfountain_12 = harris_correspondences(fountain1_img, points_f1, fountain2_img, points_f2, 'SSD', 10 )
filename3 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'SSD fountain' +'.jpg'
cv2.imwrite(filename3, SSDfountain_12)

NCCfountain_12 = harris_correspondences(fountain1_img, points_f1, fountain2_img, points_f2, 'NCC', 10 )
filename4 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'NCC fountain' +'.jpg'
cv2.imwrite(filename4, NCCfountain_12)


# In[14]:


points_f1, opt_img_f1, sigma = harris(fountain1_copy, 0.6)
points_f2, opt_img_f2, sigma = harris(fountain2_copy, 0.6)

filename1 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'f1' +'.jpg'
cv2.imwrite(filename1, opt_img_f1)

filename2 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'f2' +'.jpg'
cv2.imwrite(filename2, opt_img_f2)

SSDfountain_12 = harris_correspondences(fountain1_img, points_f1, fountain2_img, points_f2, 'SSD', 10 )
filename3 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'SSD fountain' +'.jpg'
cv2.imwrite(filename3, SSDfountain_12)

NCCfountain_12 = harris_correspondences(fountain1_img, points_f1, fountain2_img, points_f2, 'NCC', 10 )
filename4 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'NCC fountain' +'.jpg'
cv2.imwrite(filename4, NCCfountain_12)


# In[15]:


points_f1, opt_img_f1, sigma = harris(fountain1_copy, 1.2)
points_f2, opt_img_f2, sigma = harris(fountain2_copy, 1.2)

filename1 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'f1' +'.jpg'
cv2.imwrite(filename1, opt_img_f1)

filename2 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'f2' +'.jpg'
cv2.imwrite(filename2, opt_img_f2)

SSDfountain_12 = harris_correspondences(fountain1_img, points_f1, fountain2_img, points_f2, 'SSD', 10 )
filename3 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'SSD fountain' +'.jpg'
cv2.imwrite(filename3, SSDfountain_12)

NCCfountain_12 = harris_correspondences(fountain1_img, points_f1, fountain2_img, points_f2, 'NCC', 10 )
filename4 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'NCC fountain' +'.jpg'
cv2.imwrite(filename4, NCCfountain_12)


# In[16]:


points_f1, opt_img_f1, sigma = harris(fountain1_copy, 2.4)
points_f2, opt_img_f2, sigma = harris(fountain2_copy, 2.4)

filename1 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'f1' +'.jpg'
cv2.imwrite(filename1, opt_img_f1)

filename2 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'f2' +'.jpg'
cv2.imwrite(filename2, opt_img_f2)

SSDfountain_12 = harris_correspondences(fountain1_img, points_f1, fountain2_img, points_f2, 'SSD', 10 )
filename3 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'SSD fountain' +'.jpg'
cv2.imwrite(filename3, SSDfountain_12)

NCCfountain_12 = harris_correspondences(fountain1_img, points_f1, fountain2_img, points_f2, 'NCC', 10 )
filename4 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'NCC fountain' +'.jpg'
cv2.imwrite(filename4, NCCfountain_12)


# ### 1.2 SIFT algorithm

# In[29]:


#convert image into grey scale of channel 1
def normalize_img2(image):
    if len (image.shape) == 3:
        norm_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        norm_img = image
    return norm_img


# In[30]:


#compute the SIFT matching using OpenCV techniques
def SIFT(image1, image2):
    img1 = normalize_img2(image1)
    img2 = normalize_img2(image2)
    
    #create SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    
    #compute the keypoints and descriptors for the respective images
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
    
    #match the points based on sorted distance
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key = lambda x: x.distance)

    #draw lines for corresponding matches
    img_concat = cv2.drawMatches(image1, keypoints_1, image2, keypoints_2, matches[0:80], image2, flags=2)
    
    plt.imshow(img_concat)
    return img_concat


# In[31]:


sift_books = SIFT(books1_img, books2_img)
filename = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\SIFT' + 'books' +'.jpg'
cv2.imwrite(filename, sift_books)


# In[32]:


sift_fountain = SIFT(fountain1_img, fountain2_img)
filename = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\SIFT' + 'fountain' +'.jpg'
cv2.imwrite(filename, sift_fountain)


# ## Task 2 (Own images)
# ###  2.1 Harris Corner Detector

# In[20]:


nova1_img = cv2.imread('nova1.jpg')
nova2_img = cv2.imread('nova2.jpg')

# night1_img = cv2.imread('night1.jpg')
# night2_img = cv2.imread('night2.jpg')

bottles1_img = cv2.imread('bottles1.jpg')
bottles2_img = cv2.imread('bottles2.jpg')

# apts1_img = cv2.imread('apts1.jpg')
# apts2_img = cv2.imread('apts2.jpg')

#make copy of images to not override the original images required for further tasks
nova1_copy = nova1_img.copy()
nova2_copy = nova2_img.copy()
bottles1_copy = bottles1_img.copy()
bottles2_copy = bottles2_img.copy()


# In[21]:


points_n1, opt_img_n1, sigma = harris(nova1_copy, 0.3)
points_n2, opt_img_n2, sigma = harris(nova2_copy, 0.3)

filename1 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'n1' +'.jpg'
cv2.imwrite(filename1, opt_img_n1)

filename2 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'n2' +'.jpg'
cv2.imwrite(filename2, opt_img_n2)

SSDnova_12 = harris_correspondences(nova1_img, points_n1, nova2_img, points_n2, 'SSD', 10 )
filename3 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'SSD nova' +'.jpg'
cv2.imwrite(filename3, SSDnova_12)

NCCnova_12 = harris_correspondences(nova1_img, points_n1, nova2_img, points_n2, 'NCC', 10 )
filename4 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'NCC nova' +'.jpg'
cv2.imwrite(filename4, NCCnova_12)


# In[22]:


points_n1, opt_img_n1, sigma = harris(nova1_copy, 0.6)
points_n2, opt_img_n2, sigma = harris(nova2_copy, 0.6)

filename1 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'n1' +'.jpg'
cv2.imwrite(filename1, opt_img_n1)

filename2 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'n2' +'.jpg'
cv2.imwrite(filename2, opt_img_n2)

SSDnova_12 = harris_correspondences(nova1_img, points_n1, nova2_img, points_n2, 'SSD', 10 )
filename3 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'SSD nova' +'.jpg'
cv2.imwrite(filename3, SSDnova_12)

NCCnova_12 = harris_correspondences(nova1_img, points_n1, nova2_img, points_n2, 'NCC', 10 )
filename4 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'NCC nova' +'.jpg'
cv2.imwrite(filename4, NCCnova_12)


# In[23]:


points_n1, opt_img_n1, sigma = harris(nova1_copy, 1.2)
points_n2, opt_img_n2, sigma = harris(nova2_copy, 1.2)

filename1 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'n1' +'.jpg'
cv2.imwrite(filename1, opt_img_n1)

filename2 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'n2' +'.jpg'
cv2.imwrite(filename2, opt_img_n2)

SSDnova_12 = harris_correspondences(nova1_img, points_n1, nova2_img, points_n2, 'SSD', 10 )
filename3 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'SSD nova' +'.jpg'
cv2.imwrite(filename3, SSDnova_12)

NCCnova_12 = harris_correspondences(nova1_img, points_n1, nova2_img, points_n2, 'NCC', 10 )
filename4 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'NCC nova' +'.jpg'
cv2.imwrite(filename4, NCCnova_12)


# In[24]:


points_n1, opt_img_n1, sigma = harris(nova1_copy, 2.4)
points_n2, opt_img_n2, sigma = harris(nova2_copy, 2.4)

filename1 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'n1' +'.jpg'
cv2.imwrite(filename1, opt_img_n1)

filename2 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'n2' +'.jpg'
cv2.imwrite(filename2, opt_img_n2)

SSDnova_12 = harris_correspondences(nova1_img, points_n1, nova2_img, points_n2, 'SSD', 10 )
filename3 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'SSD nova' +'.jpg'
cv2.imwrite(filename3, SSDnova_12)

NCCnova_12 = harris_correspondences(nova1_img, points_n1, nova2_img, points_n2, 'NCC', 10 )
filename4 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'NCC nova' +'.jpg'
cv2.imwrite(filename4, NCCnova_12)


# bootles-imgs-harris

# In[25]:


points_bt1, opt_img_bt1, sigma = harris(bottles1_copy, 0.3)
points_bt2, opt_img_bt2, sigma = harris(bottles2_copy, 0.3)

filename1 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'bt1' +'.jpg'
cv2.imwrite(filename1, opt_img_bt1)

filename2 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'bt2' +'.jpg'
cv2.imwrite(filename2, opt_img_bt2)

SSDbottles_12 = harris_correspondences(bottles1_img, points_bt1, bottles2_img, points_bt2, 'SSD', 10 )
filename3 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'SSD bottles' +'.jpg'
cv2.imwrite(filename3, SSDbottles_12)

NCCbottles_12 = harris_correspondences(bottles1_img, points_bt1, bottles2_img, points_bt2, 'NCC', 10 )
filename4 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'NCC bottles' +'.jpg'
cv2.imwrite(filename4, NCCbottles_12)


# In[26]:


points_bt1, opt_img_bt1, sigma = harris(bottles1_copy, 0.6)
points_bt2, opt_img_bt2, sigma = harris(bottles2_copy, 0.6)

filename1 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'bt1' +'.jpg'
cv2.imwrite(filename1, opt_img_bt1)

filename2 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'bt2' +'.jpg'
cv2.imwrite(filename2, opt_img_bt2)

SSDbottles_12 = harris_correspondences(bottles1_img, points_bt1, bottles2_img, points_bt2, 'SSD', 10 )
filename3 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'SSD bottles' +'.jpg'
cv2.imwrite(filename3, SSDbottles_12)

NCCbottles_12 = harris_correspondences(bottles1_img, points_bt1, bottles2_img, points_bt2, 'NCC', 10 )
filename4 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'NCC bottles' +'.jpg'
cv2.imwrite(filename4, NCCbottles_12)


# In[27]:


points_bt1, opt_img_bt1, sigma = harris(bottles1_copy, 1.2)
points_bt2, opt_img_bt2, sigma = harris(bottles2_copy, 1.2)

filename1 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'bt1' +'.jpg'
cv2.imwrite(filename1, opt_img_bt1)

filename2 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'bt2' +'.jpg'
cv2.imwrite(filename2, opt_img_bt2)

SSDbottles_12 = harris_correspondences(bottles1_img, points_bt1, bottles2_img, points_bt2, 'SSD', 10 )
filename3 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'SSD bottles' +'.jpg'
cv2.imwrite(filename3, SSDbottles_12)

NCCbottles_12 = harris_correspondences(bottles1_img, points_bt1, bottles2_img, points_bt2, 'NCC', 10 )
filename4 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'NCC bottles' +'.jpg'
cv2.imwrite(filename4, NCCbottles_12)


# In[28]:


points_bt1, opt_img_bt1, sigma = harris(bottles1_copy, 2.4)
points_bt2, opt_img_bt2, sigma = harris(bottles2_copy, 2.4)

filename1 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'bt1' +'.jpg'
cv2.imwrite(filename1, opt_img_bt1)

filename2 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'bt2' +'.jpg'
cv2.imwrite(filename2, opt_img_bt2)

SSDbottles_12 = harris_correspondences(bottles1_img, points_bt1, bottles2_img, points_bt2, 'SSD', 10 )
filename3 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'SSD bottles' +'.jpg'
cv2.imwrite(filename3, SSDbottles_12)

NCCbottles_12 = harris_correspondences(bottles1_img, points_bt1, bottles2_img, points_bt2, 'NCC', 10 )
filename4 = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\Harris' + str(sigma) + 'NCC bottles' +'.jpg'
cv2.imwrite(filename4, NCCbottles_12)


# ### 2.2 SIFT algorithm

# In[33]:


sift_nova = SIFT(nova1_img, nova2_img)
filename = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\SIFT' + 'nova' +'.jpg'
cv2.imwrite(filename, sift_nova)


# In[34]:


sift_bottles = SIFT(bottles1_img, bottles2_img)
filename = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\SIFT' + 'bottles' +'.jpg'
cv2.imwrite(filename, sift_bottles)


# In[ ]:


# sift_apts = SIFT(apts1_img, apts2_img)
# filename = 'D:\Purdue\ECE661_CV\HW4_outputs' + '\SIFT' + 'apts' +'.jpg'
# cv2.imwrite(filename, sift_apts)

