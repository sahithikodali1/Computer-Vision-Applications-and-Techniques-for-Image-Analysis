#!/usr/bin/env python
# coding: utf-8

# <h2><center>ECE661 COMPUTER VISION</center></h2>
# <h3><center>Homework - 5 </center></h3>
# <h3><center>Sahithi Kodali - 34789866</center></h3>
# <h3><center>kodali1@purdue.edu</center></h3>

# In[1]:


# Import libraries needed

import numpy as np
import matplotlib.pyplot as plt
import skimage.io as sk
import cv2
import math


# In[65]:


#Image files list
images_all = ["0.jpg", "1.jpg", "2.jpg", "3.jpg", "4.jpg"]
# own_images_all = ["01.jpg", "02.jpg", "03.jpg", "04.jpg", "05.jpg"]


# In[27]:


#convert image into grey scale of channel 1
def normalize_img2(image):
    if len (image.shape) == 3:
        norm_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        norm_img = image
    return norm_img


# In[28]:


#Compute the distance metrics SSD and NCC to compute distance required to match corresponsing interest points in the image pair

def distance_metric(w1,w2, metric = 'SSD'):
    
    #compute SSD and NCC accoording to the formula in the theory concepts
    if metric == 'SSD':
        distance = np.sum( (w1 - w2) ** 2)
    elif metric == 'NCC':
        mean1 = w1.mean()
        mean2 = w2.mean()
        
        ncc_num = np.sum ((w1 - mean1) * (w2 - mean2))
        ncc_den = np.sqrt (np.sum((w1 - mean1) ** 2) * np.sum((w2 - mean2) ** 2))
        
        distance = 1 - (ncc_num/ncc_denom)
    
    return distance


# In[29]:


#compute the SIFT matching using OpenCV techniques

def SIFT(image1, image2):
    
    img1_read = cv2.imread(image1)
    img2_read = cv2.imread(image2)
    
    img1 = normalize_img2(img1_read)
    img2 = normalize_img2(img2_read)
    
    #create SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    
    #compute the keypoints and descriptors for the respective images
    kps1, desc1 = sift.detectAndCompute(img1, None)
    kps2, desc2 = sift.detectAndCompute(img2, None)
    
    #finding correspondences
    images_concat = np.concatenate((img1_read, img2_read), axis = 1)

    domain_coord = np.array([[0,0]])
    range_coord = np.array([[0,0]])
    
    #get the best match fro each point in image1
    for i in range(desc1.shape[0]):
        dist_tmp = []
        kp1 = kps1[i].pt
        
        for j in range(desc2.shape[0]):
            kp2 = kps2[j].pt
            dist = distance_metric(desc1[i], desc2[j])
            dist_tmp.append(dist)
            
        best_kp2_coord = list(kps2[np.argsort(dist_tmp)[0]].pt)
        best_kp2_coord = list(map(int, best_kp2_coord))
        
        if np.min(dist_tmp) < 10000:
            domain_coord = np.append(domain_coord, [list(map(int, kp1))], axis = 0)
            range_coord = np.append(range_coord, [best_kp2_coord], axis = 0)
        
            final_kp1 = tuple(list(map(int, kp1)))
            final_kp2 = (best_kp2_coord[0] + img1_read.shape[1], best_kp2_coord[1])
            
            #draw the matching correspondences
            cv2.circle (images_concat, final_kp1 , 2, (0,0,255) , 1)
            cv2.circle (images_concat, final_kp2 , 2, (0,0,255) , 1)
            cv2.line (images_concat, final_kp1 , final_kp2, (0,255,0) , 1)
    
    #remove the first coordinates that are initialized as (0,0)
    domain_coord = np.delete(domain_coord, 0, axis = 0)
    range_coord = np.delete(range_coord, 0, axis = 0)

    
    return images_concat, domain_coord, range_coord


# In[30]:


# Homography matrix for overdetermined system using Linear Least-Squares Minimization (Inhomogeneous equations)

def homo_matrix_LS(X,X_prime):
    
    #give the matrix A shape
    n = X.shape[0]
    A = np.zeros((2*n,8))
    
    #fill the matrix A
    for i in range(n):
        A[2*i,0] = X[i,0]
        A[2*i,1] = X[i,1]
        A[2*i,2] = 1
        A[2*i,6] = -1*X[i,0]*X_prime[i,0]
        A[2*i,7] = -1*X[i,1]*X_prime[i,0]
        A[2*i+1,3] = X[i,0]
        A[2*i+1,4] = X[i,1]
        A[2*i+1,5] = 1
        A[2*i+1,6] = -1*X[i,0]*X_prime[i,1]
        A[2*i+1,7] = -1*X[i,1]*X_prime[i,1]


    B = X_prime.copy()
    B = B.reshape((-1,1))
    
    #compute the H = A_pseudoInverse * b, for each correspondences and obtain the Homography matrix
    A_pseudoInverse = np.matmul(np.linalg.inv(np.dot(A.T,A)), A.T)
    K = np.dot(A_pseudoInverse, B)

    H = np.concatenate((K,np.array([[1]])),axis = 0)
    H = H.reshape((3,3))
    
    return H


# In[110]:


# Ransac algorithm to return the Homography using the best inlier set obtained

def ransac(domain_coord, range_coord):
    
    # Initializations (refer to theory)
    delta = 2
    n = 6
    eps = 0.4
    prob = 0.99
    
    # Total number of correspondences
    n_total = domain_coord.shape[0]
    
    # Calculate number of trails
    N = int((math.log(1 - prob)/math.log(1 - (1 - eps) ** n)))
    
    # Calculate Minimum size of the inlier set
    M = int(((1 - eps) * n_total))
    
    best_inlier_indxs = []
    inlier_percent = 0
    indices_range = list(range(n_total))
    
    #Check for the maximum inlier set > M among the trails
    for trail in range(N):
        rand_corr_indices = np.random.choice(indices_range,n)
        
        rand_domain_coord = domain_coord[rand_corr_indices]
        rand_range_coord = range_coord[rand_corr_indices]
        
        H_initial = homo_matrix_LS(rand_domain_coord, rand_range_coord)
        
        inliers_tmp_indices = find_inliers(H_initial, domain_coord, range_coord, delta)
        num_inliers = inliers_tmp_indices.shape[0]

        if num_inliers >= M and num_inliers/n_total > inlier_percent:
            inlier_percent = num_inliers/n_total
            best_inlier_indxs = inliers_tmp_indices
    
#     print('Best inliers are')
#     print(best_inlier_indxs)
    
    # compute the final homography using the best inlier set obtained
    H_final = homo_matrix_LS(domain_coord[best_inlier_indxs], range_coord[best_inlier_indxs])
#     print(H_final)
    
    return H_final, best_inlier_indxs
    


# In[111]:


# Function to find the inlier set by comparing the difference of original and transformed coordinates to delta (distance threshold)

def find_inliers(H_initial, domain_coord, range_coord, delta):
    
    domain_coord_HC = np.insert(domain_coord, 2, 1, axis = 1)
    range_coord_HC = np.insert(range_coord, 2, 1, axis = 1)
    
    Y = np.matmul(H_initial, domain_coord_HC.T).T
    Y = Y.T/Y.T[2,:]
    
    #calculate error and use it to get the distance
    error = (np.abs(Y.T - range_coord_HC))**2
    dist_squared = np.sum(error, axis = 1)
    
    #obtian the best inlier coordinates indexes
    inlier_indices = np.where(dist_squared <= delta ** 2)[0]
    
    return inlier_indices


# In[112]:


# Plot the inliers and outliers in a pair of images

def plot_inliers_outliers(image1, image2, domain_coord, range_coord, inlier_idxs):
    
    #read images
    img1_read = cv2.imread(image1)
    img2_read = cv2.imread(image2)
    
    #create image frame
    images_concat_inliers = np.concatenate((img1_read, img2_read), axis = 1)
    images_concat_outliers = np.concatenate((img1_read, img2_read), axis = 1)
    
    #plot for the inliers and outliers saparately
    for idx in range(domain_coord.shape[0]):
        if idx in inlier_idxs:
            pt1 = tuple(domain_coord[idx])
            pt2 = range_coord[idx]
            pt2[0] = pt2[0] + img1_read.shape[1]
            pt2 = tuple(pt2)
            
            cv2.circle (images_concat_inliers, pt1 , 2, (0,255,0) , 1)
            cv2.circle (images_concat_inliers, pt2 , 2, (0,255,0) , 1)
            cv2.line (images_concat_inliers, pt1 , pt2, (0,255,0) , 1)
            
        else:
            pt1 = tuple(domain_coord[idx])
            pt2 = range_coord[idx]
            pt2[0] = pt2[0] + img1_read.shape[1]
            pt2 = tuple(pt2)
            
            cv2.circle (images_concat_outliers, pt1 , 2, (255,255,255) , 1)
            cv2.circle (images_concat_outliers, pt2 , 2, (255,255,255) , 1)
            cv2.line (images_concat_outliers, pt1 , pt2, (255,255,255) , 1)
    
    return images_concat_inliers, images_concat_outliers


# In[113]:


# Levenberg-Marquardt (LM) algorithm initialization using the LM threshold

def LM_homo(H, best_indices, domain_coord, range_coord):
    H_initial = np.ravel(H)
    
    H_tmp,cost = LM_algo(H_initial, jacob_cost_func, args = (domain_coord[best_indices], range_coord[best_indices]), delta_threshold = 0.0001)
    H_refined = H_tmp/H_tmp[-1]
    
    H_refined = H_refined.reshape(3,3)
    
    return H_refined


# In[156]:


# Levenberg-Marquardt (LM) algorithm to give the refined homography and the cost(geometric error)
def LM_algo(H_p0, j_cost_func, args = None, delta_threshold=0.0001):
        
    #check for inliers to call the Jacobian cost function
    def check_inliers_exist(H, args):
        if args == None:
            return j_cost_func(H)
        else:
            return j_cost_func(H, args[0], args[1])
    
    #initialize T (usually between 0.5 and 2)
    T = 0.5 
    
    #intialize intital H (p0) and find the cost and jacobian
    pk = H_p0
    [e, Je] = check_inliers_exist(H_p0, args)
    
    #initialize the damping coefficient
    JtJ = np.matmul(Je.T, Je)
    u0 = T * np.diag(JtJ).max()
    uk = u0
    
    I = np.eye(JtJ.shape[0])
    
    # until the LM threshold is not reached, repeat the steps of computing step, Jacobian, cost function and damping coefficent
    # obtain the new reduced geometric error (cost) and the refined homography
    while(True):
        [e,Je] = check_inliers_exist(pk, args)

        JtJ = np.matmul(Je.T, Je)
        JTE = - np.matmul(Je.T, e)

        delta_p = np.matmul(np.linalg.inv(JtJ + uk * I), JTE)

        pk_1 = pk + delta_p

        [e_next, Je_next] = check_inliers_exist(pk_1, args)
        
        initial_cost = np.matmul(e.T, e)
#         print(initial_cost)
        
        new_cost = np.matmul(e_next.T, e_next)
#         print(new_cost)
        
        ratio_num = initial_cost - new_cost
        ratio_denom = np.matmul(delta_p.T, JTE) + np.matmul(delta_p.T, uk * delta_p)
        LM_ratio = ratio_num / ratio_denom

        uk_1 = uk * max(1/3, 1 - ((2 * LM_ratio) - 1)**3)
        uk = uk_1

        if ratio_num >= 0:
            pk = pk_1
            
            if np.linalg.norm(delta_p) < delta_threshold:
                break

    
    return pk, new_cost


# In[157]:


# Function to compute the partial differences required for Jacobian matrix
def jacob_cost_func(h, domain_coord, range_coord):
    
    X = []
    F = []
    J = []
    
    for i in range(domain_coord.shape[0]):
        
        p1 = domain_coord[i]
        p2 = range_coord[i]
        
        X.extend([p2[0], p2[1]])
        
        f1num = h[0] * p1[0] + h[1] * p1[1] + h[2]
        f2num = h[3] * p1[0] + h[4] * p1[1] + h[5]
        fdenom = h[6] * p1[0] + h[7] * p1[1] + h[8]
        F.extend([f1num/fdenom, f2num/fdenom])
        
        ele1 = -f1num/fdenom**2
        ele2 = -f2num/fdenom**2
        J.extend([p1[0]/fdenom, p1[1]/fdenom, 1/fdenom, 0, 0, 0, p1[0] * ele1, p1[1] * ele2, ele1])
        J.extend([0, 0, 0, p1[0]/fdenom, p1[1]/fdenom, 1/fdenom, p1[0] * ele1, p1[1] * ele2, ele2])

    X = np.array(X)
    F = np.array(F)
    J = np.array(J)
    J = J.reshape(-1,9)
    jacobian = -J
    
    #error/cost is determined
    E = X - F
    cost = E
    E = E.reshape(-1,1)

    return cost, jacobian   


# In[167]:


#Get the homography and the images plots for each pair of images given/not LM
def imgPairs_H(img1, img2, LM = False):
    SIFT_img, domain_coord, range_coord = SIFT(img1, img2)
    H, best_indices = ransac(domain_coord, range_coord)
    inliers_img, outliers_img = plot_inliers_outliers(img1, img2, domain_coord, range_coord, best_indices)
    
    if LM:
        H = LM_homo(H, best_indices, domain_coord, range_coord)
        
    return H, SIFT_img, inliers_img, outliers_img


# In[168]:


# Dtermine the valid pixel coordinates using x, y min and max values 
def get_validPixels(coord, H_coord, x_min, x_max, y_min, y_max):
    
    i = H_coord[0, :] >= x_min
    coord = coord[:, i]
    H_coord = H_coord[:, i]

    i = H_coord[0, :] <= x_max
    coord = coord[:, i]
    H_coord = H_coord[:, i]
    
    i = H_coord[1, :] >= y_min
    coord = coord[:, i]
    H_coord = H_coord[:, i]
    
    i = H_coord[1, :] <= y_max
    coord = coord[:, i]
    H_coord = H_coord[:, i]
    
    return [coord, H_coord]


# In[169]:


#Obtain the weighted pixel values of the valid coordinates
def get_weighted_PixelValues(img, valid_coord):
        
    coord = np.array([valid_coord[0], valid_coord[1]])
    x = int(np.floor(valid_coord[0]))
    y = int(np.floor(valid_coord[1]))
    
    #computer distance using L2 normalization
    d_00 = np.linalg.norm(coord - np.array([x,y]))
    d_01 = np.linalg.norm(coord - np.array([x,y+1]))
    d_10 = np.linalg.norm(coord - np.array([x+1,y]))
    d_11 = np.linalg.norm(coord - np.array([x+1,y+1]))
    
    w_num = img[y][x][:] * d_00 + img[y+1][x][:] * d_01 + img[y][x+1][:] * d_10 + img[y+1][x+1][:] * d_11
    w_den = d_00 + d_01 + d_10 + d_11
    
    w_pixelVal = w_num/w_den
    
    return w_pixelVal
    


# In[170]:


#Function that maps the changed homographies to the whole canvas in the center frame
def mapping_img2canvas(img, canvas, H_pair):
    
    h = img.shape[0]
    w = img.shape[1]
    
    canvas_h = canvas.shape[0]
    canvas_w = canvas.shape[1]
    
    H = np.linalg.inv(H_pair)
    
    #original canvas coordinates
    canvas_coord = np.array([[i,j,1] for i in range(canvas_w) for j in range(canvas_h)])
    canvas_coord = canvas_coord.T
    
    #transformed canvas coordinates
    coord_translated = np.dot(H, canvas_coord)
    coord_translated = coord_translated/coord_translated[2, :]
    
    #check the validity of the coordinates
    [valid_canvas_coord, valid_canvas_translated] = get_validPixels(canvas_coord, coord_translated, 0, w - 2, 0, h - 2)
    
    #obtian the pixel values for the valid points and project onto canvas
    for i in range(valid_canvas_coord.shape[1]):
        
        valid_point_coord = valid_canvas_translated[:, i]
        canvas[valid_canvas_coord[1][i]][valid_canvas_coord[0][i]][:] = get_weighted_PixelValues(img, valid_point_coord)
    
    return canvas
    


# In[171]:


# function to get the homographies of image pairs and transformed to middle frame
# Further the transformed images are stitched together to get a panorama

def panorama(images_all):
    
    num_images = len(images_all)
    H_pairs = []
    
    #get all image pairs homographies
    for i in range(num_images - 1):
        H, SIFT_img, inliers_img, outliers_img = imgPairs_H(images_all[i], images_all[i+1])
        
        #Save the SIFT, iniliers and outliers images for each image pair adn append all homographies
        filename1 = 'D:\Purdue\ECE661_CV\HW5_outputs\SIFT' + str(i) + str(i+1) +'.jpg'
        cv2.imwrite(filename1 ,  SIFT_img)

        filename2 = 'D:\Purdue\ECE661_CV\HW5_outputs\Inliers' + str(i) + str(i+1) +'.jpg'
        cv2.imwrite(filename2,  inliers_img)

        filename3 = 'D:\Purdue\ECE661_CV\HW5_outputs\Outliers' + str(i) + str(i+1) +'.jpg'
        cv2.imwrite(filename3,  outliers_img)
        
        H_pairs.append(H)
    
#     print('appended Hpairs')
#     print(len(H_pairs))
#     print(H_pairs)
    
    #get the center image location
    center_img_loc = int((num_images + 1)/2) - 1
    
    #shifting right half imgs to center frame
    H_ref = np.eye(3, dtype = np.float64)
    
    for i in range(center_img_loc, len(H_pairs)):
        H_ref = np.matmul(H_ref,np.linalg.inv(H_pairs[i]))
        H_pairs[i] = H_ref
    
    #shifting left half imgs to center frame
    H_ref = np.eye(3, dtype = np.float64)

    for i in range(center_img_loc - 1, -1, -1):
        H_ref = np.matmul(H_ref,H_pairs[i])
        H_pairs[i] = H_ref
    
    #insert center_img frame homography
    H_pairs.insert(center_img_loc, np.eye(3, dtype = np.float64))
#     print('centered Hpairs')
#     print(H_pairs)
    
    #Translating the center image location to middle frame of panorama and translate the homographies of image pairs
    tx = 0
    
    for i in range(center_img_loc):
        each_img = cv2.imread(images_all[i])
        tx = tx + each_img.shape[1]
        
    H_translated = np.array([1,0,tx,0,1,0,0,0,1], dtype = float)
    H_translated = H_translated.reshape(3,3)
    
    for i in range(len(H_pairs)):
        H_pairs[i] = np.matmul(H_translated, H_pairs[i])
    
#     print('Translated Hpairs')
#     print(H_pairs)
    
    #stitch the images using their new translated homographies
    h = 0
    w = 0
    all_imgs_stitching = []
    
    for img in images_all:
        each_img = cv2.imread(img)
        h = max(h, each_img.shape[0])
        w = w + each_img.shape[1]
        all_imgs_stitching.append(each_img)
    
    panorama_canvas = np.zeros((h,w,3), np.uint8)
    
    for img_idx, each_img in enumerate(all_imgs_stitching):
        panorama_mapped = mapping_img2canvas(each_img, panorama_canvas, H_pairs[img_idx])
    
    return panorama_mapped


# In[174]:


panorama_result = panorama(images_all)


# In[175]:


#Save the panorama images with LM
filename = 'D:\Purdue\ECE661_CV\HW5_outputs' + '\panorama_withLM' + 'given_imgs' +'.jpg'
cv2.imwrite(filename,  panorama_result)


# In[176]:


# #Save the panorama images without LM
# filename = 'D:\Purdue\ECE661_CV\HW5_outputs' + '\panorama_withoutLM' + 'given_imgs' +'.jpg'
# cv2.imwrite(filename,  panorama_result)


# ### Task -2

# In[166]:


own_images_all = ["001.jpg", "002.jpg", "003.jpg", "004.jpg", "005.jpg"]
panorama_result_own = panorama(own_images_all)


# In[109]:


#Save the panorama image of own images with LM
filename = 'D:\Purdue\ECE661_CV\HW5_outputs' + '\panorama_withLM' + 'own_imgs' +'.jpg'
cv2.imwrite(filename,  panorama_result_own)


# In[172]:


# #Save the panorama image of own images without LM
# filename = 'D:\Purdue\ECE661_CV\HW5_outputs' + '\panorama_withoutLM' + 'own_imgs' +'.jpg'
# cv2.imwrite(filename,  panorama_result_own)

