#!/usr/bin/env python
# coding: utf-8

# <h2><center>ECE661 COMPUTER VISION</center></h2>
# <h3><center>Homework - 3</center></h3>
# <h3><center>Sahithi Kodali - 34789866</center></h3>
# <h3><center>kodali1@purdue.edu</center></h3>

# ## Task 1
# ### Task 1.1 (Point-to-point correspndence method)

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

building_img = sk.imread('building.jpg')
canvas_img = sk.imread('nighthawks.jpg')


# In[3]:


# Create an array of pixel corodinates located at the four corners of the Region of Interest (RoI) in the images 
# Here the GIMP - GUI based tool is used to estimate the pixel ordinates of the images

# The target coordinates are chosen based on the height and width of the images

building_points = np.array ([[240,119], [240,457],[723,450],[717,291]], np.int32)
building_target_points = np.array([[0,0], [0,533],[800,533],[800,0]], np.int32)

canvas_points = np.array ([[15,103], [15,727], [865,678], [862,162]], np.int32)
canvas_target_points = np.array([[0,0], [0,958],[1440,958],[1440,0]], np.int32)


# In[4]:


#To check that the pixel coordinates obtained are as desired, we draw a boundary box using these corodinates 
#Below function takes the image and its pixel coordinates as input and returns the image with boundary box 

def boundarybox_check(img, points):
    img_bb = cv2.polylines(img.copy(), [points.reshape(-1,1,2)], True, (255,0,0), thickness = 5)
    return img_bb


# In[5]:


#checking the validity of boundary box by displaying

building_bb = boundarybox_check(building_img, building_points)
canvas_bb = boundarybox_check(canvas_img, canvas_points)
plt.imshow(building_bb)


# In[6]:


# Now that the coordinates are verified, the Homography matrix [H] in the derived equation AH = B is estimated
# Below function returns the [H] by taking the points of image plane and the world plane on which it is projected 

def homo_matrix(X,X_prime):
    
    #create an empty matrix
    A = np.zeros((8,8))
    
    #iterate throuogh each element and add values in A
    for i in range(4):
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

    #make a copy of target coordinates and reshape to required dimension
    B = X_prime.copy()
    B = B.reshape((-1,1))
    
    #calculate H = inv(A)B
    K = np.dot(np.linalg.inv(A), B)
    
    #complete the homography matrix H
    H = np.concatenate((K,np.array([[1]])),axis = 0)
    H = H.reshape((3,3))
    
    return H


# In[7]:


#Calculate the Homo matrix for the image plane and world plane points

H_building = homo_matrix(building_points, building_target_points)
H_canvas = homo_matrix(canvas_points, canvas_target_points)


# In[8]:


#find the minimum and maximum of the mapped coordinates to define an offset while taking the world plane size

def find_min_max(image, homo_matrix):
    
    #initialize min and max of numbers
    xmin = math.inf
    xmax = -1 * math.inf

    ymin = math.inf
    ymax = -1 * math.inf

    #iterate through the image coordinates to compute mapped coordinates
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            coord = np.array([j,i,1])
            coord = coord.reshape((-1,1))
            
            #calculate mapped coordinates by finding HX
            mapped_coord = np.matmul(homo_matrix, coord)
            mapped_coord = mapped_coord/mapped_coord[2,0]
            
            #determine the max and min values of x and y
            if mapped_coord[0,0]<xmin:
                xmin = mapped_coord[0,0]

            if mapped_coord[0,0]>xmax:
                xmax = mapped_coord[0,0] 

            if mapped_coord[1,0]<ymin:
                ymin = mapped_coord[1,0]

            if mapped_coord[1,0]>ymax:
                ymax = mapped_coord[1,0]

    return xmin,xmax,ymin,ymax


# In[9]:


#Initialize a zero matrix as the world plane based on min and max values of the coordinates

output_img_p2p_building = np.zeros((2400,1400,3), dtype = 'uint8')
output_img_p2p_canvas = np.zeros((2100,2900,3), dtype = 'uint8')


# In[10]:


#finding the min and max of image1
find_min_max(building_img, H_building)


# In[11]:


#finding the min and max of image2
find_min_max(canvas_img, H_canvas)


# In[12]:


# Calculate mapped coordinates from image plane to the world plane

def mapping_func_weighted(image, homo_matrix, output_img, offset_x, offset_y): 
    
    for i in range(0, output_img.shape[0]):
        for j in range(0, output_img.shape[1]):
            
            coord = np.array([j-offset_x,i-offset_y,1])
            coord = coord.reshape((-1,1))
        
            mapped_coord = np.matmul(np.linalg.pinv(homo_matrix), coord)
            mapped_coord = (mapped_coord/mapped_coord[2,0])

            mapped_coord_new = mapped_coord.reshape((3,1))
            
            #find the weighted pixels value
            if (0 <= mapped_coord_new[0] < image.shape[1]-1 and 0 <= mapped_coord_new[1] < image.shape[0]-1):
                yf = int(math.floor(mapped_coord_new[0]))
                yc = int(math.ceil(mapped_coord_new[0]))
                xf = int(math.floor(mapped_coord_new[1]))
                xc = int(math.ceil(mapped_coord_new[1]))

                wff = 1/np.linalg.norm(np.array([mapped_coord_new[1] - xf, mapped_coord_new[0] - yf])) 
                wfc = 1/np.linalg.norm(np.array([mapped_coord_new[1] - xf, mapped_coord_new[0] - yc])) 
                wcf = 1/np.linalg.norm(np.array([mapped_coord_new[1] - xc, mapped_coord_new[0] - yf])) 
                wcc = 1/np.linalg.norm(np.array([mapped_coord_new[1] - xc, mapped_coord_new[0] - yc]))
                
                pixel_value = wff*image[xf,yf,:] + wfc*image[xf,yc,:] + wcf*image[xc,yf,:] + wcc*image[xc,yc,:]
                weights = wff + wcc + wfc + wcf
                
                output_img[i,j,:] = pixel_value/weights
                
    #map the corresponsing pixel value and clip to RGB range
    output_img = np.clip(output_img, 0.0, 255.0)
    output_img = output_img.astype(np.uint8)
    
    return output_img


# In[13]:


#Apply the mapping function for Image1(building) to showcase the output world plane image and save the image to local computer

output_p2p_building_weighted = mapping_func_weighted(building_img, H_building, output_img_p2p_building, 149, 1466)
plt.imshow(output_p2p_building_weighted)
filename = 'D:\Purdue\ECE661_CV\HW3_outputs' + '\Image' + '_output_p2p_building_weighted' +'.png'
sk.imsave(filename, output_p2p_building_weighted)


# In[14]:


#Apply the mapping function for Image2(canvas) to showcase the output world plane image and save the image to local computer

output_p2p_canvas_weighted = mapping_func_weighted(canvas_img, H_canvas, output_img_p2p_canvas, 21, 440)
plt.imshow(output_p2p_canvas_weighted)
filename = 'D:\Purdue\ECE661_CV\HW3_outputs' + '\Image' + '_output_p2p_canvas_weighted' +'.png'
sk.imsave(filename, output_p2p_canvas_weighted)


# ### Task 1.2 (Two- step method)
# #### Removing Projective distortion

# In[15]:


#Change the points to HC representation by adding z axis coordinate

building_points_3 = np.array ([[240,119,1], [240,457,1],[723,450,1],[717,291,1]], np.int32)
canvas_points_3 = np.array ([[15,103,1], [15,727,1], [865,678,1], [862,162,1]], np.int32)


# In[16]:


#compute the homography function that gets rid of projective distortion

def proj_homo_matrix(points):

    #find l1(passing through P,S points) and l2 (passing through Q,R points) parallel lines in PQRS frame 
    l1 = np.cross(points[0],points[3])
    l2 = np.cross(points[1],points[2])
    
    #find l3(passing through P,Q points) and l4 (passing through S,R points) parallel lines in PQRS frame 
    l3 = np.cross(points[0],points[1]) 
    l4 = np.cross(points[3],points[2])
    
    #find the vanishing point of l1 and l2
    vp1 = np.cross(l1, l2)
    vp1 = vp1/vp1[2]
    
    #find the vanishing point of l3 and l4
    vp2 = np.cross(l3, l4)
    vp2 = vp2/vp2[2]
    
    #find the vanishing line from the two vanishing points
    vl = np.cross(vp1,vp2)
    print(vl)
    vl = vl/vl[2]
    print(vl)
    
    #compute H
    H = np.zeros((3,3))
    H[0][0] = 1
    H[1][1] = 1    
    H[2] = vl
    
    return H


# In[17]:


#Calculate the projective Homo matrix for the image plane and world plane points

H_proj_building = proj_homo_matrix(building_points_3)
H_proj_canvas = proj_homo_matrix(canvas_points_3)


# In[18]:


#finding the min and max of images1 and 2

find_min_max(building_img, H_proj_building)
find_min_max(canvas_img, H_proj_canvas)


# In[19]:


##Initialize a zero matrix as the world plane based on min and max values of the coordinates

output_img_proj_building = np.zeros((1700,2900,3), dtype = 'uint8')
output_img_proj_canvas = np.zeros((1400,2100,3), dtype = 'uint8')


# In[20]:


#Apply the mapping function for Image1(building) to showcase the output world plane image and save the image to local computer

image_proj_building = mapping_func_weighted(building_img, H_proj_building, output_img_proj_building, 0, 0)
plt.imshow(image_proj_building)
filename = 'D:\Purdue\ECE661_CV\HW3_outputs' + '\Image' + '_proj_building' +'.png'
sk.imsave(filename, image_proj_building)


# In[21]:


#Apply the mapping function for Image2(canvas) to showcase the output world plane image and save the image to local computer

image_proj_canvas = mapping_func_weighted(canvas_img, H_proj_canvas, output_img_proj_canvas, 0, 0)
plt.imshow(image_proj_canvas)
filename = 'D:\Purdue\ECE661_CV\HW3_outputs' + '\Image' + '_proj_canvas' +'.png'
sk.imsave(filename, image_proj_canvas)


# #### Removing Affine distortion

# In[22]:


#compute the homography function that gets rid of affine distortion by following the concept explained in the report

def affine_homo_matrix(points):
    
    #find the pair of orthogonal lines of PQRS frame i.e, PQ&PS, QR&RS pairs
    l1 = np.cross(points[0],points[1])
    l1 = l1/l1[2]
    
    m1 = np.cross(points[0],points[3])
    m1 = m1/m1[2]
    
    l2 = np.cross(points[1],points[2]) 
    l2 = l2/l2[2]
    
    m2 = np.cross(points[2],points[3])
    m2 = m2/m2[2]
    
    #create lists to append the linear equations
    mat1 = []
    mat2 = []

    mat1.append([ l1[0]*m1[0], l1[0]*m1[1] + l1[1]*m1[0] ])
    mat1.append([ l2[0]*m2[0], l2[0]*m2[1] + l2[1]*m2[0] ])

    mat2.append([ -l1[1]*m1[1] ])
    mat2.append([ -l2[1]*m2[1] ])
    
    mat1 = np.asarray(mat1)
    mat2 = np.asarray(mat2)
    
    #compute S matrix
    S_temp = np.dot(np.linalg.pinv(mat1), mat2)
    
    S = np.zeros((2,2))
    S[0][0] = S_temp[0]
    S[0][1] = S[1][0] = S_temp[1]
    S[1][1] = 1
    
    #Perform a Singular Value Decomposition on S and compute A from its values
    u,s,vh = np.linalg.svd(S)
    D = np.sqrt(np.diag(s))
    A = np.dot(np.dot(u,D),np.transpose(u))
    
    #compute H
    H = np.zeros((3,3))
    
    H[0][0] = A[0][0]
    H[0][1] = A[0][1]   
    H[1][0] = A[1][0]
    H[1][1] = A[1][1]
    H[2][2] = 1
    
    return H


# In[23]:


#Calculate the affine Homo matrix for the image plane and world plane points

H_affine_building = affine_homo_matrix(building_points_3)
H_affine_canvas = affine_homo_matrix(canvas_points_3)


# #### Removing both Projective and Affine distortion

# In[24]:


#Calculate the homography to get rid of both affine and projective distortions
#Multiply Affine homography matrix with projective homography matrix obtained for the images

H_affine_proj_building = np.dot(np.linalg.pinv(H_affine_building), H_proj_building)
H_affine_proj_canvas = np.dot(np.linalg.pinv(H_affine_canvas), H_proj_canvas)


# In[25]:


#finding the min and max of images1 and 2

find_min_max(building_img, H_affine_proj_building)
find_min_max(canvas_img, H_affine_proj_canvas)


# In[26]:


#Initialize a zero matrix as the world plane

output_img_affine_proj_building = np.zeros((1700,9300,3), dtype = 'uint8')
output_img_affine_proj_canvas = np.zeros((1500,9700,3), dtype = 'uint8')


# In[ ]:


#Apply the mapping function for Image1(building) to showcase the output world plane image and save the image to local computer

image_affine_proj_building = mapping_func_weighted(building_img, H_affine_proj_building, output_img_affine_proj_building, 46, 250)
plt.imshow(image_affine_proj_building)
filename = 'D:\Purdue\ECE661_CV\HW3_outputs' + '\Image' + '_affine_proj_building' +'.png'
sk.imsave(filename, image_affine_proj_building)


# In[ ]:


#Apply the mapping function for Image2(canvas) to showcase the output world plane image and save the image to local computer

image_affine_proj_canvas = mapping_func_weighted(canvas_img, H_affine_proj_canvas, output_img_affine_proj_canvas, 12, 25)
plt.imshow(image_affine_proj_canvas)
filename = 'D:\Purdue\ECE661_CV\HW3_outputs' + '\Image' + '_affine_proj_canvas' +'.png'
sk.imsave(filename, image_affine_proj_canvas)


# ### Task 1.3 (One- step method)

# In[ ]:


#Compute the one step method matrix based on the concept discussed in the report to remove affine and projective distortions

def onestep_homo_matrix(points):
    
    # Compute the pair of orthogonal lines from points of PQRS consdiering adjacent sides for each pair 
    # Choose a diagonal for the last orthogonal pair
    
    l1 = np.cross(points[0],points[1])
    l1 = l1/l1[2]
    
    m1 = np.cross(points[0],points[3])
    m1 = m1/m1[2]
    
    l2 = np.cross(points[0],points[1]) 
    l2 = l2/l2[2]
    
    m2 = np.cross(points[1],points[2])
    m2 = m2/m2[2]

    l3 = np.cross(points[1],points[2])
    l3 = l3/l3[2]
    
    m3 = np.cross(points[2],points[3])
    m3 = m3/m3[2]
    
    l4 = np.cross(points[2],points[3]) 
    l4 = l4/l4[2]
    
    m4 = np.cross(points[3],points[0])
    m4 = m4/m4[2]
    
    l5 = np.cross(points[0],points[2]) 
    l5 = l5/l5[2]
    
    m5 = np.cross(points[1],points[3])
    m5 = m5/m5[2]
    
    #give two matrixes to append the linear equations of the orthogonal lines according to the concept in the report
    mat1 = []
    mat2 = []

    mat1.append([ l1[0]*m1[0], (l1[0]*m1[1] + l1[1]*m1[0])/2, l1[1]*m1[1], (l1[0] + m1[0])/2, (l1[1] + m1[1])/2 ])
    mat1.append([ l2[0]*m2[0], (l2[0]*m2[1] + l2[1]*m2[0])/2, l2[1]*m2[1], (l2[0] + m2[0])/2, (l2[1] + m2[1])/2 ])
    mat1.append([ l3[0]*m3[0], (l3[0]*m3[1] + l3[1]*m3[0])/2, l3[1]*m3[1], (l3[0] + m3[0])/2, (l3[1] + m3[1])/2 ])
    mat1.append([ l4[0]*m4[0], (l4[0]*m4[1] + l4[1]*m4[0])/2, l4[1]*m4[1], (l4[0] + m4[0])/2, (l4[1] + m4[1])/2 ])
    mat1.append([ l5[0]*m5[0], (l5[0]*m5[1] + l5[1]*m5[0])/2, l5[1]*m5[1], (l5[0] + m5[0])/2, (l5[1] + m5[1])/2 ])


    mat2.append([ -l1[2]*m1[2] ])
    mat2.append([ -l2[2]*m2[2] ])
    mat2.append([ -l3[2]*m3[2] ])
    mat2.append([ -l4[2]*m4[2] ])
    mat2.append([ -l5[2]*m5[2] ])

    
    mat1 = np.asarray(mat1)
    mat2 = np.asarray(mat2)
    
    #Compute c_infinity to find their respective coefficents and determine S
    C_inf = np.dot(np.linalg.pinv(mat1), mat2)
    C_inf = C_inf/np.max(C_inf)
    
    S = np.zeros((2,2))
    S[0][0] = C_inf[0]
    S[0][1] = S[1][0] = C_inf[1]/2
    S[1][1] = C_inf[2]
    
    #compute SVD of matrix S
    u,s,vh = np.linalg.svd(S)
    D = np.sqrt(np.diag(s))
    
    #compute matrix A from the SVD values computed 
    A = np.dot(np.dot(u,D),np.transpose(u))
    
    #Compute matrix V from A
    V = np.dot(np.linalg.pinv(A), np.array([C_inf[3]/2, C_inf[4]/2]))

    #Determine H
    H = np.zeros((3,3))
    
    H[0][0] = A[0][0]
    H[0][1] = A[0][1]   
    H[1][0] = A[1][0]
    H[1][1] = A[1][1]
    H[2][0] = V[0]
    H[2][1] = V[1]
    H[2][2] = 1
    print(H)
    
    return H


# In[ ]:


#Calculate the 1-step method Homo matrix for the image plane and world plane points

H_1step_building = onestep_homo_matrix(building_points_3)
H_1step_canvas = onestep_homo_matrix(canvas_points_3)


# In[ ]:


#finding the min and max of image 1

find_min_max(building_img, H_1step_building)


# In[ ]:


#finding the min and max of image 2

find_min_max(canvas_img, H_1step_canvas)


# In[ ]:


# Initialize a zero matrix as the world plane based on min and max values of the coordinates

output_img_1step_building = np.zeros((250,550,3), dtype = 'uint8')
output_img_1step_canvas = np.zeros((600,1200,3), dtype = 'uint8')


# In[ ]:


#Apply the mapping function for Image1(building) to showcase the output world plane image and save the image to local computer

image_1step_building = mapping_func_weighted(building_img, H_1step_building, output_img_1step_building, 0, 0)
plt.imshow(image_1step_building)
filename = 'D:\Purdue\ECE661_CV\HW3_outputs' + '\Image' + '_1step_building' +'.png'
sk.imsave(filename, image_1step_building)


# In[ ]:


#Apply the mapping function for Image2(canvas) to showcase the output world plane image and save the image to local computer

image_1step_canvas = mapping_func_weighted(canvas_img, H_1step_canvas, output_img_1step_canvas, 0, 0)
plt.imshow(image_1step_canvas)
filename = 'D:\Purdue\ECE661_CV\HW3_outputs' + '\Image' + '_1step_canvas' +'.png'
sk.imsave(filename, image_1step_canvas)


# ## Task-2
# ### Task 2.1 (Point-to-point correspondence method)

# In[ ]:


# Read the images from uplaoded files/directly by giving path in the computer

frame_img = sk.imread('frame.jpg')
lappy_img = sk.imread('lappy.jpg')


# In[ ]:


# Create an array of pixel corodinates located at the four corners of the Region of Interest (RoI) in the images 
# Here the GIMP - GUI based tool is used to estimate the pixel ordinates of the images

# The target coordinates are chosen based on the height and width of the images

frame_points = np.array ([[156,37], [130,251], [346,291], [375,43]], np.int32)
frame_target_points = np.array([[0,0], [0,339],[509,339],[509,0]], np.int32)

lappy_points = np.array ([[306,73], [344,424], [656,307], [639,7]], np.int32)
lappy_target_points = np.array([[0,0], [0,628],[1200,628],[1200,0]], np.int32)


# In[ ]:


#verify that the points are as desired using boundary box function

frame_bb = boundarybox_check(frame_img, frame_points)
lappy_bb = boundarybox_check(lappy_img, lappy_points)
plt.imshow(lappy_bb)


# In[ ]:


#Calculate the Homo matrix of images between the image and world plane

H_frame = homo_matrix(frame_points, frame_target_points)
H_lappy = homo_matrix(lappy_points, lappy_target_points)


# In[ ]:


#finding min and max of coordinates of frame image

find_min_max(frame_img, H_frame)


# In[ ]:


#finding min and max of coordinates of lappy image

find_min_max(lappy_img, H_lappy)


# In[ ]:


#Initialize a zero matrix as the world plane based on min and max values of the coordinates

output_img_p2p_frame = np.zeros((700,1400,3), dtype = 'uint8')
output_img_p2p_lappy = np.zeros((2800,6000,3), dtype = 'uint8')


# In[ ]:


#Apply the mapping function for Image3(frame) to showcase the output world plane image and save the image to local computer

output_p2p_frame_weighted = mapping_func_weighted(frame_img, H_frame, output_img_p2p_frame, 488, 60)
plt.imshow(output_p2p_frame_weighted)
filename = 'D:\Purdue\ECE661_CV\HW3_outputs' + '\Image' + '_output_p2p_frame_weighted' +'.png'
sk.imsave(filename, output_p2p_frame_weighted)


# In[ ]:


#Apply the mapping function for Image4(lappy) to showcase the output world plane image and save the image to local computer

output_p2p_lappy_weighted = mapping_func_weighted(lappy_img, H_lappy, output_img_p2p_lappy, 986, 191)
plt.imshow(output_p2p_lappy_weighted)
filename = 'D:\Purdue\ECE661_CV\HW3_outputs' + '\Image' + '_output_p2p_lappy_weighted' +'.png'
sk.imsave(filename, output_p2p_lappy_weighted)


# ### Task 2.2 (Two - step method)
# #### Removing projective distortion

# In[ ]:


#Change the points to HC representation by adding z axis coordinate

frame_points_3 = np.array ([[156,37,1], [130,251,1], [346,291,1], [375,43,1]], np.int32)
lappy_points_3 = np.array ([[306,73,1], [344,424,1], [656,307,1], [639,7,1]], np.int32)


# In[ ]:


#Calculate the projective Homo matrix of images between the image and world plane

H_proj_frame = proj_homo_matrix(frame_points_3)
H_proj_lappy = proj_homo_matrix(lappy_points_3)


# In[ ]:


#finding min and max of coordinates of frame image

find_min_max(frame_img, H_proj_frame)


# In[ ]:


#finding min and max of coordinates of lappy image

find_min_max(lappy_img, H_proj_lappy)


# In[ ]:


#Initialize a zero matrix as the world plane based on min and max values of the coordinates

output_img_proj_frame = np.zeros((350,400,3), dtype = 'uint8')
output_img_proj_lappy = np.zeros((1500,2850,3), dtype = 'uint8')


# In[ ]:


#Apply the mapping function for Image3(frame) to showcase the output world plane image and save the image to local computer

image_proj_frame = mapping_func_weighted(frame_img, H_proj_frame, output_img_proj_frame, 0, 0)
plt.imshow(image_proj_frame)
filename = 'D:\Purdue\ECE661_CV\HW3_outputs' + '\Image' + '_proj_frame' +'.png'
sk.imsave(filename, image_proj_frame)


# In[ ]:


#Apply the mapping function for Image4(lappy) to showcase the output world plane image and save the image to local computer

image_proj_lappy = mapping_func_weighted(lappy_img, H_proj_lappy, output_img_proj_lappy, 0, 0)
plt.imshow(image_proj_lappy)
filename = 'D:\Purdue\ECE661_CV\HW3_outputs' + '\Image' + '_proj_lappy' +'.png'
sk.imsave(filename, image_proj_lappy)


# #### Removing affine distortion

# In[ ]:


#compute the affine homography for images 3 and 4 (frame and lappy)

H_affine_frame = affine_homo_matrix(frame_points_3)
H_affine_lappy = affine_homo_matrix(lappy_points_3)


# #### Removing affine & projective distortion

# In[ ]:


#Calculate the homography to get rid of both affine and projective distortions
#Multiply Affine homography matrix with projective homography matrix obtained for the images

H_affine_proj_frame = np.dot(np.linalg.pinv(H_affine_frame), H_proj_frame)
H_affine_proj_lappy = np.dot(np.linalg.pinv(H_affine_lappy), H_proj_lappy)


# In[ ]:


#finding min and max of coordinates of frame image

find_min_max(frame_img, H_affine_proj_frame)


# In[ ]:


#finding min and max of coordinates of lappy image

find_min_max(lappy_img, H_affine_proj_lappy)


# In[ ]:


#Initialize a zero matrix as the world plane based on min and max values of the coordinates

output_img_affine_proj_frame = np.zeros((450,2200,3), dtype = 'uint8')
output_img_affine_proj_lappy = np.zeros((1500,5000,3), dtype = 'uint8')


# In[ ]:


#Apply the mapping function for Image3(frame) to showcase the output world plane image and save the image to local computer

image_affine_proj_frame = mapping_func_weighted(frame_img, H_affine_proj_frame, output_img_affine_proj_frame, 0, 0)
plt.imshow(image_affine_proj_frame)
filename = 'D:\Purdue\ECE661_CV\HW3_outputs' + '\Image' + '_affine_proj_frame' +'.png'
sk.imsave(filename, image_affine_proj_frame)


# In[ ]:


#Apply the mapping function for Image4(lappy) to showcase the output world plane image and save the image to local computer

image_affine_proj_lappy = mapping_func_weighted(lappy_img, H_affine_proj_lappy, output_img_affine_proj_lappy, 67, 253)
plt.imshow(image_affine_proj_lappy)
filename = 'D:\Purdue\ECE661_CV\HW3_outputs' + '\Image' + '_affine_proj_lappy' +'.png'
sk.imsave(filename, image_affine_proj_lappy)


# ### Task 2.3 (One-step method)

# In[ ]:


#Calculate the 1-step method Homo matrix for the image plane and world plane points

H_1step_frame = onestep_homo_matrix(frame_points_3)
H_1step_lappy = onestep_homo_matrix(lappy_points_3)


# In[ ]:


#finding the min and max of image 1

find_min_max(frame_img, H_1step_frame)


# In[ ]:


#finding the min and max of image 1

find_min_max(lappy_img, H_1step_lappy)


# In[ ]:


# Initialize a zero matrix as the world plane based on min and max values of the coordinates

output_img_1step_frame = np.zeros((550,650,3), dtype = 'uint8')
output_img_1step_lappy = np.zeros((550,850,3), dtype = 'uint8')


# In[ ]:


#Apply the mapping function for Image3(frame) to showcase the output world plane image and save the image to local computer

image_1step_frame = mapping_func_weighted(frame_img, H_1step_frame, output_img_1step_frame, 24, 53)
plt.imshow(image_1step_frame)
filename = 'D:\Purdue\ECE661_CV\HW3_outputs' + '\Image' + '_1step_frame' +'.png'
sk.imsave(filename, image_1step_frame)


# In[ ]:


#Apply the mapping function for Image4(lappy) to showcase the output world plane image and save the image to local computer

image_1step_lappy = mapping_func_weighted(lappy_img, H_1step_lappy, output_img_1step_lappy, 13, 17)
plt.imshow(image_1step_lappy)
filename = 'D:\Purdue\ECE661_CV\HW3_outputs' + '\Image' + '_1step_lappy' +'.png'
sk.imsave(filename, image_1step_lappy)

