import cv2
import numpy as np
import matplotlib.pyplot as plt

def median_blur(image, kernel_size):
    assert kernel_size in [3,5,7,9], "O tamanho do filtro deve estar entre 3,5,7,9"
    mb = cv2.medianBlur(image, kernel_size)
    return mb

def gaussian_blur(image, std, ksize = 11):
    assert 0<=std<=1, "o desvio padrão do filtro deve estar entre 0 e 1"
    gb = cv2.GaussianBlur(image, (ksize, ksize), std)
    return gb

def color_space(image, transformation):
    if transformation == 'RGB':
        cs = image.copy()
    elif transformation == 'LAB':
        cs = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    elif transformation == 'HSV':
        cs = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif transformation == 'XYZ':
        cs = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
    elif transformation == 'YCBCR':
        cs = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    return cs

def choose_channel(image, channel):
    if channel == 'all':
        cc = image
    elif channel == 'first':
        cc = image[:,:,0]
    elif channel == 'second':
        cc = image[:,:,1]
    elif channel == 'third':
        cc = image[:,:,2]
    return cc

def erode(image, size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    e = cv2.erode(image, kernel, iterations=1)
    return e

def dilate(image, size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    d = cv2.dilate(image, kernel, iterations=1)
    return d

def segmentation(image, segmentation_type):
    if segmentation_type == 'otsu': #OTSU
        if len(image.shape)>2:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image.copy()
        ret, bit_mask = cv2.threshold(image_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
    elif segmentation_type == 'clustering': #kmeans
        if len(image.shape)>2:
            vectorized = image.reshape((-1, 3)).astype(np.float32)
        else:
            vectorized = image.reshape((-1,1)).astype(np.float32)

        attempts = 10
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.9)

        ret,label,center = cv2.kmeans(vectorized,2,None,criteria,attempts,cv2.KMEANS_RANDOM_CENTERS)

        bit_mask = label.reshape((image.shape[:2]))
        bit_mask = cv2.normalize(bit_mask.astype('float32'), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        bit_mask = bit_mask.astype(np.uint8)
        
    elif segmentation_type == 'adaptive': #adaptive threshold
        if len(image.shape)>2:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image.copy()
        bit_mask = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
    elif segmentation_type == 'pca': #PCA
        if len(image.shape)<3:
            projected_image = image.copy()
        else:
            matrix = image.reshape((-1, 3)).astype(np.float32)
            mean_matrix = np.mean(matrix, axis=0, keepdims=True)
        
            centered_matrix = matrix - mean_matrix

            cov_matrix = np.cov(centered_matrix, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            
            max_arg = np.argmax(eigenvalues)
            top_eigenvector = eigenvectors[:, max_arg]
            
            projected_data = np.dot(centered_matrix, top_eigenvector)
            
            projected_image = projected_data.reshape((image.shape[:2]))        
        
        pca_image = cv2.normalize(projected_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ret, bit_mask = cv2.threshold(pca_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
    return bit_mask