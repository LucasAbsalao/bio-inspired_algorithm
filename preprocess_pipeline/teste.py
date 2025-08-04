from metrics import *
import cv2

b1 = cv2.imread("dataset_folhas/batata1.JPG")
b1_mask = cv2.imread("dataset_folhas/batata1_mask.png")

print(global_consistency_error(b1,b1), global_consistency_error(b1,b1_mask))