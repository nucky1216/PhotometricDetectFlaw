import numpy as np
import cv2

a=np.array([[1,0,0],
            [0,4,0],
            [0,0,4]])
print(np.linalg.svd(a))
print(np.linalg.eig(a))
