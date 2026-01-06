import cv2
import numpy as np
import os

def get_images():
    if not os.path.exists("left.jpg"):
        os.system("curl -L -o left.jpg https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/stitching/boat1.jpg")
    if not os.path.exists("right.jpg"):
        os.system("curl -L -o right.jpg https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/stitching/boat2.jpg")
    return cv2.imread('left.jpg'), cv2.imread('right.jpg')

# 1. Load Images 
img1, img2 = get_images()
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 2. Keypoint Detection & Descriptor Computation
orb = cv2.ORB_create(nfeatures=2000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# 3. Feature Matching 
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 4. Robust Estimation (RANSAC) 
good_matches = matches[:int(len(matches) * 0.15)] 

src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

# 5. Image Warping & Stitching 
h, w, c = img2.shape
warped_img = cv2.warpPerspective(img2, M, (w + img1.shape[1], h))
warped_img[0:img1.shape[0], 0:img1.shape[1]] = img1

# 6. Output 
cv2.imwrite('output_panorama.jpg', warped_img)
