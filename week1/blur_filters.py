import cv2

img = cv2.imread('elephant.jpg')

k_size = 11 # kernel size 
blur_img = cv2.blur(img, (k_size, k_size))
median_blur_img = cv2.medianBlur(img, k_size)
gaussian_blur_img = cv2.GaussianBlur(img, (k_size, k_size), 5)

#resizing for better visualization
target_size = (600, 300)
img_s = cv2.resize(img, target_size)
blur_s = cv2.resize(blur_img, target_size)
median_s = cv2.resize(median_blur_img, target_size)
gauss_s = cv2.resize(gaussian_blur_img, target_size)

cv2.imshow('Original Image', img_s)
cv2.imshow('Average Blur', blur_s)
cv2.imshow('Gaussian Blur', gauss_s)
cv2.imshow('Median Blur', median_s)
cv2.waitKey(0)