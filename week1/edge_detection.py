import cv2

pic = cv2.imread('ronaldo.png')
pic_edge = cv2.Canny(pic, 120, 255)#this is adjusted according to the image
cv2.imshow('Image', pic)
cv2.imshow('Edge Detected Image', pic_edge)
cv2.waitKey(0)
