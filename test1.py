import cv2
file = r"C:\Users\CCSX009\Desktop\FH\CHAU1\qw\2023-06-21_11-41-08-294112-1CD.jpg"
img2_orgin = cv2.imread(file)
print(type(img2_orgin) == type(None))
cv2.imshow('a', img2_orgin)
cv2.waitKey(0)
cv2.destroyAllWindows()