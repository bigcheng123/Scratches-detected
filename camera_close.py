import cv2


cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
cap3 = cv2.VideoCapture(2)
cap4 = cv2.VideoCapture(3)
cap5 = cv2.VideoCapture(4)
cap6 = cv2.VideoCapture(5)


cap1.release()
print('capr', cap1.release())
cap2.release()
print('cap2r', cap2.release())
cap3.release()
print('cap3r', cap3.release())
cap4.release()
print('cap4r', cap4.release())
cap5.release()
print('cap5r', cap5.release())
cap6.release()
print('cap6r', cap6.release())
cv2.destroyAllWindows()



