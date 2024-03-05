import cv2


cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
cap3 = cv2.VideoCapture(2)
cap4 = cv2.VideoCapture(4)

if cap.isOpened():
    if cap2.isOpened():
        if cap3.isOpened():
            if cap4.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2600)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
                cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 2600)
                cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
                cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 2600)
                cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
                cap4.set(cv2.CAP_PROP_FRAME_WIDTH, 2600)
                cap4.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
                while True:
                    read_code, frame = cap.read()
                    read_code2, frame2 = cap2.read()
                    read_code3, frame3 = cap2.read()
                    read_code4, frame4 = cap2.read()
                    if not read_code or not read_code2 or not read_code3 or not read_code4:
                        break
                    cv2.imshow("1", frame)
                    cv2.imshow("2", frame2)
                    cv2.imshow("3", frame3)
                    cv2.imshow('4', frame4)
                    if cv2.waitKey(1) == ord('q'):
                        print('cap', cap)
                        print('cap2', cap2)
                        print('cap3', cap3)
                        print('cap4', cap4)
                        cap.release()
                        print('capr', cap.release())
                        cap2.release()
                        print('cap2r', cap2.release())
                        cap3.release()
                        print('cap3r', cap3.release())
                        break
cap4.release()
print('cap4r', cap4.release())
cv2.destroyAllWindows()



