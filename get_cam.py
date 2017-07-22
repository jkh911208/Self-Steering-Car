import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(3,350) # set height
cap.set(4,200) # set width

def get_frame():
    # read the frame from webcam
    _, frame = cap.read()

    # change the color to gray
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # webcam is install in car upside down, so flip the image
    frame = cv2.flip(frame,0)
    frame = cv2.flip(frame,1)

    # cut the top part of image because it is sky and bottom of image
    # frame = frame[150:-15,::]

    # resize the image by 80% to make it right size to feed into CNN
    frame = cv2.resize(frame,(0,0),fx=0.8, fy=0.8)

    # print(frame.shape)
    # cv2.imshow('frame',frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    return frame

def main():
    while True:
        frame = get_frame()

        print(frame.shape[0], frame.shape[1])
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break   

main()

cap.release()
# cv2.destroyAllWindows()
