from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np

class Drone:
    yaw_velocity = 0
    forward_back_velocity = 0
    up_down_velocity = 0
    yaw_velocity = 0


def getFrame(camera, w=640,h=480):
    img = cv2.resize(camera,(w,h))
    return img

def trackBodyInXAxis(myDrone,info,w,pid,pError):
 
    ## PID
    error = info[0][0] - w//2
    speed = pid[0]*error + pid[1]*(error-pError)
    speed = int(np.clip(speed,-100,100))
 
 
    #print(speed)
    if info[0][0] !=0:
        myDrone.yaw_velocity = speed
    else:
        myDrone.for_back_velocity = 0
        myDrone.left_right_velocity = 0
        myDrone.up_down_velocity = 0
        myDrone.yaw_velocity = 0
        error = 0
    return error


def trackBodyInYAxis(myDrone,info,h,pid,pError):
 
    ## PID
    error = info[0][1] - h//2
    # error = info[1] - 70000
    speed = pid[0]*error + pid[1]*(error-pError)
    speed = int(np.clip(speed,-100,100))
 
    if info[0][0] !=0:
        myDrone.forward_back_velocity = -speed
    else:
        myDrone.forward_back_velocity = 0
        myDrone.left_right_velocity = 0
        myDrone.up_down_velocity = 0
        myDrone.yaw_velocity = 0
        error = 0
    return error

def findBody(img, boxes,annotator):
    myFaceListCenter = []
    myFaceListArea = []
    for box in boxes:
        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
        x, y, w, h = np.array(box.xywh.cpu(), dtype=np.int32).squeeze() # get box coordinates in (x, y, w, h) format
        annotator.box_label(b, 'Tracking Unit') # add box at top-left corner
        area = w*h
        myFaceListArea.append(area)
        myFaceListCenter.append([x,y])
        img = cv2.circle(img, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
    if len(myFaceListArea) !=0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListCenter[i],myFaceListArea[i]]
    else:
        return img,[[0,0],0]

def main():
    model = YOLO("yolov8m.pt")
    cap = cv2.VideoCapture(0)
    _with,_heigh = 640,480
    pid = [0.4,0.4,0]
    pErrorX = 0
    pErrorY = 0
    myDrone = Drone()
   
    while True:
        _, img = cap.read()
        img = getFrame(img,_with,_heigh)
        results = model.predict(img, tracker = 'bytetrack.yaml', classes=0, max_det = 1, verbose = False)
        img = cv2.circle(img, (_with//2, _heigh//2), radius=5, color=(0, 255, 0), thickness=-1)
        for r in results:
            annotator = Annotator(img) # init annotator
            boxes = r.boxes
            img, info = findBody(img, boxes,annotator)
            pErrorX = trackBodyInXAxis(myDrone,info,_with,pid,pErrorX)
            pErrorY = trackBodyInYAxis(myDrone,info,_heigh,pid,pErrorY)
            img = cv2.putText(img, 'Target Coordinate', (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            img = cv2.putText(img, f'X: {pErrorX}', (550, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            img = cv2.putText(img, f'Y: {pErrorY}', (551, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            img = cv2.putText(img, f'Yaw Speed: {myDrone.yaw_velocity}', (350, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            img = cv2.putText(img, f'Prop Speed: {myDrone.forward_back_velocity}', (350, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            # print('Yaw: ',myDrone.yaw_velocity)
            # print('Forward: ',myDrone.forward_back_velocity)
            
        img = annotator.result()   # get annotated image
        cv2.imshow('YOLO V8 Detection', img)     
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()