import cv2
import sys

from video import create_capture
from common import clock, draw_str
from face_points_detector import FacePointsClassifier

def draw_rects(img, rects, color):
    for x1, y1, w, h in rects:
        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), color, 2)

if __name__ == '__main__':
    from net_common import FlipBatchIterator, AdjustVariable, EarlyStopping
    clss = FacePointsClassifier('net_cpu.pickle')

    if len(sys.argv) == 2:
        img_name = sys.argv[1]
        img = cv2.imread(img_name)
        for (x,y,w,h,face_points) in clss.detect_keypoints(img):
                for (x, y) in face_points:
                    cv2.circle(img,(x,y), 2, (0,0,255), 2)
        cv2.imshow('img',img)
        cv2.waitKey(0) 
    else:
        cam = create_capture(0, fallback='synth:bg=../data/lena.jpg:noise=0.05')
        while True:
            ret, img = cam.read()
            t = clock()
            vis = img.copy()
            for (x,y,w,h,face_points) in clss.detect_keypoints(img):
                for (x, y) in face_points:
                    cv2.circle(vis,(x,y), 2, (0,0,255), 2)
            dt = clock() - t

            draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
            cv2.imshow('facedetect', vis)

            if cv2.waitKey(1) == 27: # ESC key
                break
    cv2.destroyAllWindows()
