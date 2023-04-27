import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(os.getcwd()))
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from ultralytics import YOLO

def infer():
    model = YOLO('8npt/best.pt')
    results = model('img.jpg')
    print(results[0].plot())
    cv2.imwrite('result.png', results[0].plot())

if __name__ == '__main__':
    infer()