import cv2
import sys
from yolov7 import runModel
import matplotlib.pyplot as plt
from Transformation import align_images_Perspective_sift
from Transformation import align_images_affine_sift


def PrintRowLine():
    print(' ', end='')
    for i in range(0, 5):
        for j in range(0, 8):
            print('-', end='')
        print(' ', end='')
    print()


def PrintColLine():
    for i in range(0, 5):
        print('|', end='')
        for j in range(0, 8):
            print(' ', end='')
    print('|')


source_image_path = sys.argv[1]
transformation_mode = int(sys.argv[2])
target_image_path = "./IMG_7671.jpg"
default_box = [[0, 2, 0, 0, 0],
               [2, 0, 0, 0, 4],
               [5, 2, 1, 5, 4],
               [10, 3, 3, 2, 4]
               ]
if source_image_path:
    if transformation_mode == 0:
        image = align_images_affine_sift(source_image_path, target_image_path)
    elif transformation_mode == 1:
        image = align_images_Perspective_sift(source_image_path, target_image_path)
    if image is not None:
        box_num, img = runModel(image)  # 輸入yolov7模型偵測
        tmp = 0
        for i in range(0, 4):  # 輸出偵測的結果
            PrintRowLine()
            PrintColLine()
            for j in range(0, 5):
                if default_box[i][j] == 0:
                    print("|  None  ", end='')
                else:
                    print("|  {:02d}/{:02d} ".format(box_num[tmp], default_box[i][j]), end='')
                tmp += 1
            print('|')
            PrintColLine()
        PrintRowLine()
        plt.imshow(img)  # 偵測結果圖片
        plt.pause(0)
            
    else:
        print("Can't read image!")
else:
    print("File Not Exit!")
