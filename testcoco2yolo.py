import os
import cv2
# import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def normalize_keypoints(keypoints, image_width, image_height):
    normalized_keypoints = []

    for x, y in keypoints:
        normalized_x = x / image_width
        normalized_y = y / image_height
        normalized_keypoints.append((normalized_x, normalized_y))

    return normalized_keypoints

def denormalize_keypoints(normalized_keypoints, image_width, image_height):
    denormalized_keypoints = []

    for norm_x, norm_y in normalized_keypoints:
        denormalized_x = norm_x * image_width
        denormalized_y = norm_y * image_height
        denormalized_keypoints.append((denormalized_x, denormalized_y))

    return denormalized_keypoints

def coco_to_pascal_voc(x, y, width, height):
    # x, y, width, height = coco_bbox
    xmin = x
    ymin = y
    xmax = x + width
    ymax = y + height
    return xmin, ymin, xmax, ymax

def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return [abs((x2 + x1)/(2*image_w)), abs((y2 + y1)/(2*image_h)), abs((x2 - x1)/image_w), abs((y2 - y1)/image_h)]

def yolo_to_pascal_voc(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]

part = 'train'
# part = 'val'

dst = 'destination path'

images_p = os.path.join(dst, 'images', part)
txts_p = os.path.join(dst, 'labels', part)

txts = os.listdir(txts_p)

for txt in tqdm(txts):
    
    idd = txt.split('.')[0]
    img_path = os.path.join(images_p, idd+'.jpg')
    img = cv2.imread(img_path)
    
    width = img.shape[1]
    height = img.shape[0]

    txt_path = os.path.join(txts_p, idd+'.txt')
    # Using readlines()
    file1 = open(txt_path, 'r')
    lines = file1.readlines()

    for line in lines:
        
        data_list = line.strip().split()
        
        xmin, ymin, xmax, ymax = yolo_to_pascal_voc(float(data_list[1]), float(data_list[2]), 
                                                    float(data_list[3]), float(data_list[4]), 
                                                    width, height)
        
        points = data_list[5:]
        keypoints = list(map(float, points))
        
        points = []
        for i in range(0, len(keypoints), 3):
            points.append((keypoints[i], keypoints[i+1]))
        points2 = denormalize_keypoints(points, width, height)
        
        # Draw the bounding box
        cv2.rectangle(img, (int(xmin),int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)  # (0, 255, 0) is the color in BGR format, and 2 is the thickness
    
        # Draw keypoints on the image
        for x, y in points2:
            if (x, y) != (0, 0):
                cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)  # Draw a green circle at each keypoint
            
    # Plot the image using matplotlib
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Turn off axis labels
    plt.show() 
    plt.clf() 


