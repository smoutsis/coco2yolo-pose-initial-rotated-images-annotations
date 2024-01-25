# Import the libraries
import os
import cv2
import json
# import pandas as pd
from tqdm import tqdm 
# import matplotlib.pyplot as plt
from itertools import zip_longest

# A def that trasnform the lists to the desied txts for the yolov8 training
def create_text_file(list1, list2, list3, output_file):
    # Open a file for writing
    with open(output_file, 'w') as file:
        # Iterate through the sublists using zip_longest
        for items in zip_longest(list1, list2, list3, fillvalue=[]):
            # Flatten each sublist and combine items into a single line
            line = ' '.join(str(item) for sublist in items for item in sublist) + '\n'
            
            # Write the line to the file
            file.write(line)

# Rotate the bounding boxxes and coordinates for the rotated images. The rotation angle can be 90, 270 and 180. 
# In any other value the script will display as error
def rotate_coordinates_and_keypoints(x1, y1, x2, y2, keypoints, width, height, rotation_angle):
    # Calculate the center of the bounding box
    
    new_points = []
    # print(keypoints)
    if rotation_angle == 90:
        x1_r90 = (height - y2)
        x2_r90 = (height - y1)
        y1_r90 = x2
        y2_r90 = x1
        start_point = (int(x1_r90), int(y1_r90))
        end_point = (int(x2_r90), int(y2_r90))
        
        for points in keypoints:
            if points != (0, 0):
                x, y = points
                x_90 = height - y
                y_90 = x
                new_points.append((x_90, y_90))
            else:
                new_points.append((0, 0))
    
    elif rotation_angle == 270:
        x1_r270 = y1
        x2_r270 = y2
        y1_r270 = (width - x2)
        y2_r270 = (width - x1)
        start_point = (int(x1_r270), int(y1_r270))
        end_point = (int(x2_r270), int(y2_r270))
        
        for points in keypoints:
            if points != (0, 0):
                x, y = points
                x_270 = y
                y_270 = width - x
                new_points.append((x_270, y_270))
            else:
                new_points.append((0, 0))
                
    elif rotation_angle == 180:
        x1_r180 = width - x2
        x2_r180 = width - x1
        y1_r180 = height - y1
        y2_r180 = height - y2
        start_point = (int(x1_r180), int(y1_r180))
        end_point = (int(x2_r180), int(y2_r180))

        for points in keypoints:
            if points != (0, 0):
                x, y = points
                x_180 = width - x
                y_180 = height - y
                new_points.append((x_180, y_180))
            else:
                new_points.append((0, 0))
        
    else:
        x1_r0 = x1
        x2_r0 = x2
        y1_r0 = y1
        y2_r0= y2
        start_point = (int(x1_r0), int(y1_r0))
        end_point = (int(x2_r0), int(y2_r0))

        for points in keypoints:
            if points != (0, 0):
                x, y = points
                new_points.append((x, y))
            else:
                new_points.append((0, 0))
                
    xmi, ymi = start_point
    xma, yma = end_point
    
    return xmi, ymi, xma, yma, new_points

# A def that normalize the coordinates values based to the image shape
def normalize_keypoints(keypoints, image_width, image_height):
    normalized_keypoints = []

    for x, y in keypoints:
        normalized_x = x / image_width
        normalized_y = y / image_height
        normalized_keypoints.append((normalized_x, normalized_y))

    return normalized_keypoints

# A def that denormalize the coordinates values based to the image shape
def denormalize_keypoints(normalized_keypoints, image_width, image_height):
    denormalized_keypoints = []

    for norm_x, norm_y in normalized_keypoints:
        denormalized_x = norm_x * image_width
        denormalized_y = norm_y * image_height
        denormalized_keypoints.append((denormalized_x, denormalized_y))

    return denormalized_keypoints

#  The followin defs are used the transform the bounding boxxes to different formats

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

# choose which of the subset you want to transform 
# part = 'train'
part = 'val'

root_path = 'root path'
dst = 'destination path'

path = os.path.join(root_path, 'annotations_trainval2017/annotations')
file = 'person_keypoints_'+part+'2017.json'
imgs_path = os.path.join(root_path,part+'2017')

# If you do not want the rotated images and annotations make the followng flag "rotated_flag" from True -> False
rotated_flag = True

json_file_path  = os.path.join(path,file)

# Open the JSON file for reading
with open(json_file_path, 'r') as file2:
    # Load the JSON data
    data = json.load(file2)

length = len(data['annotations'])

in_list = []
dict_ids = {}
for i in tqdm(range(length)):
    
    img_id = str(data['annotations'][i]['image_id'])
    
    if img_id not in in_list:
        dict_ids[img_id] = [i]
        in_list.append(img_id)
    elif img_id in in_list:
        dict_ids[img_id].append(i)

for ids in tqdm(in_list):
    
    indexes = dict_ids[ids]
    
    classes = []
    bboxes = []
    keypoints = []
    
    if rotated_flag:
        classes_90 = []
        bboxes_90 = []
        keypoints_90 = []
        
        classes_270 = []
        bboxes_270 = []
        keypoints_270 = []
        
        classes_180 = []
        bboxes_180 = []
        keypoints_180 = []
    
    img_id = str(ids).zfill(12)+'.jpg'
    img = cv2.imread(os.path.join(imgs_path,img_id))
    if rotated_flag:
        img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_180 = cv2.rotate(img, cv2.ROTATE_180)
    
    width = img.shape[1]
    height = img.shape[0]
    if rotated_flag:
        width_90 = img_90.shape[1]
        height_90 = img_90.shape[0]
        width_270 = img_270.shape[1]
        height_270 = img_270.shape[0]
        width_180 = img_180.shape[1]
        height_180 = img_180.shape[0]
    
    for i in indexes:
        
        clas = ([0])
        
        bbox = (data['annotations'][i]['bbox'])
        keypoint = (data['annotations'][i]['keypoints'])
        
        points = []
        for j in range(0, len(keypoint), 3):
            points.append((keypoint[j], keypoint[j+1]))
        
        
        xmin, ymin, xmax, ymax = coco_to_pascal_voc(bbox[0], bbox[1], bbox[2], bbox[3])
        
        if rotated_flag:
            xmin_90, ymin_90, xmax_90, ymax_90, points_90 = rotate_coordinates_and_keypoints(
                xmin, ymin, xmax, ymax, points, width, height, rotation_angle=90
            )
            
            xmin_270, ymin_270, xmax_270, ymax_270, points_270 = rotate_coordinates_and_keypoints(
                xmin, ymin, xmax, ymax, points, width, height, rotation_angle=270
            )
            
            xmin_180, ymin_180, xmax_180, ymax_180, points_180 = rotate_coordinates_and_keypoints(
                xmin, ymin, xmax, ymax, points, width, height, rotation_angle=180
            )
        
        points_n = normalize_keypoints(points, width, height)
        if rotated_flag:
            points_90_n = normalize_keypoints(points_90, width_90, height_90)
            points_270_n = normalize_keypoints(points_270, width_270, height_270)
            points_180_n = normalize_keypoints(points_180, width_180, height_180)
        
        x_n, y_n, w_n, h_n = pascal_voc_to_yolo(xmin, ymin, xmax, ymax, width, height)
        if rotated_flag:
            x_90_n, y_90_n, w_90_n, h_90_n = pascal_voc_to_yolo(xmin_90, ymin_90, xmax_90, ymax_90, width_90, height_90)
            x_270_n, y_270_n, w_270_n, h_270_n = pascal_voc_to_yolo(xmin_270, ymin_270, xmax_270, ymax_270, width_270, height_270)
            x_180_n, y_180_n, w_180_n, h_180_n = pascal_voc_to_yolo(xmin_180, ymin_180, xmax_180, ymax_180, width_180, height_180)
        
        key = keypoint.copy()
        if rotated_flag:
            key_90 = keypoint.copy()
            key_270 = keypoint.copy()
            key_180 = keypoint.copy()
        
        for k in range(len(keypoint)):
            if (k+1)%3 != 0:
                key[k] = -5
                if rotated_flag:
                    key_90[k] = -5
                    key_270[k] = -5 
                    key_180[k] = -5 
        
        c = 0
        for x, y in points_n:
            key[c] = x
            key[c+1] = y
            c += 3
        if rotated_flag:
            c = 0
            for x, y in points_90_n:
                key_90[c] = x
                key_90[c+1] = y
                c += 3
            c = 0
            for x, y in points_270_n:
                key_270[c] = x
                key_270[c+1] = y
                c += 3
            c = 0
            for x, y in points_180_n:
                key_180[c] = x
                key_180[c+1] = y
                c += 3
            
        classes.append(clas)
        bboxes.append([x_n, y_n, w_n, h_n])
        keypoints.append(key)
        
        if rotated_flag:
            classes_90.append(clas)
            bboxes_90.append([x_90_n, y_90_n, w_90_n, h_90_n])
            keypoints_90.append(key_90)
            
            classes_270.append(clas)
            bboxes_270.append([x_270_n, y_270_n, w_270_n, h_270_n])
            keypoints_270.append(key_270)
            
            classes_180.append(clas)
            bboxes_180.append([x_180_n, y_180_n, w_180_n, h_180_n])
            keypoints_180.append(key_180)
        
    create_text_file(classes, bboxes, keypoints, os.path.join(dst,'labels',part,img_id.split('.')[0]+'.txt'))
    if rotated_flag:
        create_text_file(classes_90, bboxes_90, keypoints_90, os.path.join(dst,'labels',part,img_id.split('.')[0]+'_90.txt'))
        create_text_file(classes_270, bboxes_270, keypoints_270, os.path.join(dst,'labels',part,img_id.split('.')[0]+'_270.txt'))
        create_text_file(classes_180, bboxes_180, keypoints_180, os.path.join(dst,'labels',part,img_id.split('.')[0]+'_180.txt'))
    
    cv2.imwrite(os.path.join(dst,'images',part,img_id.split('.')[0]+'.jpg'), img)
    if rotated_flag:
        cv2.imwrite(os.path.join(dst,'images',part,img_id.split('.')[0]+'_90.jpg'), img_90)
        cv2.imwrite(os.path.join(dst,'images',part,img_id.split('.')[0]+'_270.jpg'), img_270)
        cv2.imwrite(os.path.join(dst,'images',part,img_id.split('.')[0]+'_180.jpg'), img_180)


