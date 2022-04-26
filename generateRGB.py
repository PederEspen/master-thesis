#!/usr/bin/env python

import glob
import os
import sys
import json

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import copy
import os.path
import time
import math
import matplotlib.pyplot as plt

import carla

from carla import ColorConverter as cc

import cv2
import re

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# Create Directory ################
dir_my = 'my_data/'
dir_custom = 'custom_data/'
dir_draw = 'draw_bounding_box/'
if not os.path.exists(dir_my):
    os.makedirs(dir_my)
if not os.path.exists(dir_custom):
    os.makedirs(dir_custom)
if not os.path.exists(dir_draw):
    os.makedirs(dir_draw)
###################################

path, dirs, files = next(os.walk('custom_data/'))
files = [ fi for fi in files if fi.endswith(".png")]
print(files)
dataEA = len(files)



VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

VBB_COLOR = (0, 0, 255)
WBB_COLOR = (255, 0, 0)
TLBB_COLOR = (30, 170, 250)
TSBB_COLOR = (0, 220, 220)

#Vehicle_COLOR = (142, 0, 0)
Vehicle_COLOR = (10, 0, 0)
Walker_COLOR = (60, 20, 220)
Traffic_light_COLOR = (30, 170, 250)
Traffic_sign_COLOR = (0, 220, 220)

rgb_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
seg_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
area_info = np.zeros(shape=[VIEW_HEIGHT, VIEW_WIDTH, 3], dtype=np.uint8)
index_count = 0

# Brings Images and Bounding Box Information
def reading_data(index):
    global rgb_info, seg_info
    v_data = []
    w_data = []
    k = 0
    w = 0

    rgb_img = cv2.imread('custom_data/image'+ str(index)+ '.png', cv2.IMREAD_COLOR)
    seg_img = cv2.imread('SegmentationImageSeg/seg'+ str(index)+ '.png', cv2.IMREAD_COLOR)
    #print("seg image ", index, seg_img)
    #print(np.shape(seg_img))

    if str(rgb_img) != "None" and str(seg_img) != "None":
        # Vehicle
        bounding_box_rawdata = ''
        ids = ''
        types = []
        with open('ActorsBBox/bbox'+ str(index) + '.txt', 'r') as f:
            lines_in_ABBox = f.readlines()
        
        #Get every 3rd line starting from line 0, 1 and 2 respectively for id, type and bbox.
        ids = ids + str(lines_in_ABBox[::3])
        types = types + lines_in_ABBox[1::3]
        bounding_box_rawdata = bounding_box_rawdata + str(lines_in_ABBox[2::3])
        
        #Finding the numbers for bbox and ids, and removing /n for types
        ids = re.findall(r"-?\d+", ids)
        types = [type[:-1] for type in types]
        bounding_box_data = re.findall(r"-?\d+", bounding_box_rawdata)
        line_length = int(len(bounding_box_data) / 16)
        bbox_data = [[0 for col in range(8)] for row in range(line_length)] 


        for i in range(int(len(bounding_box_data)/2)):
            j = i*2
            v_data.append(tuple((int(bounding_box_data[j]), int(bounding_box_data[j+1]))))

        #print("v_data: ", v_data)

        for i in range(int(len(bounding_box_data)/16)):
            for j in range(8):
                bbox_data[i][j] = v_data[k]
                k += 1
       

        origin_rgb_info = rgb_img
        rgb_info = rgb_img
        seg_info = seg_img
        return bbox_data, line_length, ids, types

    else:
        return False

# Converts 8 Vertices to 4 Vertices
def converting(bounding_boxes, line_length):
    points_array = []
    bb_4data = [[0 for col in range(4)] for row in range(line_length)]
    k = 0
    for i in range(line_length):
        points_array_x = []
        points_array_y = []      
        for j in range(8):
            points_array_x.append(bounding_boxes[i][j][0])
            points_array_y.append(bounding_boxes[i][j][1])

            max_x = max(points_array_x)
            min_x = min(points_array_x)
            max_y = max(points_array_y)
            min_y = min(points_array_y)           

        points_array.append(tuple((min_x, min_y)))
        points_array.append(tuple((max_x, min_y)))
        points_array.append(tuple((max_x, max_y)))
        points_array.append(tuple((min_x, max_y)))

    for i in range(line_length):
        for j in range(int(len(points_array)/line_length)):
            bb_4data[i][j] = points_array[k]
            k += 1  

    #print('lads', bb_4data)
    return bb_4data

# Gets Object's Bounding Box Area
def object_area(data):
    global area_info
    area_info = np.zeros(shape=[VIEW_HEIGHT, VIEW_WIDTH, 3], dtype=np.uint8)

    for vehicle_area in data:
        array_x = []
        array_y = []
        for i in range(4):
           array_x.append(vehicle_area[i][0])
        for j in range(4):
           array_y.append(vehicle_area[j][1])

        for i in range(4):
            if array_x[i] <= 0:
                array_x[i] = 1
            elif array_x[i] >= VIEW_WIDTH:
                array_x[i] = VIEW_WIDTH -1
        for j in range(4):
            if array_y[j] <= 0:
                array_y[j] = 1
            elif array_y[j] >= VIEW_HEIGHT:
                array_y[j] = VIEW_HEIGHT -1
       
        min_x = min(array_x) 
        max_x = max(array_x) 
        min_y = min(array_y) 
        max_y = max(array_y) 
        array = [min_x, max_x, min_y, max_y]
        if filtering(array, Vehicle_COLOR): 
            cv2.rectangle(area_info, (min_x, min_y), (max_x, max_y), Vehicle_COLOR, -1)

# Fits Bounding Box to the Object
def fitting_x(x1, x2, range_min, range_max, color):
    global seg_info
    
    state = False
    cali_point = 0
    if (x1 < x2):
        for search_point in range(x1, x2):
            for range_of_points in range(range_min, range_max):
                if seg_info[range_of_points, search_point][1] == color[1] and seg_info[range_of_points, search_point][0] == color[0]:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    else:
        for search_point in range(x1, x2, -1):
            for range_of_points in range(range_min, range_max):
                if seg_info[range_of_points, search_point][1] == color[1] and seg_info[range_of_points, search_point][0] == color[0]:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break
    
    return cali_point

def fitting_y(y1, y2, range_min, range_max, color):
    global seg_info

    state = False
    cali_point = 0

    if (y1 < y2):
        for search_point in range(y1, y2):
            for range_of_points in range(range_min, range_max):
                if seg_info[search_point, range_of_points][1] == color[1] and seg_info[search_point, range_of_points][0] == color[0]: 
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    else:
        for search_point in range(y1, y2, -1):
            for range_of_points in range(range_min, range_max):
                if seg_info[search_point, range_of_points][1] == color[1] and seg_info[search_point, range_of_points][0] == color[0]:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    return cali_point

# Removes small objects that obstruct to learning
def small_objects_excluded(array, bb_min_x, bb_min_y):
    diff_x = array[1] - array[0]
    diff_y = array[3] - array[2]
    if (diff_x > bb_min_x and diff_y > bb_min_y):
        return True
    return False

# Filters occluded objects
def post_occluded_objects_excluded(array, color):
   
    global seg_info
    top_left = seg_info[array[2]+1, array[0]+1][0]
    top_right = seg_info[array[2]+1, array[1]-1][0] 
    bottom_left = seg_info[array[3]-1, array[0]+1][0] 
    bottom_right = seg_info[array[3]-1, array[1]-1][0]
    if top_left == color[0] and top_right == color[0] and bottom_left == color[0] and bottom_right == color[0]:
        return False
    
    return True

def pre_occluded_objects_excluded(array, area_image, color):

    top_left = area_image[array[2]-1, array[0]-1][0]
    top_right = area_image[array[2], array[1]+1][0] 
    bottom_left = area_image[array[3]+1, array[1]-1][0] 
    bottom_right = area_image[array[3]+1, array[0]+1][0]
    
    if top_left == color[0] and top_right == color[0] and bottom_left == color[0] and bottom_right == color[0]:
        return False
    
    return True

def get_vehicle_class(type_id, json_path=None):
    f = open(json_path)
    json_data = json.load(f)
    vehicles_data = json_data['classification']
    other_class = json_data["reference"].get('others')
    vehic = vehicles_data.get(type_id, other_class)
    if (type(vehic) == int):
        v_class = int(vehic)
        return v_class

# Filters objects not in the scene
def filtering(array, color):
    global seg_info
    for y in range(array[2], array[3]):
        for x in range(array[0], array[1]):
            if seg_info[y, x][0] == color[0] and seg_info[y, x][1] == color[1]:
                return True
    return False

# Processes Post-Processing
def processing(img, a_data, index, a_ids, a_types):
    global seg_info, area_info
    traffic_light_class = 5
    traffic_sign_class = 6

    object_area(a_data)
    f = open("custom_data/image"+str(index) + ".txt", 'w')
    json_path = 'vehicle_class_json_file.txt'

    
    # Actor 
    a_data = [[x, y, z] for x, y, z in zip(a_ids, a_data, a_types)] 
    for a_bbox in a_data:
        # disregard objects marked with ID 9, as this is an error from Carla which
        # we have not able to remove. Happens very rarely on distant objects,
        # where the proper objectID is not found

        array_x = []
        array_y = []
        for i in range(4):
            array_x.append(a_bbox[1][i][0])
        for j in range(4):
            array_y.append(a_bbox[1][j][1])

        for i in range(4):
            if array_x[i] <= 0:
                array_x[i] = 1
            elif array_x[i] >= VIEW_WIDTH - 1:
                array_x[i] = VIEW_WIDTH - 2
        for j in range(4):
            if array_y[j] <= 0:
                array_y[j] = 1
            elif array_y[j] >= VIEW_HEIGHT - 1:
                array_y[j] = VIEW_HEIGHT - 2
    
        min_x = min(array_x) 
        max_x = max(array_x) 
        min_y = min(array_y) 
        max_y = max(array_y) 
        a_bb_array = [min_x, max_x, min_y, max_y]

        actor_id = int(a_bbox[0])
        actor_color_red =  10
        actor_color_green = (actor_id & 0x00ff)
        actor_color_blue = (actor_id & 0xff00) >> 8

        actor_color = (actor_color_blue, actor_color_green, actor_color_red)

        a_class = get_vehicle_class(a_bbox[2], json_path=json_path)
        if filtering(a_bb_array, actor_color) and pre_occluded_objects_excluded(a_bb_array, area_info, actor_color):
            
            cali_min_x = fitting_x(min_x, max_x, min_y, max_y, actor_color)
            cali_max_x = fitting_x(max_x, min_x, min_y, max_y, actor_color)
            cali_min_y = fitting_y(min_y, max_y, min_x, max_x, actor_color)
            cali_max_y = fitting_y(max_y, min_y, min_x, max_x, actor_color)
            a_cali_array = [min_x, max_x, min_y, max_y]

            tightened_percent_min_x = abs((min_x - cali_min_x) / (max_x - min_x))
            tightened_percent_max_x = abs((max_x - cali_max_x) / (max_x - min_x))
            tightened_percent_min_y = abs((min_y - cali_min_y) / (max_y - min_y))
            tightened_percent_max_y = abs((max_y - cali_max_y) / (max_y - min_y))
            
            # if tightened_percent_max_x > 0.3:
            #     cali_max_x = max_x
            
            # # if tightened_percent_max_y < 0.15:
            # #     cali_max_y = max_y
            
            # if tightened_percent_min_x > 0.3:
            #     cali_min_x = min_x
            
            # if tightened_percent_min_y > 0.30:
            #     cali_min_y = min_y

            cali_max_x = max_x
            cali_max_y = max_y
            cali_min_x = min_x
            cali_min_y = min_y

            bb_min_x = VIEW_HEIGHT*0.0125 ## Small bboxes 1.25% of max height
            bb_min_y = VIEW_WIDTH*0.0125 ## Small bboxes 1.25% of max width
            if small_objects_excluded(a_cali_array, bb_min_x, bb_min_y) and post_occluded_objects_excluded(a_cali_array, actor_color):

                darknet_x = float((cali_min_x + cali_max_x) // 2) / float(VIEW_WIDTH)
                darknet_y = float((cali_min_y + cali_max_y) // 2) / float(VIEW_HEIGHT)
                darknet_width = float(cali_max_x - cali_min_x) / float(VIEW_WIDTH)
                darknet_height= float(cali_max_y - cali_min_y) / float(VIEW_HEIGHT)

                f.write(str(a_class) + ' ' + str("%0.6f" % darknet_x) + ' ' + str("%0.6f" % darknet_y) + ' ' + 
                str("%0.6f" % darknet_width) + ' ' + str("%0.6f" % darknet_height) + "\n")


                if a_class == 4:
                    ABB_COLOR = (0, 0, 255)
                else:
                    ABB_COLOR = (255, 0, 0)

                cv2.line(img, (cali_min_x, cali_min_y), (cali_max_x, cali_min_y), ABB_COLOR, 2)
                cv2.line(img, (cali_max_x, cali_min_y), (cali_max_x, cali_max_y), ABB_COLOR, 2)
                cv2.line(img, (cali_max_x, cali_max_y), (cali_min_x, cali_max_y), ABB_COLOR, 2)
                cv2.line(img, (cali_min_x, cali_max_y), (cali_min_x, cali_min_y), ABB_COLOR, 2)


    # Traffic lights
    tl_bbox_array = detect_static_objects(seg_info, 18, 15)
    for bbox in tl_bbox_array:
        min_x = bbox[0]
        max_x = bbox[1]
        min_y = bbox[2]
        max_y = bbox[3]
        darknet_x = float((min_x + max_x) // 2) / float(VIEW_WIDTH)
        darknet_y = float((min_y + max_y) // 2) / float(VIEW_HEIGHT)
        darknet_width = float(max_x - min_x) / float(VIEW_WIDTH)
        darknet_height= float(max_y - min_y) / float(VIEW_HEIGHT)

        f.write(str(traffic_light_class) + ' ' + str("%0.6f" % darknet_x) + ' ' + str("%0.6f" % darknet_y) + ' ' + 
        str("%0.6f" % darknet_width) + ' ' + str("%0.6f" % darknet_height) + "\n")

        cv2.line(img, (min_x, min_y), (max_x, min_y), TLBB_COLOR, 2)
        cv2.line(img, (max_x, min_y), (max_x, max_y), TLBB_COLOR, 2)
        cv2.line(img, (max_x, max_y), (min_x, max_y), TLBB_COLOR, 2)
        cv2.line(img, (min_x, max_y), (min_x, min_y), TLBB_COLOR, 2)

    # Traffic signs
    ts_bbox_array = detect_static_objects(seg_info, 12, 15)
    for bbox in ts_bbox_array:
        min_x = bbox[0]
        max_x = bbox[1]
        min_y = bbox[2]
        max_y = bbox[3]

        darknet_x = float((min_x + max_x) // 2) / float(VIEW_WIDTH)
        darknet_y = float((min_y + max_y) // 2) / float(VIEW_HEIGHT)
        darknet_width = float(max_x - min_x) / float(VIEW_WIDTH)
        darknet_height= float(max_y - min_y) / float(VIEW_HEIGHT)

        f.write(str(traffic_sign_class) + ' ' + str("%0.6f" % darknet_x) + ' ' + str("%0.6f" % darknet_y) + ' ' + 
        str("%0.6f" % darknet_width) + ' ' + str("%0.6f" % darknet_height) + "\n")

        cv2.line(img, (min_x, min_y), (max_x, min_y), TSBB_COLOR, 2)
        cv2.line(img, (max_x, min_y), (max_x, max_y), TSBB_COLOR, 2)
        cv2.line(img, (max_x, max_y), (min_x, max_y), TSBB_COLOR, 2)
        cv2.line(img, (min_x, max_y), (min_x, min_y), TSBB_COLOR, 2)

    f.close()
    cv2.imwrite('draw_bounding_box/image'+str(index)+'.png', img)

def detect_static_objects(seg_img, red_channel, threshold):
    thresholded_image = cv2.inRange(seg_img, (0, 0, red_channel), (255, 255, red_channel))
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        x_values = []
        y_values = []
        for i in range (len(contour)):
            x_values.append(contour[i][0][0])
            y_values.append(contour[i][0][1])
        min_x = min(x_values)
        max_x = max(x_values)
        min_y = min(y_values)
        max_y = max(y_values)
        # only add bounding boxes that are over the threshold in diagonal
        #if (max_x-min_x > threshold and max_y-min_y > threshold):
        if (math.hypot(max_x - min_x, max_y - min_y) > threshold):
            bboxes.append([min_x, max_x, min_y, max_y])

    return bboxes


def removeImages():

    _, _, seg_images = next(os.walk('SegmentationImageSeg/'))

    _, _, rgb_images = next(os.walk('custom_data/'))
    rgb_images = [ fi for fi in rgb_images if fi.endswith(".png")]

    image_rgb = []
    image_seg = []
    for x in rgb_images:
        image_rgb.append(re.findall(r'\d+', x)[0])
    
    for y in seg_images:
        image_seg.append(re.findall(r'\d+', y)[0])

    remove_files = [x for x in image_rgb if x not in image_seg]
    count = 0
    for x in remove_files:
        count += 1
        if os.path.isfile('./custom_data/image' + x + '.png'):
            os.remove('./custom_data/image' + x + '.png')
        if os.path.isfile('./custom_data/image' + x + '.txt'):
            os.remove('./custom_data/image' + x + '.txt')
        
    print("Images removed", count)

            


def run():
    global rgb_info
    global index_count
    removeImages()
    for i in range(1, dataEA + 1):
        if reading_data(i) != False:
            #Vehicle?
            a_four_points = converting(reading_data(i)[0], reading_data(i)[1])
            processing(rgb_info, a_four_points, i, reading_data(i)[2], reading_data(i)[3])
            #index_count = index_count + 1
            index_count += 1
            print("Image number", index_count, 'processed')
    print('Images processed', index_count)

if __name__ == "__main__":
    start = time.time()

    run()

    end = time.time()
    print('Script runtime:', float(end - start), 'seconds')