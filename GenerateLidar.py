import numpy as np
import os
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import math
from readPLY import readPLY
import json

# Color values used to create segmentation images.
# The values are taken from the Cityscapes color palette.
seg_colors = {
  0: (0, 0, 0),
  1: (70, 70, 70),
  2: (100, 40, 40),
  3: (50, 90, 80),
  4: (220, 20, 60),
  5: (153, 153, 153),
  6: (157, 234, 50),
  7: (128, 64, 128),
  8: (244, 35, 232),
  9: (107, 142, 35),
  10: (0, 0, 142),
  11: (102, 102, 156),
  12: (220, 220, 0),
  13: (70, 130, 180),
  14: (81, 0, 81),
  15: (150, 100, 100),
  16: (230, 150, 140),
  17: (180, 165, 180),
  18: (250, 170, 30),
  19: (110, 190, 160),
  20: (170, 120, 50),
  21: (45, 60, 150),
  22: (145, 170, 100)
}

# create folders that dont exist
dir_data = 'lidar_data/'
dir_imgs= 'lidar_imgs/'

if not os.path.exists(dir_data):
    os.makedirs(dir_data)
    os.makedirs(dir_data + 'range/')
    os.makedirs(dir_data + 'intensity/')
if not os.path.exists(dir_imgs):
    os.makedirs(dir_imgs)
    os.makedirs(dir_imgs + 'drawbbox/')
    os.makedirs(dir_imgs + 'segmentation/')

# Function to visualize the point cloud using the Open3D package
def displayPCD(array):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(array)
  o3d.visualization.draw_geometries([pcd], zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])

# Function to show images. Mostly used for debugging purposes.
def renderImages(img_range, img_seg):
  fig = plt.figure()
  rows = 2
  columns = 1
  fig.add_subplot(rows, columns, 1)
  plt.imshow(np.transpose(img_range), cmap='gray', vmin=0, vmax=255)
  plt.axis('off')
  plt.title("Range")
  fig.add_subplot(rows, columns, 2)
  plt.imshow(np.transpose(img_seg, (1, 0, 2)))
  plt.axis('off')
  plt.title("Segmentation Image")
  plt.show()    

# Function to save images to disk.
def saveImages(img_r, img_i, img_bbox, img_seg, filename):
  plt.imsave('./lidar_data/range/' + filename + '_range.png', np.transpose(img_r), cmap='gray')
  plt.imsave('./lidar_data/intensity/' + filename + '_intensity.png', np.transpose(img_i), cmap='gray')
  plt.imsave('./lidar_imgs/drawbbox/' + filename + '_bbox.png', np.transpose(img_bbox), cmap='gray')
  plt.imsave('./lidar_imgs/segmentation/' + filename + '_segmentation.png', np.transpose(img_seg, (1, 0, 2)))

def processPointCloud(filename):
  img_range = np.zeros((1024, 128), dtype=np.uint8)
  img_seg = np.zeros((1024, 128, 3), dtype=np.uint8)
  img_labeled = np.zeros((1024, 128))
  pcd_points = readPLY(filename)['points'].to_numpy()
  pcd_array = []
  for point in pcd_points:
    x = point[0]
    y = point[1]
    z = point[2]
    object_id = point[4]
    object_tag = point[5]
    pcd_array.append([x, y, z])

    r = np.linalg.norm([x, y, z])
    r_norm = (r/240) * 255
    azimuth = math.atan2(x, y)
    elevation = math.asin(z / r)
    elevation_up = math.radians(11.25)
    elevation_down = math.radians(-11.25)

    u = round(0.5 * (1 + azimuth/math.pi) * 1024)
    v = math.floor(((elevation_up - elevation) / (elevation_up - elevation_down)) * 128)

    img_range[u-1][v-1] = r_norm
    img_seg[u-1][v-1][:] = seg_colors[object_tag]
    img_labeled[u-1][v-1] = int(object_id)

  img_intensity = np.invert(img_range)
  return img_range, img_intensity, img_seg, img_labeled

def get_vehicle_class(vehicles, json_path=None):
    f = open(json_path)
    json_data = json.load(f)
    vehicles_data = json_data['classification']
    other_class = json_data["reference"].get('others')
    class_list = []
    for v in vehicles:
        type_id = v.split(',')[1][:-1]
        v_class = vehicles_data.get(type_id, other_class)
        if (type(v_class) == int):
            v_class = int(v_class)
            class_list.append(v_class)
    return class_list

def detect_objects(img, actor_id):
    x_values = []
    y_values = []
    for i in range (np.shape(img)[0]):
      for j in range(np.shape(img)[1]):
        if img[i][j] == actor_id:
          x_values.append(i)
          y_values.append(j)

    if (x_values and y_values):
      min_x = np.min(x_values)
      max_x = np.max(x_values)
      min_y = np.min(y_values)
      max_y = np.max(y_values)

      # If the object appears on both sides of the image (due to it being 360 degrees), dont draw the 
      # bounding box. Or else it will span the entire screen
      if ((max_x - min_x) < 900):
        bbox = [min_x, max_x, min_y, max_y]
        return bbox
    
    return []
  
def draw_and_save_bounding_boxes(img, img_labeled, filename):
  bboxes = []
  img_copy = np.copy(img)
  with open('ids_labels.txt') as labels:
    labels_array = labels.readlines()
    all_classes = get_vehicle_class(labels_array, 'vehicle_class_json_file.txt')
    filtered_classes = []
    for label in labels_array:
      object = label.split(',')
      object_id = object[0]
      bbox = detect_objects(img_labeled, int(object_id))
      #if any bounding boxes are found (meaning the object is in the current point cloud)
      if bbox:
        bboxes.append(bbox)
        filtered_classes.append(all_classes[labels_array.index(label)])
  
  VIEW_WIDTH = np.shape(img)[0]
  VIEW_HEIGHT = np.shape(img)[1]
  file_range = open("lidar_data/range/" + filename + "_range.txt", 'w')
  file_intensity = open("lidar_data/intensity/" + filename + "_intensity.txt", 'w')

  # any bounding boxes with an area under this pixel value will not be saved
  threshold_area = 30

  for i in range(len(bboxes)):
    min_x = bboxes[i][0]
    max_x = bboxes[i][1]
    min_y = bboxes[i][2]
    max_y = bboxes[i][3]

    if (max_x - min_x) * (max_y - min_y)  > threshold_area:
      #save darknet format on bounding boxes
      darknet_x = float((min_x + max_x) // 2) / float(VIEW_WIDTH)
      darknet_y = float((min_y + max_y) // 2) / float(VIEW_HEIGHT)
      darknet_width = float(max_x - min_x) / float(VIEW_WIDTH)
      darknet_height= float(max_y - min_y) / float(VIEW_HEIGHT)

      if (darknet_width != 0 and darknet_height != 0):
        file_range.write(str(filtered_classes[i]) + ' ' + str("%0.6f" % darknet_x) + ' ' + str("%0.6f" % darknet_y) + ' ' + 
        str("%0.6f" % darknet_width) + ' ' + str("%0.6f" % darknet_height) + "\n")
        file_intensity.write(str(filtered_classes[i]) + ' ' + str("%0.6f" % darknet_x) + ' ' + str("%0.6f" % darknet_y) + ' ' + 
        str("%0.6f" % darknet_width) + ' ' + str("%0.6f" % darknet_height) + "\n")

      #draw white bounding boxes on images
      #opencv .line expects (y,x) coords instead of (x,y).
      min_x = bboxes[i][2]
      max_x = bboxes[i][3]
      min_y = bboxes[i][0]
      max_y = bboxes[i][1]
      cv2.line(img_copy, (min_x, min_y), (max_x, min_y), (255, 255, 255), 1)
      cv2.line(img_copy, (max_x, min_y), (max_x, max_y), (255, 255, 255), 1)
      cv2.line(img_copy, (max_x, max_y), (min_x, max_y), (255, 255, 255), 1)
      cv2.line(img_copy, (min_x, max_y), (min_x, min_y), (255, 255, 255), 1)
  
  file_range.close()
  file_intensity.close()

  return img_copy

def main():
  path = './semantic_lidar_output/'
  count = 0
  with os.scandir(path) as dir:
    for pcl_file in dir:
      filename = str(pcl_file.name[:-4])
      img_range, img_intensity, img_seg, img_labeled = processPointCloud(path + filename + '.ply')
      img_bbox = draw_and_save_bounding_boxes(img_range, img_labeled, filename)
      saveImages(img_range, img_intensity, img_bbox, img_seg, filename)

      count += 1
      print('Saved', count, 'images')

main()
