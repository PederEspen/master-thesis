import os
def countLabels():
  path = 'train/labels'
  car_count = 0
  truck_count = 0
  cyclist_count = 0
  pedestrian_count = 0

  with os.scandir(path) as it:
      for entry in it:
          fo = open(entry.path, 'r')
          lines = fo.readlines()
          for line in lines:
            if line[0] == '0':
              car_count = car_count + 1
            if line[0] == '1':
              truck_count = truck_count + 1
            if line[0] == '2':
              cyclist_count = cyclist_count + 1
            if line[0] == '3':
              pedestrian_count = pedestrian_count + 1
            
  print('Car count:', car_count)
  print('Truck count:', truck_count)
  print('Cyclist count:', cyclist_count)
  print('Pedestrian count:', pedestrian_count)

countLabels()