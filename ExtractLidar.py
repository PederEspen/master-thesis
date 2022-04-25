import glob
import os
import sys
import time
from cv2 import log
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
import logging
import random

capture_count = 0

def saveActorLabels(vehicles, pedestrians):
    actors = []
    for vehicle in vehicles:
        actors.append([vehicle.id, vehicle.type_id])
    for pedestrian in pedestrians:
        actors.append([pedestrian.id, pedestrian.type_id])
    f = open('ids_labels.txt', 'w')
    for actor in actors:
        f.write(str(actor[0]) + ',' + str(actor[1]) + '\n')
    f.close()
    
def save_plt(point_cloud):
    global capture_count
    capture_count += 1
    print('Saved image number', capture_count)
    point_cloud.save_to_disk('./semantic_lidar_output/%.6d.ply' % point_cloud.frame)

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:
        ego_vehicle = None
        world = client.get_world()
        world.wait_for_tick()

        vehicles = world.get_actors().filter('vehicle.*')
        pedestrians = world.get_actors().filter('walker.*')
        saveActorLabels(vehicles, pedestrians)

        # --------------
        # Spawn ego vehicle
        # --------------
        
        ego_bp = world.get_blueprint_library().find('vehicle.mini.cooper_s')
        ego_bp.set_attribute('role_name','ego')
        ego_color = random.choice(ego_bp.get_attribute('color').recommended_values)
        ego_bp.set_attribute('color',ego_color)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if 0 < number_of_spawn_points:
            random.shuffle(spawn_points)
            ego_transform = spawn_points[0]
            ego_vehicle = world.spawn_actor(ego_bp,ego_transform)
            print('Starting lidar data collection..')
        else: 
            logging.warning('Could not found any spawn points')

        ego_vehicle.set_autopilot(True)

        # Lidar settings are for FPS=20, image with dimensions 1024x128
        lidar_sem_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        lidar_sem_bp.set_attribute('channels',str(128))
        lidar_sem_bp.set_attribute('points_per_second',str(2621440))
        lidar_sem_bp.set_attribute('upper_fov',str(11.25))
        lidar_sem_bp.set_attribute('lower_fov',str(-11.25))
        lidar_sem_bp.set_attribute('rotation_frequency',str(20))
        lidar_sem_bp.set_attribute('range',str(240))
        lidar_sem_bp.set_attribute('sensor_tick',str(5))
        lidar_location = carla.Location(0,0,2)
        lidar_rotation = carla.Rotation(0,0,0)
        lidar_transform = carla.Transform(lidar_location,lidar_rotation)

        lidar_sem_sen = world.spawn_actor(lidar_sem_bp,lidar_transform,attach_to=ego_vehicle,attachment_type=carla.AttachmentType.Rigid)
        lidar_sem_sen.listen(lambda point_cloud: save_plt(point_cloud))


        # --------------
        # Game loop. Prevents the script from finishing.
        # --------------
        while True:
            #attach spectator view to sensor
            world.get_spectator().set_transform(lidar_sem_sen.get_transform())
            world_snapshot = world.wait_for_tick()

    finally:
        # --------------
        # Stop recording and destroy actors
        # --------------
        client.stop_recorder()
        if ego_vehicle is not None:
            if lidar_sem_sen is not None:
                lidar_sem_sen.stop()
                lidar_sem_sen.destroy()
            ego_vehicle.destroy()

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nDone with tutorial_ego.')