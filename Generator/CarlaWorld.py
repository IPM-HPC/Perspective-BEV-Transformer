import sys
import settings

sys.path.append(settings.CARLA_EGG_PATH)
import carla
import random
import time
import numpy as np

from spawn_npc import NPCClass
from client_bounding_boxes import ClientSideBoundingBoxes
from set_synchronous_mode import CarlaSyncMode
from bb_filter import apply_filters_to_3d_bb
from WeatherSelector import WeatherSelector
from saver import treat_single_image

from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions


class CarlaWorld:
    def __init__(self, HDF5_file):
        self.HDF5_file = HDF5_file
        # Carla initialization
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world('Town01_Opt') # changing the map
        self.world = self.client.get_world()
        print('Successfully connected to CARLA')
        self.blueprint_library = self.world.get_blueprint_library()
        # Sensors stuff
        self.camera_x_location = 1.0
        self.camera_y_location = 0.0
        self.camera_z_location = 2.0
        self.sensors_list = []
        # Weather stuff
        self.weather_options = WeatherSelector().get_weather_options()  # List with weather options

        # Recording stuff
        self.total_recorded_frames = 0
        self.first_time_simulating = True

    def set_weather(self, weather_option):
        # Changing weather https://carla.readthedocs.io/en/stable/carla_settings/
        # Weather_option is one item from the list self.weather_options, which contains a list with the parameters
        weather = carla.WeatherParameters(*weather_option)
        self.world.set_weather(weather)

    def remove_npcs(self):
        print('Destroying actors...')
        self.NPC.remove_npcs()
        print('Done destroying actors.')

    def spawn_npcs(self, number_of_vehicles, number_of_walkers):
        self.NPC = NPCClass()
        self.vehicles_list, _ = self.NPC.create_npcs(number_of_vehicles, number_of_walkers)

    def put_rgb_sensor(self, vehicle, sensor_width=640, sensor_height=480, fov=110):
        # https://carla.readthedocs.io/en/latest/cameras_and_sensors/
        bp = self.blueprint_library.find('sensor.camera.rgb')
        # bp.set_attribute('enable_postprocess_effects', 'True')  # https://carla.readthedocs.io/en/latest/bp_library/
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', f'{fov}')

        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=self.camera_x_location, z=self.camera_z_location))
        self.rgb_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        self.rgb_camera.blur_amount = 0.0
        self.rgb_camera.motion_blur_intensity = 0
        self.rgb_camera.motion_max_distortion = 0

        # Camera calibration
        calibration = np.identity(3)
        calibration[0, 2] = sensor_width / 2.0
        calibration[1, 2] = sensor_height / 2.0
        calibration[0, 0] = calibration[1, 1] = sensor_width / (2.0 * np.tan(fov * np.pi / 360.0))
        self.rgb_camera.calibration = calibration  # Parameter K of the camera
        self.sensors_list.append(self.rgb_camera)
        return self.rgb_camera

    def put_depth_sensor(self, vehicle, sensor_width=640, sensor_height=480, fov=110):
        # https://carla.readthedocs.io/en/latest/cameras_and_sensors/
        bp = self.blueprint_library.find('sensor.camera.depth')
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', f'{fov}')

        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=self.camera_x_location, z=self.camera_z_location))
        self.depth_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        self.sensors_list.append(self.depth_camera)
        return self.depth_camera

    def put_birdview_sensor(self,
                            width,
                            height,
                            pixels_per_meter,
                            render_lanes_on_junctions=False,
                            crop_type=BirdViewCropType.FRONT_AREA_ONLY):
        self.birdview_producer = BirdViewProducer(
            self.client,
            target_size=PixelDimensions(width=width, height=height),
            pixels_per_meter=pixels_per_meter,
            render_lanes_on_junctions=render_lanes_on_junctions,
            crop_type=crop_type
        )

    def process_depth_data(self, data, sensor_width, sensor_height):
        """
        normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        in_meters = 1000 * normalized
        """
        data = np.array(data.raw_data)
        data = data.reshape((sensor_height, sensor_width, 4))
        data = data.astype(np.float32)
        # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
        normalized_depth = np.dot(data[:, :, :3], [65536.0, 256.0, 1.0])
        normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
        depth_meters = normalized_depth * 1000
        return depth_meters

    def get_bb_data(self, ego_vehicle):
        vehicles_on_world = self.world.get_actors().filter('vehicle.*')
        walkers_on_world = self.world.get_actors().filter('walker.*')
        bounding_boxes_vehicles, vehicle_ids, vehicle_names, vehicle_distances =\
            ClientSideBoundingBoxes.get_bounding_boxes(vehicles_on_world, self.rgb_camera, ego_vehicle)
        bounding_boxes_walkers, _, _, _ = ClientSideBoundingBoxes.get_bounding_boxes(walkers_on_world, self.rgb_camera, ego_vehicle)
        return [[bounding_boxes_vehicles, bounding_boxes_walkers], vehicle_ids, vehicle_names, vehicle_distances]

    def process_rgb_img(self, img, sensor_width, sensor_height, ego_vehicle):
        img = np.array(img.raw_data)
        img = img.reshape((sensor_height, sensor_width, 4))
        img = img[:, :, :3]  # taking out opacity channel
        bb, ids, names, distances = self.get_bb_data(ego_vehicle)
        return [img, bb, ids, names, distances]

    def remove_sensors(self):
        for sensor in self.sensors_list:
            sensor.destroy()
        self.sensors_list = []

    def begin_data_acquisition(self, sensor_width, sensor_height, fov, frames_to_record_one_ego=1, timestamps=[],
                               egos_to_run=10):
        # Changes the ego vehicle to be put the sensor
        current_ego_recorded_frames = 0
        # These vehicles are not considered because the cameras get occluded without changing their absolute position
        exception_list = ['vehicle.ford.ambulance', 'vehicle.audi.tt', 'vehicle.carlamotors.carlacola',
                          'vehicle.volkswagen.t2', 'vehicle.jeep.wrangler_rubicon', 'vehicle.mercedes.sprinter',
                          'vehicle.micro.microlino', 'vehicle.nissan.patrol', 'vehicle.nissan.patrol_2021',
                          'vehicle.tesla.cybertruck', 'vehicle.volkswagen.t2_2021', 'vehicle.carlamotors.firetruck']
        ego_vehicle = random.choice([x for x in self.world.get_actors().filter("vehicle.*") if x.type_id not in exception_list])
        self.put_rgb_sensor(ego_vehicle, sensor_width, sensor_height, fov)
        self.put_depth_sensor(ego_vehicle, sensor_width, sensor_height, fov)
        self.put_birdview_sensor(width=200, height=200, pixels_per_meter=4)

        # Begin applying the sync mode
        with CarlaSyncMode(self.world, self.rgb_camera, self.depth_camera, fps=30) as sync_mode:
            # Skip initial frames where the car is being put on the ambient
            if self.first_time_simulating:
                for _ in range(30):
                    sync_mode.tick_no_data()

            while True:
                if current_ego_recorded_frames == frames_to_record_one_ego:
                    print('\n')
                    self.remove_sensors()
                    return timestamps
                # Advance the simulation and wait for the data
                # Skip every nth frame for data recording, so that one frame is not that similar to another
                wait_frame_ticks = 0
                while wait_frame_ticks < 5:
                    sync_mode.tick_no_data()
                    wait_frame_ticks += 1

                _, rgb_data, depth_data = sync_mode.tick(timeout=2.0)  # If needed, self.frame can be obtained too
                # Processing raw data
                rgb_array, bounding_box, vehicle_ids_rgb, names, distances = \
                    self.process_rgb_img(rgb_data, sensor_width, sensor_height, ego_vehicle)
                depth_array = self.process_depth_data(depth_data, sensor_width, sensor_height)

                ego_speed = ego_vehicle.get_velocity()
                ego_speed = np.array([ego_speed.x, ego_speed.y, ego_speed.z])

                bounding_box, vehicle_ids_rgb, names, distances = \
                    apply_filters_to_3d_bb(bounding_box, depth_array, sensor_width, sensor_height, vehicle_ids_rgb, names, distances)

                birdview, vehicle_ids_birdview, coords = self.birdview_producer.produce(agent_vehicle=ego_vehicle)
                # birdview_frame = BirdViewProducer.as_rgb(birdview)

                timestamp = round(time.time() * 1000.0)

                if not self.HDF5_file is None:
                # Saving into opened HDF5 dataset file
                    self.HDF5_file.record_data(rgb_array,
                                            depth_array,
                                            bounding_box,
                                            ego_speed,
                                            birdview,
                                            vehicle_ids_rgb,
                                            vehicle_ids_birdview,
                                            coords,
                                            names,
                                            distances,
                                            timestamp)
                else:
                    treat_single_image(rgb_array,
                                       depth_array,
                                       bounding_box[0],
                                       bounding_box[1],
                                       ego_speed,
                                       birdview,
                                       vehicle_ids_rgb,
                                       vehicle_ids_birdview,
                                       coords,
                                       names,
                                       distances,
                                       timestamp)
                current_ego_recorded_frames += 1
                self.total_recorded_frames += 1
                timestamps.append(timestamp)

                sys.stdout.write("\r")
                sys.stdout.write('Frame {0}/{1}'.format(
                    self.total_recorded_frames, frames_to_record_one_ego * egos_to_run * len(self.weather_options)))
                sys.stdout.flush()
