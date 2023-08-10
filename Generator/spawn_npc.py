"""Spawn NPCs into the simulation"""
import glob
import os
import sys
import settings
sys.path.append(settings.CARLA_EGG_PATH)
import carla
import logging
import random


class NPCClass:
    def __init__(self):
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(20.0)

    def create_npcs(self, number_of_vehicles=150, number_of_walkers=70):
        world = self.client.get_world()
        blueprints = world.get_blueprint_library().filter('vehicle.*')
        blueprints = [x for x in blueprints if not x.id.endswith('etron')]
        blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, number_of_vehicles, number_of_spawn_points)
            number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            # Taking out bicycles and motorcycles, since the semantic/bb labeling for that is mixed with pedestrian
            if int(blueprint.get_attribute('number_of_wheels')) > 2:
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                if blueprint.has_attribute('driver_id'):
                    driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                    blueprint.set_attribute('driver_id', driver_id)
                blueprint.set_attribute('role_name', 'autopilot')
                batch.append(SpawnActor(blueprint, transform).then(SetAutopilot(FutureActor, True)))

        for response in self.client.apply_batch_sync(batch):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        # 2. we spawn the walker object
        batch = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invencible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})

        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = world.get_actors(self.all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        world.tick()
        world.wait_for_tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # random max speed
            self.all_actors[i].set_max_speed(1 + random.random()/2)    # max speed between 1 and 2 (default is 1.4 m/s)
        print('Spawned %d vehicles and %d walkers' % (len(self.vehicles_list), len(self.walkers_list)))
        return self.vehicles_list, self.walkers_list

    def remove_npcs(self):
        print('Destroying %d NPC vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

        # stop walker controllers (list is [controler, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        print('Destroying %d NPC walkers' % len(self.walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

