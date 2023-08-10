import json

import h5py
import cv2
import numpy as np
import sys

sys.path.append('../../')
CARLA_EGG_PATH = "/home/abtin/Desktop/projects/carla/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg"
DST_PATH = "generated_data/"
sys.path.append(CARLA_EGG_PATH)
from carla_birdeye_view import BirdViewProducer


def read_hdf5_test(hdf5_file):
    with h5py.File(hdf5_file, 'r') as file:
        rgb = file['rgb']
        bb_vehicles = file['bounding_box']['vehicles']
        bb_walkers = file['bounding_box']['walkers']
        depth = file['depth']
        birdview = file['birdview']
        vehicle_ids_rgb = file['vehicle_ids_rgb']
        vehicle_ids_birdview = file['vehicle_ids_birdview']
        bb_bv = file['bb_bv']
        names = file['names']
        distances = file['distances']
        timestamps = file['timestamps']

        rgb_data, bb_vehicles_data, bb_walkers_data, depth_data, birdview_data, vi_rgb_data, vi_birdview_data,\
        bb_bv_data, names_data, distances_data = [], [], [], [], [], [], [], [], [], []

        for time in timestamps['timestamps']:
            rgb_data.append(np.array(rgb[str(time)]))
            bb_vehicles_data.append(np.array(bb_vehicles[str(time)]))
            bb_walkers_data.append(np.array(bb_walkers[str(time)]))
            depth_data.append(np.array(depth[str(time)]))
            birdview_data.append(np.array(birdview[str(time)]))
            vi_rgb_data.append(np.array(vehicle_ids_rgb[str(time)]))
            vi_birdview_data.append(np.array(vehicle_ids_birdview[str(time)]))
            bb_bv_data.append(np.array(bb_bv[str(time)]))
            names_data.append(np.array(names[str(time)]))
            distances_data.append(np.array(distances[str(time)]))

        return rgb_data, bb_vehicles_data, bb_walkers_data, depth_data, birdview_data, vi_rgb_data, vi_birdview_data, bb_bv_data, names_data, distances_data

def treat_single_image(rgb_data,
                       bb_vehicles_data,
                       bb_walkers_data,
                       depth_data,
                       birdview_data,
                       vi_rgb,
                       vi_bv,
                       bb_bv,
                       names,
                       distances,
                       index,
                       save_to_many_single_files=False):
    index = str(index)

    # joint vehicles in birview and frontview
    joint_rgb = [idx for idx, id in enumerate(vi_rgb) if id in vi_bv]
    joint_bv = [idx for idx, id in enumerate(vi_bv) if id in vi_rgb]
    rgb_canvas = np.zeros(shape=(200, 200, 3), dtype=np.uint8)
    for instance in bb_bv[joint_bv]:
        rgb_canvas = cv2.fillPoly(img=rgb_canvas, pts=np.int32([instance]), color=(255, 255, 255))
    cv2.imwrite(DST_PATH + 'gmask' + index + '.png', rgb_canvas)


    # # raw rgb
    # if save_to_many_single_files:
    #     cv2.imwrite(DST_PATH + 'raw_img' + index + '.png', rgb_data)

    # birdview
    birdview_frame = BirdViewProducer.as_rgb(birdview_data)
    cv2.imwrite(DST_PATH + "birdview" + index + ".png", birdview_frame)

    # # birdview_mask
    # rgb_canvas = np.zeros(shape=(200, 200, 3), dtype=np.uint8)
    # nonzero_indices = lambda arr: arr == 1
    # rgb_canvas[nonzero_indices(birdview_data[:, :, 3])] = (255, 255, 255)
    # rgb_canvas[nonzero_indices(birdview_data[:, :, 4])] = (128, 128, 128)
    # cv2.imwrite(DST_PATH + "birdview_mask" + index + ".png", rgb_canvas)

    # bb
    bb_vehicles = []
    for idx in joint_rgb:
        bb_vehicles.append(bb_vehicles_data[idx * 4])
        bb_vehicles.append(bb_vehicles_data[idx * 4 + 1])
        bb_vehicles.append(bb_vehicles_data[idx * 4 + 2])
        bb_vehicles.append(bb_vehicles_data[idx * 4 + 3])


    bb_vehicles = np.array(bb_vehicles)
    bb_walkers = bb_walkers_data
    if all(bb_vehicles != -1):
        for bb_idx in range(0, len(bb_vehicles), 4):
            coordinate_min = (int(bb_vehicles[0 + bb_idx]), int(bb_vehicles[1 + bb_idx]))
            coordinate_max = (int(bb_vehicles[2 + bb_idx]), int(bb_vehicles[3 + bb_idx]))
            cv2.rectangle(rgb_data, coordinate_min, coordinate_max, (0, 255, 0), 1)
    if all(bb_walkers != -1):
        for bb_idx in range(0, len(bb_walkers), 4):
            coordinate_min = (int(bb_walkers[0 + bb_idx]), int(bb_walkers[1 + bb_idx]))
            coordinate_max = (int(bb_walkers[2 + bb_idx]), int(bb_walkers[3 + bb_idx]))

            cv2.rectangle(rgb_data, coordinate_min, coordinate_max, (0, 0, 255), 1)
    if save_to_many_single_files:
        cv2.imwrite(DST_PATH + 'filtered_boxed_img' + index + '.png', rgb_data)

    # # depth
    # depth_data[depth_data==1000] = 0.0
    # normalized_depth = cv2.normalize(depth_data, depth_data, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # normalized_depth = np.stack((normalized_depth,)*3, axis=-1)  # Grayscale into 3 channels
    # normalized_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_HOT)
    # if save_to_many_single_files:
    #     cv2.imwrite(DST_PATH + 'depth_minmaxnorm' + index + '.png', normalized_depth)
    # return rgb_data, normalized_depth


    # info
    res = []
    vi_bv_list, vi_rgb_list = vi_bv.tolist(), vi_rgb.tolist()
    for id in vi_bv_list:
        if id in vi_rgb_list:
            idx_bv, idx_rgb = vi_bv_list.index(id), vi_rgb_list.index(id)
            res.append({
                "id": id,
                "name": names[idx_rgb].decode("utf-8"),
                'distance': distances[idx_rgb],
                'birdview_coords': bb_bv[idx_bv].tolist(),
                'front_coords': [bb_vehicles_data.tolist()[idx_rgb * 4],
                                 bb_vehicles_data.tolist()[idx_rgb * 4 + 1],
                                 bb_vehicles_data.tolist()[idx_rgb * 4 + 2],
                                 bb_vehicles_data.tolist()[idx_rgb * 4 + 3]]
            })


    with open(DST_PATH + 'info' + index + ".json", "w") as final:
        json.dump(res, final, indent=4)


def create_video_sample(hdf5_file, show_depth=True):
    with h5py.File(hdf5_file, 'r') as file:
        frame_width = file.attrs['sensor_width']
        frame_height = file.attrs['sensor_height']
        if show_depth:
            frame_width = frame_width * 2
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (frame_width, frame_height))

        for time_idx, time in enumerate(file['timestamps']['timestamps']):
            rgb_data = np.array(file['rgb'][str(time)])
            bb_vehicles_data = np.array(file['bounding_box']['vehicles'][str(time)])
            bb_walkers_data = np.array(file['bounding_box']['walkers'][str(time)])
            depth_data = np.array(file['depth'][str(time)])

            sys.stdout.write("\r")
            sys.stdout.write('Recording video. Frame {0}/{1}'.format(time_idx, len(file['timestamps']['timestamps'])))
            sys.stdout.flush()
            rgb_frame, depth_frame = treat_single_image(rgb_data, bb_vehicles_data, bb_walkers_data, depth_data)
            if show_depth:
                composed_frame = np.hstack((rgb_frame, depth_frame))
            else:
                composed_frame = rgb_frame                
            cv2.putText(composed_frame, 'timestamp', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(composed_frame, str(time), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(composed_frame)

    print('\nDone.')


if __name__ == "__main__":
    path = "../../data/test.hdf5"
    rgb_data, bb_data_vehicles, bb_data_walkers, depth_data, birdview_data, vi_rgb_data, vi_birdview_data, bb_bv_data,\
    names_data, distances_data = read_hdf5_test(path)
    print("length data: ", len(rgb_data))
    for index, (rgb, bb_v, bb_w, depth, birdview, vi_rgb, vi_bv, bb_bv, names, distances) in enumerate(zip(rgb_data, bb_data_vehicles, bb_data_walkers, depth_data, birdview_data, vi_rgb_data, vi_birdview_data, bb_bv_data, names_data, distances_data)):
        treat_single_image(rgb, bb_v, bb_w, depth, birdview, vi_rgb, vi_bv, bb_bv, names, distances, index, save_to_many_single_files=True)
    # create_video_sample(path, show_depth=False)
