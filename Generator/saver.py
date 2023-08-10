import json

import h5py
import cv2
import numpy as np
import sys

sys.path.append('../../')
CARLA_EGG_PATH = "path/to/carla/egg"
DST_PATH = "generated_data/"
sys.path.append(CARLA_EGG_PATH)
from carla_birdeye_view import BirdViewProducer

def treat_single_image(rgb_data,
                       depth_data,
                       bb_vehicles_data,
                       bb_walkers_data,   
                       ego_speed,
                       birdview_data,
                       vi_rgb,
                       vi_bv,
                       bb_bv,
                       names,
                       distances,
                       timestamp,
                       save_to_many_single_files=True):
    index = str(timestamp)

    # joint vehicles in birview and frontview
    joint_rgb = [idx for idx, id in enumerate(vi_rgb) if id in vi_bv]
    joint_bv = [idx for idx, id in enumerate(vi_bv) if id in vi_rgb]
    rgb_canvas = np.zeros(shape=(200, 200, 3), dtype=np.uint8)
    for instance in joint_bv:
        rgb_canvas = cv2.fillPoly(img=rgb_canvas, pts=np.int32([bb_bv[instance]]), color=(255, 255, 255))
    cv2.imwrite(DST_PATH + 'mask/gmask' + index + '.png', rgb_canvas)


    # # raw rgb
    if save_to_many_single_files:
        cv2.imwrite(DST_PATH + 'raw_pv/raw_img' + index + '.png', rgb_data)

    # birdview
    birdview_frame = BirdViewProducer.as_rgb(birdview_data)
    cv2.imwrite(DST_PATH + "birdview/birdview" + index + ".png", birdview_frame)

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
    rgb_data = np.array(rgb_data)
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
        cv2.imwrite(DST_PATH + 'filtered_pv/filtered_boxed_img' + index + '.png', rgb_data)

    # depth
    depth_data[depth_data==1000] = 0.0
    normalized_depth = cv2.normalize(depth_data, depth_data, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    normalized_depth = np.stack((normalized_depth,)*3, axis=-1)  # Grayscale into 3 channels
    normalized_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_HOT)
    if save_to_many_single_files:
        cv2.imwrite(DST_PATH + 'depth/depth_minmaxnorm' + index + '.png', normalized_depth)
    # return rgb_data, normalized_depth


    # info
    res = []
    vi_bv, vi_rgb = np.array(vi_bv), np.array(vi_rgb)
    bb_bv, bb_vehicles_data = np.array(bb_bv), np.array(bb_vehicles_data)
    vi_bv_list, vi_rgb_list = vi_bv.tolist(), vi_rgb.tolist()
    for id in vi_bv_list:
        if id in vi_rgb_list:
            idx_bv, idx_rgb = vi_bv_list.index(id), vi_rgb_list.index(id)
            res.append({
                "id": id,
                "name": names[idx_rgb],
                'distance': distances[idx_rgb],
                'birdview_coords': bb_bv[idx_bv].tolist(),
                'front_coords': [bb_vehicles_data.tolist()[idx_rgb * 4],
                                 bb_vehicles_data.tolist()[idx_rgb * 4 + 1],
                                 bb_vehicles_data.tolist()[idx_rgb * 4 + 2],
                                 bb_vehicles_data.tolist()[idx_rgb * 4 + 3]]
            })


    with open(DST_PATH + 'info/info' + index + ".json", "w") as final:
        json.dump(res, final, indent=4)
