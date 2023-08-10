import os
import h5py
import cv2
import numpy as np
import pandas as pd
import math

import tensorflow as tf


def read_hdf5_file(path, dst_path):
    df = pd.DataFrame(columns=['id', 'img_path', 'front_coords', 'birdview_coords_x', 'birdview_coords_y'])
    index = 0

    for file_name in os.listdir(path):
        with h5py.File(os.path.join(path, file_name), 'r') as file:
            rgb = file['rgb']
            bb_vehicles = file['bounding_box']['vehicles']
            vehicle_ids_rgb = file['vehicle_ids_rgb']
            vehicle_ids_birdview = file['vehicle_ids_birdview']
            bb_bv = file['bb_bv']
            timestamps = file['timestamps']

            for time in timestamps['timestamps']:
                rgb_data = np.array(rgb[str(time)])
                bb_vehicles_data = np.array(bb_vehicles[str(time)])
                id_rgb_data = np.array(vehicle_ids_rgb[str(time)])
                id_bv_data = np.array(vehicle_ids_birdview[str(time)])
                bb_bv_data = np.array(bb_bv[str(time)])

                for rgb, bb_rgb, id_rgb, id_bv, bb_bv in zip(rgb_data, bb_vehicles_data, id_rgb_data, id_bv_data, bb_bv_data):
                    index_str = str(index)
                    index += 1

                    # Joint vehicles in birdview and frontview
                    joint_rgb = [idx for idx, id in enumerate(id_rgb) if id in id_bv]
                    joint_bv = [idx for idx, id in enumerate(id_bv) if id in id_rgb]

                    # Saving raw rgb image
                    path_img = os.path.join(dst_path, 'raw_img' + index_str + '.png')
                    cv2.imwrite(path_img, rgb)

                    id_bv_list, id_rgb_list = id_bv.tolist(), id_rgb.tolist()
                    for id in id_bv_list:
                        if id in id_rgb_list:
                            idx_bv, idx_rgb = id_bv_list.index(id), id_rgb_list.index(id)
                            pts = bb_bv[idx_bv].tolist()
                            record = {
                                'id': id,
                                'img_path': path_img,
                                'birdview_coords_x': np.array([min([pt[0] for pt in pts]),
                                                             max([pt[0] for pt in pts])]),
                                'birdview_coords_y': np.array([min([pt[1] for pt in pts]),
                                                             max([pt[1] for pt in pts])]),
                                'front_coords': np.array([bb_rgb.tolist()[idx_rgb * 4] / 1024,
                                                         bb_rgb.tolist()[idx_rgb * 4 + 1] / 768,
                                                         bb_rgb.tolist()[idx_rgb * 4 + 2] / 1024,
                                                         bb_rgb.tolist()[idx_rgb * 4 + 3] / 768])
                            }
                            df = df.append(record, ignore_index=True)
    return df

def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_width = max(0, yi2 - yi1)
    inter_height = max(0, xi2 - xi1)
    inter_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def compute_centroid_distance(point1, point2):
    center_x_gt, center_y_gt = (point1[2] - point1[0]) / 2, (point1[3] - point1[1]) / 2
    center_x_pred, center_y_pred = (point2[2] - point2[0]) / 2, (point2[3] - point2[1]) / 2
    return math.sqrt((center_x_gt - center_x_pred) ** 2 + (center_y_gt - center_y_pred) ** 2)

def evaluate_metrics(model, test_df):
    sum_iou = 0
    sum_CD = 0
    sum_h = 0
    sum_w = 0
    sum_ar = 0

    for index in range(len(test_df)):
        img_path = test_df.iloc[index]['img_path']
        bv_coords_x = tf.convert_to_tensor(test_df.iloc[index]['birdview_coords_x'], dtype=tf.float32)
        bv_coords_y = tf.convert_to_tensor(test_df.iloc[index]['birdview_coords_y'], dtype=tf.float32)
        front_coords = tf.convert_to_tensor(test_df.iloc[index]['front_coords'], dtype=tf.float32)
        img = cv2.imread(img_path) / 255.0
        img = cv2.resize(img, (224, 224))
        img = tf.convert_to_tensor(img, dtype=tf.float32)

        result = model.predict([tf.convert_to_tensor([front_coords], dtype=tf.float32),
                                tf.convert_to_tensor([img], dtype=tf.float32)])
        result_x = result[0][0]
        result_y = result[1][0]

        bv_coords_x = bv_coords_x.numpy()
        bv_coords_y = bv_coords_y.numpy()

        x1g, y1g, x2g, y2g = bv_coords_x[0], bv_coords_y[0], bv_coords_x[1], bv_coords_y[1]
        x1, y1, x2, y2 = result_x[0], result_y[0], result_x[1], result_y[1]

        iou = compute_iou((x1g, y1g, x2g, y2g), (x1, y1, x2, y2))
        sum_iou += iou

        CD = compute_centroid_distance((x1g, y1g, x2g, y2g), (x1, y1, x2, y2))
        sum_CD += CD

        h_gt, h_pred = y2g - y1g, y2 - y1
        sum_h += h_pred / h_gt

        w_gt, w_pred = x2g - x1g, x2 - x1
        sum_w += w_pred / w_gt

        sum_ar += abs(w_pred / h_pred - w_gt / h_gt)

    num_samples = len(test_df)
    avg_iou = sum_iou / num_samples
    avg_CD = sum_CD / num_samples
    avg_h = sum_h / num_samples
    avg_w = sum_w / num_samples
    avg_ar = sum_ar / num_samples

    print("\niou:", avg_iou)
    print("CD:", avg_CD)
    print("hE:", avg_h)
    print("wE:", avg_w)
    print("arE:", avg_ar)
    