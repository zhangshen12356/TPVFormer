import numpy as np
import json
from tqdm import tqdm
import os
import math
import pdb


def rotate_point(center, point, angle_degrees):
    """
    Rotate a point around a given center by a given angle in degrees.

    Parameters:
    center (tuple): Center point (x, y).
    point (tuple): Point to rotate (x, y).
    angle_degrees (float): Angle of rotation in degrees.

    Returns:
    tuple: Rotated point (x, y).
    """
    angle_radians = math.radians(angle_degrees)
    cx, cy = center
    px, py = point

    # Translate point to origin
    x = px - cx
    y = py - cy

    # Perform rotation
    xn = x * math.cos(angle_radians) - y * math.sin(angle_radians)
    yn = x * math.sin(angle_radians) + y * math.cos(angle_radians)

    # Translate point back
    rotated_x = xn + cx
    rotated_y = yn + cy

    return rotated_x, rotated_y


def is_point_inside_rectangle(rect_center, rect_width, rect_height, rotation_angle, point):
    """
    Check if a point is inside a rotated rectangle.

    Parameters:
    rect_center (tuple): Center point of the rectangle (x, y).
    rect_width (float): Width of the rectangle.
    rect_height (float): Height of the rectangle.
    rotation_angle (float): Angle of rotation in degrees.
    point (tuple): Point to check (x, y).

    Returns:
    bool: True if the point is inside the rectangle, False otherwise.
    """
    # Calculate the four corners of the rectangle before rotation
    x, y = rect_center
    half_width = rect_width / 2
    half_height = rect_height / 2

    corners = [
        (x - half_width, y + half_height),
        (x + half_width, y + half_height),
        (x + half_width, y - half_height),
        (x - half_width, y - half_height)
    ]

    # Rotate the corners of the rectangle
    rotated_corners = [
        (rotate_point(rect_center, corner, rotation_angle) for corner in corners)]

    # Sort the corners by angle to the center
    rotated_corners.sort(key=lambda c: math.atan2(c[1] - y, c[0] - x))

    # Check if the point is inside the rotated rectangle
    inside = False
    n = len(rotated_corners)
    j = n - 1
    for i in range(n):
        xi, yi = rotated_corners[i]
        xj, yj = rotated_corners[j]

        if ((yi > point[1] != yj > point[1]) and
                (point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi)):
            inside = not inside
        j = i

    return inside


def is_point_in_rotated_rectangle(center_x, center_y, width, height, angle, point_x, point_y):
    # 计算旋转后的矩形四个顶点坐标
    angle_rad = math.radians(angle)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    half_width = width / 2
    half_height = height / 2

    x1 = center_x + half_width * cos_angle - half_height * sin_angle
    y1 = center_y + half_width * sin_angle + half_height * cos_angle

    x2 = center_x - half_width * cos_angle - half_height * sin_angle
    y2 = center_y - half_width * sin_angle + half_height * cos_angle

    x3 = center_x - half_width * cos_angle + half_height * sin_angle
    y3 = center_y - half_width * sin_angle - half_height * cos_angle

    x4 = center_x + half_width * cos_angle + half_height * sin_angle
    y4 = center_y + half_width * sin_angle - half_height * cos_angle

    # 将矩形四个顶点坐标顺时针连接起来构成一个四边形
    polygon = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    # 判断点是否在多边形内
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        if ((polygon[i][1] > point_y) != (polygon[j][1] > point_y)) and (point_x < (polygon[j][0] - polygon[i][0]) * (point_y - polygon[i][1]) / (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
            inside = not inside
        j = i

    return inside


if __name__ == "__main__":
    label_dir_json = "/home/zs/Data/2023-11-13/label"
    lidar_dir = "/home/zs/Data/2023-11-13/lidar_bin"
    # label_dir_json = "/home/zs/Data/2023-11-10/label"
    # lidar_dir = "/home/zs/Data/2023-11-10/lidar_bin"

    label_jsons = os.listdir(label_dir_json)

    for label_json in tqdm(label_jsons):
        label_json_path = os.path.join(label_dir_json, label_json)
        lidar_path = os.path.join(lidar_dir, label_json[:-5]+".bin")
        lidar_datas = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 3)
        lidar_seg_list = []
        with open(label_json_path, 'r', encoding='utf-8') as f:
            json_datas = json.loads(f.read())
            for lidar_data in lidar_datas:
                x = lidar_data[0]
                y = lidar_data[1]
                z = lidar_data[2]
                label_flag = False
                for json_data in json_datas:
                    obj_name = json_data['obj_type']
                    position_x = json_data['psr']['position']['x']
                    position_y = json_data['psr']['position']['y']
                    position_z = json_data['psr']['position']['z']

                    rotation_x = json_data['psr']['rotation']['x']
                    rotation_y = json_data['psr']['rotation']['y']
                    rotation_z = json_data['psr']['rotation']['z']
                    theta = math.degrees(rotation_z)

                    scale_x = json_data['psr']['scale']['x']
                    scale_y = json_data['psr']['scale']['y']
                    scale_z = json_data['psr']['scale']['z']

                    if z > (position_z + scale_z / 2) or z < (position_z - scale_z / 2):
                        continue
                    else:
                        # is_inside = is_point_inside_rectangle((position_x, position_y), scale_x, scale_y, theta, (x, y))
                        is_inside = is_point_in_rotated_rectangle(
                            position_x, position_y, scale_x, scale_y, theta, x, y)
                        if is_inside:
                            if (obj_name == "person" or obj_name == "pedestrian") and label_flag == False:
                                lidar_seg_list.append(1)
                                label_flag = True
                            elif (obj_name == "bicycle" or obj_name == "motorcycle" or obj_name == "rider") and label_flag == False:
                                lidar_seg_list.append(2)
                                label_flag = True
                            elif obj_name == "car" and label_flag == False:
                                lidar_seg_list.append(3)
                                label_flag = True
                            elif obj_name == "bus" and label_flag == False:
                                lidar_seg_list.append(4)
                                label_flag = True
                            elif obj_name == "truck" and label_flag == False:
                                lidar_seg_list.append(5)
                                label_flag = True
                            elif obj_name == "forklift" and label_flag == False:
                                lidar_seg_list.append(6)
                                label_flag = True
                            elif obj_name == "trailer" and label_flag == False:
                                lidar_seg_list.append(7)
                                label_flag = True
                            elif obj_name == "rack" and label_flag == False:
                                lidar_seg_list.append(8)
                                label_flag = True
                            elif obj_name == "shelves" and label_flag == False:
                                lidar_seg_list.append(9)
                                label_flag = True
                            elif obj_name == "traffic_cone" and label_flag == False:
                                lidar_seg_list.append(10)
                                label_flag = True
                            elif (obj_name == "goods" or obj_name == "cargo") and label_flag == False:
                                lidar_seg_list.append(11)
                                label_flag = True
                            elif obj_name == "traffic_light" and label_flag == False:
                                lidar_seg_list.append(12)
                                label_flag = True
                            elif obj_name == "other_vehicle" and label_flag == False:
                                lidar_seg_list.append(13)
                                label_flag = True
                            else:
                                continue

                if label_flag == False:
                    lidar_seg_list.append(0)
        f.close()
        lidar_seg_name = label_json[:-5] + "_seg.bin"
        # save_path = os.path.join("/home/zs/Data/2023-11-10/lidar_seg", lidar_seg_name)
        save_path = os.path.join("/home/zs/Data/2023-11-13/lidar_seg", lidar_seg_name)
        lidar_seg_list = np.array(lidar_seg_list).astype(np.uint8)
        lidar_seg_list.tofile(save_path)



'''
["car", "pedestrian", "person", "bus", "truck", "bicycle", "motorcycle", "rider",
"bicycle_motorcycle", "forklift", "trailer", "rack", "shelves", "traffic_cone",
"goods", "cargo", "traffic_light", "other_vehicle",]

1:(person, pedestrian), 2:(bicycle, motorcycle, rider) 3:(car)  4:(bus) 5:(truck) 6:(forklift) 7:(trailer)
8:(rack) 9:(shelves)  10:(traffic_cone) 11:(goods, cargo) 12:(traffic_light) 13:(other_vehicle)
'''