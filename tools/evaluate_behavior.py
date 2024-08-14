import os
import json
import numpy as np
import shapely.geometry
from matplotlib import pyplot as plt
from scenic.core.regions import PolylineRegion
from scipy.signal import butter, lfilter
from itertools import groupby

COLOR_PALLETE = [
    [0, 0.4470, 0.7410],
    [0.8500, 0.3250, 0.0980],
    [0.9290, 0.6940, 0.1250],
    [0.4940, 0.1840, 0.5560],
    [0.4660, 0.6740, 0.1880],
    [0.3010, 0.7450, 0.9330],
    [0.6350, 0.0780, 0.1840]
]

DT = 0.03

def get_trajectory(result, centerline):
    trajectory = {}
    frame0 = result["message"]["frame0"]
    for car_id, car_info in frame0.items():
        trajectory.update({
            car_id: {
                "agenttype": car_info["agenttype"],
                "position": [],
                "project": [],
                "speed": [],
                "heading": [],
                "polyline": None
            }
        })
    for frame, value in result["message"].items():
        for car_id, point_info in value.items():
            pos = [point_info["position_x"], point_info["position_y"]]
            trajectory[car_id]["position"].append(pos)
            dist = centerline.lineString.project(shapely.geometry.Point(pos))
            trajectory[car_id]["project"].append(dist)
            trajectory[car_id]["speed"].append(point_info["speed"])
            trajectory[car_id]["heading"].append(point_info["heading"])

    car_i = 0
    for car_id, value in trajectory.items():
        if all([item == value["position"][0] for item in value["position"]]):
            value["position"][0] = [value["position"][0][0]+1e-3, value["position"][0][1]+1e-3]
        value["polyline"] = PolylineRegion(value["position"])

        import matplotlib.pyplot as plt
        plt.ion()
        value["polyline"].show(plt, color=COLOR_PALLETE[car_i])
        if car_i == 0:
            plt.scatter(value["position"][0][0], value["position"][0][1], color=COLOR_PALLETE[car_i], marker="*", label="start", s=100)
        else:
            plt.scatter(value["position"][0][0], value["position"][0][1], color=COLOR_PALLETE[car_i], marker="*", s=100)
        car_i += 1
    return trajectory

EGO_BEFORE_NPC = 1
EGO_NEAR_NPC = 0
EGO_AFTER_NPC = -1
def get_action(ego_frame, npc_frame):
    bound = 5.5
    if ego_frame < npc_frame - bound*2:
        return EGO_AFTER_NPC
    elif ego_frame > npc_frame + bound:
        return EGO_BEFORE_NPC
    else:
        return EGO_NEAR_NPC

def action_frame_filter(action_frame):
    return action_frame

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_speed_avg(position, start, end):
    speed = []
    dt = DT
    if start <= 0:
        start = 1
    if end >= len(position)-1:
        end = len(position)-2
    for index in range(start, end+1):
        pos_prev = position[index-1]
        pos_curr = position[index]
        pos_next = position[index+1]
        spd_prev_x = (pos_curr[0]-pos_prev[0])/dt
        spd_prev_y = (pos_curr[1]-pos_prev[1])/dt
        spd_next_x = (pos_next[0]-pos_next[0])/dt
        spd_next_y = (pos_next[1]-pos_next[1])/dt
        spd_avg_x = (spd_prev_x + spd_next_x) / 2
        spd_avg_y = (spd_prev_y + spd_next_y) / 2
        spd_avg = np.sqrt(spd_avg_x**2 + spd_avg_y**2)
        speed.append(spd_avg)
    return np.mean(speed)

def over_detect(frame_start, frame_done, begin=0):
    detect_range = 50
    return slice(max(begin, frame_start-detect_range), frame_done+1)

def get_speed_min(speed_list, start, end):
    return np.min(speed_list[over_detect(start, end)])

def get_speed_min_except_begin(speed_list, start, end):
    return np.min(speed_list[over_detect(start, end, begin=25)])

def get_speed_max(speed_list, start, end):
    return np.max(speed_list[over_detect(start, end)])

def get_acceleration_min(acceleration_list, start, end):
    return np.min(acceleration_list[over_detect(start, end)])

def get_acceleration_max(acceleration_list, start, end):
    return np.max(acceleration_list[over_detect(start, end)])

ACTION_START = 0
ACTION_DURING = 1
ACTION_DONE = 2
def get_action_all(trajectory, ego_car_id, dir_path=None, speed_debug=True):
    action_result = {}
    ego_traj = trajectory[ego_car_id]
    ego_filtered_speed = butter_lowpass_filter(ego_traj["speed"], 0.8, 1/DT, 2)
    ego_accelerations = np.diff(ego_filtered_speed) / DT
    ego_accelerations = np.append(ego_accelerations, ego_accelerations[-1])
    ego_traj.update({
        "filtered_speed": ego_filtered_speed,
        "acceleration": ego_accelerations
    })
    
    if speed_debug:
        max_acceleration, min_acceleration = np.max(ego_accelerations), np.min(ego_accelerations)
        max_acceleration_index, min_acceleration_index = np.argmax(ego_accelerations), np.argmin(ego_accelerations)
        plt.figure(100)
        plt.plot(range(len(ego_traj["filtered_speed"])), ego_traj["filtered_speed"], label="ego speed")
        plt.plot(range(len(ego_traj["filtered_speed"])), ego_accelerations, label="acceleration")
        plt.legend()

    for npc_id, npc_traj in trajectory.items():
        if npc_id == ego_car_id:
            continue
        action = {
            "proj_pair": [],
            "frame": [],
            "record": []
        }
        if npc_traj["agenttype"] == "AgentType.PEDESTRIAN":
            npc_traj["speed"] = []
            for index in range(len(npc_traj["position"])):
                npc_traj["speed"].append(get_speed_avg(npc_traj["position"], index-2, index+2))

        for ego_frame, npc_frame in zip(ego_traj["project"], npc_traj["project"]):
            action["proj_pair"].append((ego_frame, npc_frame))
            action["frame"].append(get_action(ego_frame, npc_frame))

        filtered_speed = butter_lowpass_filter(npc_traj["speed"], 0.8, 1/DT, 2)
        accelerations = np.diff(filtered_speed) / DT
        accelerations = np.append(accelerations, accelerations[-1])
        npc_traj.update({
            "filtered_speed": filtered_speed,
            "acceleration": accelerations
        })

        action["frame"] = action_frame_filter(action["frame"])
        frame_group = []
        
        for key, group in groupby(enumerate(action["frame"]), lambda x: x[1]):
            frame_group.append((key, [item[0] for item in group]))
        frame_group_used = np.array([False for _ in range(len(frame_group))])
        for i in range(len(frame_group)-2):
            if frame_group_used[i]:
                continue
            if frame_group[i][0] == EGO_BEFORE_NPC and frame_group[i+1][0] == EGO_NEAR_NPC and frame_group[i+2][0] == EGO_AFTER_NPC:
                print("action before -> after", i)
                frame_start = frame_group[i+1][1][0]
                frame_done = frame_group[i+1][1][-1]
                for j in range(i, i+3):
                    frame_group_used[j] = True
            elif frame_group[i][0] == EGO_AFTER_NPC and frame_group[i+1][0] == EGO_NEAR_NPC and frame_group[i+2][0] == EGO_BEFORE_NPC:
                print("action after -> before ", i)
                frame_start = frame_group[i+1][1][0]
                frame_done = frame_group[i+1][1][-1]
                for j in range(i, i+3):
                    frame_group_used[j] = True
            elif frame_group[i][0] == EGO_BEFORE_NPC and frame_group[i+1][0] == EGO_NEAR_NPC:
                print("action before -> near", i)
                frame_start = frame_group[i+1][1][0]
                frame_done = frame_group[i+1][1][-1]
                for j in range(i, i+2):
                    frame_group_used[j] = True
            elif frame_group[i][0] == EGO_AFTER_NPC and frame_group[i+1][0] == EGO_NEAR_NPC:
                print("action after -> near", i)
                frame_start = frame_group[i+1][1][0]
                frame_done = frame_group[i+1][1][-1]
                for j in range(i, i+2):
                    frame_group_used[j] = True
            else:
                continue
            if speed_debug:
                plt.plot(range(len(npc_traj["filtered_speed"]))[over_detect(frame_start,frame_done)], npc_traj["filtered_speed"][over_detect(frame_start, frame_done)], label="npc speed")
                plt.legend()
            
            npc_speed_avg = get_speed_avg(npc_traj["position"], frame_start, frame_done)
            ego_filtered_speed_avg = np.mean(ego_traj["filtered_speed"][over_detect(frame_start, frame_done)])
            npc_filtered_speed_avg = np.mean(npc_traj["filtered_speed"][over_detect(frame_start, frame_done)])

            if abs(npc_speed_avg) < 1e-1:
                action_type = "bypass"
            elif get_speed_min_except_begin(ego_traj["filtered_speed"], frame_start, frame_done) < 1e-1:
                action_type = "yield"
            elif ego_filtered_speed_avg < npc_filtered_speed_avg:
                action_type = "yield"
            elif get_acceleration_min(ego_traj["acceleration"], frame_start, frame_done) < -10:
                action_type = "yield"
            else:
                action_type = "overtake"
            if action_type == "bypass":
                npc_speed_all = np.mean(npc_traj["speed"])
                if npc_speed_all > 0.3:
                    action_type = "overtake"
            action["record"].append({
                "frame_start": frame_start,
                "frame_done": frame_done,
                "slice": action["frame"][frame_start:frame_done+1],
                "traj": ego_traj["position"][frame_start:frame_done+1],
                "traj_npc": npc_traj["position"][frame_start:frame_done+1],
                "type": action_type
            })
        action_result.update({
            npc_id: action
        })
    if speed_debug:
        if dir_path:
            file_path = os.path.join(dir_path, "speed.png")
        else:
            file_path = "speed.png"
        plt.savefig(file_path)
        plt.figure(1)
    return action_result


def judge_action(dir_path, use_gt=False):
    if use_gt:
        centerline_path = os.path.join(dir_path, "result_gt_centerline.json")
        result_path = os.path.join(dir_path, "result_gt_evaluate.json")
    else:
        centerline_path = os.path.join(dir_path, "result_centerline.json")
        result_path = os.path.join(dir_path, "result_evaluate.json")
    with open(centerline_path, "r") as f:
        centerline = PolylineRegion(points=json.load(f)["centerline"])
    with open(result_path, "r") as f:
        result = json.load(f)

    frame0 = result["message"]["frame0"]
    for key, value in frame0.items():
        if value["agenttype"] == "AgentType.EGO":
            ego_car = value.copy()
            ego_car.update({
                "id": key
            })
            break
    npc_car = frame0.copy()
    del npc_car[ego_car["id"]]

    trajectory = get_trajectory(result, centerline)

    action_result = get_action_all(trajectory, ego_car["id"], dir_path=dir_path)

    plt.ion()
    plt.gca().set_aspect('equal')

    x, y = centerline.lineString.xy
    plt.plot(x, y, label="centerline", linestyle="--")

    car_i = 1
    for car_id, value in npc_car.items():
        action = action_result[car_id]
        car_type = npc_car[car_id]["agenttype"]
        for record in action["record"]:
            traj = record["traj"]
            xx = [pos[0] for pos in traj]
            yy = [pos[1] for pos in traj]
            plt.plot(xx, yy, linewidth=5, color=COLOR_PALLETE[car_i])
            traj_npc = record["traj_npc"]
            xx = [pos[0] for pos in traj_npc]
            yy = [pos[1] for pos in traj_npc]
            plt.plot(xx, yy, label="{} {} {}".format(record["type"], car_type, car_id), linewidth=5, color=COLOR_PALLETE[car_i])

        car_i += 1
    plt.legend()

    return action_result, trajectory
