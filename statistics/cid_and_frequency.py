import os
import sys
import json
import pickle
import glob
import multiprocessing
import numpy as np
from tqdm import tqdm
from auto_behavior_detection import evaluate_behavior


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class DataOrganizer:
    def __init__(self):
        self.frequency = 10
        pass

    def load_data(self, path):
        raise NotImplementedError
    
    def organize_2_object(self, data):
        for pidx in data['predict_idx']:
            cura = data['traj'][pidx]
            for i in range(len(data['traj'])):
                if i==pidx:
                    continue
                curb = data['traj'][i]
                yield cura, curb

class WaymoData(DataOrganizer):
    def __init__(self):
        super().__init__()
        self.frequency = 10

    def load_data(self, path):
        sample = load_pkl(path)
        all_id = sample['track_infos']['object_id']
        all_type = sample['track_infos']['object_type']
        all_vehicle_index = np.where(np.array(all_type)=='TYPE_VEHICLE')[0]
        all_trajs = sample['track_infos']['trajs']
        track_index = sample['tracks_to_predict']['track_index']
        track_type = sample['tracks_to_predict']['object_type']

        track_veh_index = np.where(np.array(track_type)=='TYPE_VEHICLE')[0]
        track_veh_index = np.array(track_index)[track_veh_index]
        track_veh_id = np.array(all_id)[track_veh_index]

        valid_trajs = []
        valid_ids = []
        predict_idx = []
        for i,traj in enumerate(all_trajs):
            if not np.all(traj[:,-1]):
                continue
            valid_ids.append(all_id[i])
            valid_trajs.append(traj)
            if all_id[i] in track_veh_id:
                predict_idx.append(len(valid_ids)-1)

        orig_traj = np.array(valid_trajs)
        xyh = orig_traj[:,:,[0,1,6]]
        vxvy = orig_traj[:,:,[7,8]]
        v = np.sqrt(vxvy[:,:,0]**2 + vxvy[:,:,1]**2)
        xyhv = np.concatenate([xyh, v[:,:,np.newaxis]], axis=2)
        
        ret = {
            'traj': xyhv,
            'id': valid_ids,
            'predict_idx': predict_idx
        }
        return ret

class ArgoverseData(DataOrganizer):
    def __init__(self):
        super().__init__()
        self.frequency = 10

    def load_data(self, path):
        sample = load_pkl(path)
        all_positions = np.array(sample['agent']['position'])
        all_headings = np.array(sample['agent']['heading'])
        valid_mask = np.array(sample['agent']['valid_mask'])
        all_velocity = np.array(sample['agent']['velocity'])

        predict_mask = np.array(sample['agent']['predict_mask'])
        xy = []
        h = []
        v = []
        predict_idx = []
        for i in range(len(all_positions)):
            if not np.all(valid_mask[i][1:]):
                continue
            
            xy.append(all_positions[i][1:,:2])
            h.append(all_headings[i][1:])
            vxvy = all_velocity[i][1:,:2]
            curv = np.sqrt(vxvy[:,0]**2 + vxvy[:,1]**2)
            v.append(curv)
            
            cur_predict = predict_mask[i][1:]
            if np.any(cur_predict):
                predict_idx.append(len(xy)-1)

        xy = np.array(xy)
        h = np.array(h)
        v = np.array(v)
        xyhv = np.concatenate([xy, h[:,:,np.newaxis], v[:,:,np.newaxis]], axis=2)
        ret = {
            'traj': xyhv,
            'predict_idx': predict_idx
        }
        return ret

class L2TData(DataOrganizer):
    def __init__(self):
        super().__init__()
        self.frequency = 20

def cal_interaction_types_and_distances_from_one_scenario(sample_path:str, dataset:DataOrganizer=None):
    ret = {
        "interaction_type": [],
        "interaction_distance": []
    }

    data = dataset.load_data(sample_path) # dict
    for a,b in dataset.organize_2_object(data):
        frame_distance = np.sqrt(np.sum((a[:,:2]-b[:,:2])**2, axis=1))
        min_frame_distance = np.min(frame_distance)
        if min_frame_distance > 50:
            continue
        behavior = evaluate_behavior(a,b, frequency=dataset.frequency)

        ret['interaction_distance'].append(float(min_frame_distance))
        ret['interaction_type'].append(behavior)

    return ret

def parse_result(result_file):
    with open(result_file, 'r') as f:
        result = json.load(f)
    
    types = []
    distances = []
    for g in result:
        types.extend(g['interaction_type'])
        distances.extend(g['interaction_distance'])

    type_cls, type_num = np.unique(types, return_counts=True)
    type_tot_num = np.sum(type_num)
    type_freq = type_num / type_tot_num
    min_dist, mean_dist, mid_dist = np.min(distances), np.mean(distances), np.median(distances)
    print(f"Interaction Type: {type_cls}")
    print(f"Interaction Type Frequency: {type_freq}")
    print(f"Min CID: {min_dist}")
    print(f"Mean CID: {mean_dist}")
    print(f"Median CID: {mid_dist}")

if __name__ == '__main__':
    Debug = False
    Command = True

    root_path = {}
    root_path['waymo'] = '../waymo_data'
    root_path['argo'] = '../argo_data'
    root_path['waymo_interactive'] = '../waymo_interactive_data'
    root_path['l2t'] = '../l2t_data'

    ds_cls = {}
    ds_cls['waymo'] = WaymoData
    ds_cls['argo'] = ArgoverseData
    ds_cls['waymo_interactive'] = WaymoData
    ds_cls['l2t'] = WaymoData

    if Command:
        dataset_name = sys.argv[1]
        sample_gap = int(sys.argv[2])
        save_path = sys.argv[3]
        workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
        Debug = False
        ds = ds_cls[dataset_name]()
    else:
        dataset_name = 'waymo'
        sample_gap = 100
        save_path = 'output/waymo_statistics.json'
        Debug = True
        ds = WaymoData()

    samples_file_paths = glob.glob(os.path.join(root_path[dataset_name], '*.pkl'))
    samples_file_paths = samples_file_paths[::sample_gap]
    print(f"Total samples: {len(samples_file_paths)}")

    if Debug:
        ret = cal_interaction_types_and_distances_from_one_scenario(samples_file_paths[0], dataset=ds)
    else:
        import functools
        func = functools.partial(cal_interaction_types_and_distances_from_one_scenario, dataset=ds)


        with multiprocessing.Pool(workers) as p:
            summon_ret = list(tqdm(p.imap(func, samples_file_paths), total=len(samples_file_paths)))

        with open(save_path, 'w') as f:
            json.dump(summon_ret, f, indent=4)

        # parse_result(save_path)