import numpy as np
import json
import difflib
import re

def take_number(any_str: str)->str:
    ret = re.findall('\d+', any_str)
    if len(ret) == 0:
        ret = ['-1']
    return "".join(ret)

def remove_number(any_str:str)->str:
    num = take_number(any_str)
    return any_str.split(num)[0]


def get_key_val(data, prev, record):
    if isinstance(data, dict):
        cur_keys = list(data.keys()).copy()
        for k in cur_keys:
            get_key_val(data[k], prev+'/'+k, record)
        pass
    elif isinstance(data, list):
        for i in range(len(data)):
            get_key_val(data[i], prev+'/'+str(i), record)
            # record[prev+'/'+str(i)] = data[i]
    elif isinstance(data, int) or isinstance(data, float):
        record[prev] = data
    elif isinstance(data, str):
        if data == 'STRAIGHT':
            record[prev] = 0
        elif data == 'LEFT':
            record[prev] = -10
        elif data == 'RIGHT':
            record[prev] = 10
        else:
            record[prev] = -100
        
def replace_root_objID(input_list:list)->list:
    key_list = input_list.copy()
    for i in range(len(key_list)):
        ki = key_list[i]
        kml = ki.split('/')
        kml[0] = remove_number(kml[0])
        key_list[i] = '/'.join(kml)
    return key_list

def get_JsonIoU(gt_json, pred_json, strict=True):
    gt_list = {}
    get_key_val(gt_json, "", gt_list)

    pred_list = {}
    get_key_val(pred_json, "", pred_list)

    gt_keys = list(gt_list.keys())
    pred_keys = list(pred_list.keys())
    
    if not strict:
        # remove number in objID
        gt_keys = replace_root_objID(gt_keys)
        pred_keys = replace_root_objID(pred_keys)
        
    sum = len(gt_keys) + len(pred_keys)
    same = 0
    for gtk in gt_keys:
        if gtk in pred_keys:
            same+=1
            pred_keys.remove(gtk)
    return same*2/sum

def get_ListIoU(gt_list, pred_list):
    sum = len(gt_list) + len(pred_list)
    same = 0
    for gtk in gt_list:
        if gtk in pred_list:
            same+=1
            pred_list.remove(gtk)
    return same*2/sum

def read_from_list_str(list_str:str):
    obj_list = list_str.strip().split("\'")
    ret = []
    for i in obj_list:
        if i.isalnum():
            ret.append(i)
    return ret


def get_StrDiff(gt_str, pred_str):
    if isinstance(gt_str, dict):
        gt_str = json.dumps(gt_str)
    if isinstance(pred_str, dict):
        pred_str = json.dumps(pred_str)
    
    return difflib.SequenceMatcher(None, gt_str, pred_str).quick_ratio()

def min_max_normalize(vector):
    min_val = np.min(vector)
    max_val = np.max(vector)
    normalized_vector = (vector - min_val) / (max_val - min_val)
    return normalized_vector

def is_close_to_zero(num, threshold=1e-9):
    return abs(num) <= threshold

def normalize_difference(a, b):

    return abs(a-b)

def get_Dist(gt_json, pred_json):
    gt_keys = {}
    get_key_val(gt_json, "", gt_keys)
    pred_keys = {}
    get_key_val(pred_json, "", pred_keys)
    
    tot_keys = {}
    tot_keys.update(pred_keys)
    tot_keys.update(gt_keys)
    
    sum_dif = 0
    DIFF = 10
    for i in tot_keys.keys():
        v = tot_keys[i]
        gt_val = gt_keys[i] if i in gt_keys else v - DIFF
        pred_val = pred_keys[i] if i in pred_keys else v - DIFF
        normal_diff = normalize_difference(gt_val, pred_val)
        sum_dif += normal_diff
    return sum_dif/len(tot_keys)
