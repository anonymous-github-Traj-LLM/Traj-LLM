import datetime
import re
import torch

def find_id(text:str):
    matches = re.findall(r'\b[a-zA-Z]+\d+\b', text)
    record = list(set(matches))
    return record

def get_current_time():
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return current_time

def remove_number(any_str:str)->str:
    num = take_number(any_str)
    return any_str.split(num)[0]

def take_number(any_str: str)->str:
    ret = re.findall('\d+', any_str)
    if len(ret) == 0:
        ret = ['-1']
    return "".join(ret)

def pick_str_between(content:str, begin="", end=""):
    return content.split(begin)[1].split(end)[0]


def check_str_have_json_params(strObj, jsonObj):
    def get_numbers(data, record):
        if isinstance(data, dict):
            cur_keys = list(data.keys()).copy()
            for k in cur_keys:
                get_numbers(data[k], record)
        elif isinstance(data, list):
            for i in range(len(data)):
                get_numbers(data[i], record)
        elif isinstance(data, int) or isinstance(data, float):
            record.append(data)
    
    jsonNums = []
    get_numbers(jsonObj, jsonNums)
    
    repeat = 0
    for num in jsonNums:
        if str(num) in strObj:
            repeat += 1
    
    return repeat/len(jsonNums)

def get_interaction_move_from_object(summary_orig:str, object_name_orig:str):
    from helpers.hparams import INTERACTION_MOVES
    summary = summary_orig.lower()
    object_name = object_name_orig.lower()
    
    if object_name not in summary:
        return None
    
    summary = summary.split(object_name)[0]
    
    search = [(summary.rfind(mv), mv) for mv in INTERACTION_MOVES] # -1 on failure
    sorted_search = sorted(search, key=lambda x:x[0], reverse=True)
    return sorted_search[0][1] if sorted_search[0][0] != -1 else None

def find_subsequence(tensor:torch.Tensor, subseq:torch.Tensor) -> list:
    if isinstance(subseq, list):
        subseq = torch.tensor(subseq, dtype=tensor.dtype, device=tensor.device)
        
    window_size = subseq.size(0)
    windows = tensor.unfold(0, window_size, 1)
    matches = (windows == subseq).all(dim=1)
    matching_indices = matches.nonzero(as_tuple=False).view(-1)
    return matching_indices.tolist()

def split_subsequence(tensor:torch.Tensor, subseq:torch.Tensor) -> list:
    if isinstance(subseq, list):
        subseq = torch.tensor(subseq, dtype=tensor.dtype, device=tensor.device)
        
    window_size = subseq.size(0)
    windows = tensor.unfold(0, window_size, 1)
    matches = (windows == subseq).all(dim=1)
    matching_indices = matches.nonzero(as_tuple=False).view(-1)
    
    content = tensor.tolist()
    splits = []
    idxs = [-window_size] + matching_indices.tolist()
    for i in range(len(idxs)-1):
        tmp = content[idxs[i]+window_size : idxs[i+1]]
        splits.append(torch.tensor(tmp, dtype=tensor.dtype, device=tensor.device))
    return splits

def search_key_in_dict(d: dict, key: str):
    for k,v in d.items():
        if k == key:
            return v
        if isinstance(v, dict):
            ret = search_key_in_dict(v, key)
            if ret is not None:
                return ret
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    ret = search_key_in_dict(item, key)
                    if ret is not None:
                        return ret
    return None


def print_cuda():
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"\tMemory Allocated: {torch.cuda.memory_allocated(i) / (1024 ** 3):.2f} GB")
            print(f"\tMemory Reserved: {torch.cuda.memory_reserved(i) / (1024 ** 3):.2f} GB")
            print(f"\tMax Memory Allocated: {torch.cuda.max_memory_allocated(i) / (1024 ** 3):.2f} GB")
            print(f"\tMax Memory Reserved: {torch.cuda.max_memory_reserved(i) / (1024 ** 3):.2f} GB")
    else:
        print("No CUDA-capable device is detected")