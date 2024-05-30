import json
import yaml

# ------------------------------------------------------------------------
# read/write normal file formats 
# ------------------------------------------------------------------------
def read_txt(txt_path, errors='ignore'):
    with open(txt_path, 'r', errors=errors) as f:
        content = f.read()
        return content

def write_txt(content:str, file_path:str):
    with open(file_path, "w") as file:
        file.write(content)
    
def read_jsonl(jsonl_path):
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f.readlines():
            json_obj = json.loads(line)
            data.append(json_obj)
    return data

def write_jsonl(jsonl_obj, jsonl_path, mode='w+'):
    with open(jsonl_path, mode) as save_f:
        for i,p in enumerate(jsonl_obj):
            json.dump(p,save_f,ensure_ascii=False)
            if i != len(jsonl_obj) - 1:
                save_f.write('\n')

def read_json(json_path):
    with open(json_path, 'r') as f:
        json_obj = json.loads(f.read())
    return json_obj

def write_json(json_obj, json_path, mode='w+', indent=None):
    with open(json_path, mode) as f:
        json.dump(json_obj, f, indent=indent)
        return True

def read_yaml(yaml_path):
    assert yaml_path.endswith('.yaml')

    with open(yaml_path,'r',encoding='utf-8') as f:
        config = yaml.full_load(f)
    return config


def write_config(cfg, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, encoding='utf-8', allow_unicode=True)
    return True