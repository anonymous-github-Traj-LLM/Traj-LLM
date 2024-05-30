import os
from helpers.io_helper import read_yaml, write_config

def jud_running_env(meta:dict, entire:dict=None):
    if 'local_3090' in meta.keys() and 'cluster_A800' in meta.keys():
        f = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader; nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits")
        info = f.readlines()
        f.close()
        print(info)
        if len(info) == 0:
            raise Exception("nvidia-smi is not available, please check your environment!")
        occup = int(info[-1].strip())
        print(f"Currently GPU memory size: {occup}.")
        if occup < 30000:
            print(f"Using local_3090, value: {meta['local_3090']}")
            return meta['local_3090'], True
        else:
            print(f"Using cluster_A800, value: {meta['cluster_A800']}")
            return meta['cluster_A800'], True
    return meta, False

def jud_model_type(meta:dict, entire:dict=None):
    """
    XXX:
        qwen: x
        llama: x
    """
    if 'qwen' in meta.keys() and 'llama' in meta.keys():
        if 'qwen' in entire['base_model'].lower():
            print(f"Currently using Qwen model, value: {meta['qwen']}")
            return meta['qwen'], True
        else:
            print(f"Currently using Llama model, value: {meta['llama']}")
            return meta['llama'], True
    return meta, False
    

class ConfigHelper:
    KEYWORD_TO_INHERIT_CONFIG = "inherited_from"
    multi_choose_func = [jud_running_env, jud_model_type]
    
    @staticmethod
    def combine_dict(d1:dict, d2:dict):
        """
        d1 has higher priority
        """
        
        if d2 is None:
            return d1
        
        for k in d2.keys():
            if k not in d1.keys():
                d1[k] = d2[k]
            else:
                if isinstance(d1[k], dict):
                    d1[k] = ConfigHelper.combine_dict(d1[k], d2[k])
                else:
                    pass
        # inherit
        if ConfigHelper.KEYWORD_TO_INHERIT_CONFIG in d2.keys():
            d1[ConfigHelper.KEYWORD_TO_INHERIT_CONFIG] = d2[ConfigHelper.KEYWORD_TO_INHERIT_CONFIG]
        elif ConfigHelper.KEYWORD_TO_INHERIT_CONFIG in d1.keys():
            d1.pop(ConfigHelper.KEYWORD_TO_INHERIT_CONFIG)
        
        return d1
    
    @staticmethod
    def read_config_with_inheritence(config_path:str):
        seen = [] # avoid endless loop
        cfg = {}
        while True:
            jud = [True if os.path.samefile(cp, config_path) else False for cp in seen]
            if True in jud:
                break
            else:
                seen.append(config_path)

            cur_cfg = read_yaml(config_path)
            cfg = ConfigHelper.combine_dict(cfg, cur_cfg)
            
            if ConfigHelper.KEYWORD_TO_INHERIT_CONFIG not in cfg.keys():
                break
            else:
                dirname = os.path.dirname(config_path)
                config_path = os.path.join(dirname, cfg[ConfigHelper.KEYWORD_TO_INHERIT_CONFIG])
        return cfg
    
    @staticmethod
    def save_config(cfg, file_path):
        if write_config(cfg, file_path):
            print(f'Config saved to {file_path}')
        print(f'Config saved to {file_path} failed')
    
    @staticmethod
    def simplify_config(cfg:dict, entire:dict):
        """
        choose one from multis
        
        not support list containing dict
        """
        for i,v in cfg.items():
            for func in ConfigHelper.multi_choose_func:
                if isinstance(v, dict):
                    v, ret = func(v, entire)
                    if ret:
                        break
            if isinstance(v, dict):
                cfg[i] = ConfigHelper.simplify_config(v, entire)
            else:
                cfg[i] = v

        return cfg
    
    @staticmethod
    def transform_config(cfg:dict):
        """
        transform config to args
        """
        from helpers.sim_args import SimArguments
        return SimArguments(**cfg)
        
    @staticmethod
    def load_config(cfg_path:str):
        """
        support inheritance, but only for dict, not for list
        """
        assert cfg_path.endswith('.yaml') or cfg_path.endswith('.yml')
        
        cfg = ConfigHelper.read_config_with_inheritence(cfg_path)
        simply_cfg = ConfigHelper.simplify_config(cfg, cfg.copy())
        args = ConfigHelper.transform_config(simply_cfg)
        return args