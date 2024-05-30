from dataclasses import dataclass

IGNORE_INDEX = -100

INSERT_SIGN_MAP = {
    'llama': {
        'embed_sign': '[INSERT]',
        'embed_sign_ids': [29961, 19460, 29962],
        'pos_sign': '[INSERT]',
        'pos_sign_ids': [29961, 19460, 29962],
        
    },
    'qwen': {
        'embed_sign': '[INSERT]',
        'embed_sign_ids': [58, 12698, 60],
        'pos_sign': '[INSERT]',
        'pos_sign_ids': [58, 12698, 60]
    },
    'llama_orig': {
        'embed_sign': '[INSERT]',
        'embed_sign_ids': [29961, 19460, 29962],
        'pos_sign': '[INSERT]',
        'pos_sign_ids': [29961, 19460, 29962],
        
    },
    'qwen_orig': {
        'embed_sign': '[INSERT]',
        'embed_sign_ids': [58, 12698, 60],
        'pos_sign': '[INSERT]',
        'pos_sign_ids': [58, 12698, 60]
    }
}


INTERACTION_MOVES = ['overtake', 'bypass', 'yield']


@dataclass
class SimcopilotConfig:
    MAP_DIM: int = 512
    MOTION_DIM: int = 3072
    INTERACTION_DIM: int = 256
    OBJECT_DIM: int = 128