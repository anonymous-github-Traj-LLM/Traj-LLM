from src.run_sft import run_sft
from helpers.sim_args import parse_train_args

if __name__ == '__main__':
    args = parse_train_args()
    run_sft(args)