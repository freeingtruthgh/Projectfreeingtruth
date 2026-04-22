import sys
import os
import glob
import getopt
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from offlinerl import RL


class WalkerDataset(Dataset):
    def __init__(self, path):
        data = np.load(path)

        self.observations = torch.tensor(data["observations"], dtype=torch.float32)
        self.actions = torch.tensor(data["actions"], dtype=torch.float32)
        self.rewards = torch.tensor(data["rewards"], dtype=torch.float32)
        self.next_observations = torch.tensor(data["next_observations"], dtype=torch.float32)
        self.terminals = torch.tensor(data["terminals"], dtype=torch.float32)

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_observations[idx],
            self.terminals[idx],
        )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Optional but good for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(argv):

    epochs = 50
    device = "cuda"
    batch_size = 256
    linear_expectile = False
    step_expectile = False
    expectile = 0.7
    expectile_start = 0.6
    expectile_end = 0.8
    expectile_step_size = 5
    seed = 42
    tag = "Fixed"

    try:
        opts, args = getopt.getopt(argv, "h:e:D:b:l:P:E:s:n:g:S:t:", ["epochs=", "device=", "batch_size=", "linear_expectile=", "step_expectile=", "expectile=", "expectile_start=", "expectile_end=", "expectile_step_size=", "seed=", "tag="])

    except getopt.GetoptError:
        print('Check options by typing:\n{} -h'.format(__file__))
        sys.exit(2)

    print("OPTS: {}".format(opts))
    for opt, arg in opts:
        if opt == '-h':
            print('\n{} [OPTIONS]'.format(__file__))
            print('\t -h, --help\t\t Get help')
            print('\t -e, --epochs\t\t Number of training epochs')
            print('\t -D, --device\t\t Device (cpu, cuda)')
            print('\t -b, --batch_size\t\t Batch size for training')
            print('\t -l, --linear_expectile\t\t Enable linear expectile scheduling, default is False')
            print('\t -P, --step_expectile\t\t Enable step expectile scheduling, default is False')
            print('\t -E, --expectile\t\t Expectile (0,1) for asymmetric value loss; higher values emphasize high-return actions')
            print('\t -s, --expectile_start\t Starting expectile (e.g., 0.6)')
            print('\t -n, --expectile_end\t Final expectile (e.g., 0.8)')
            print('\t -g, --expectile_step_size\t\t Step size for the expectile to update (e.g., 5) to update every 5 epochs')
            print('\t -S, --seed\t\t random seed')
            print('\t -t, --tag\t\t keyword for filename')
            sys.exit()
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-D", "--device"):
            device = arg
        elif opt in ("-b", "--batch_size"):
            batch_size = int(arg)
        elif opt in ("-l", "--linear_expectile"):
            linear_expectile = arg.lower() == "true"
        elif opt in ("-P", "--step_expectile"):
            step_expectile = arg.lower() == "true"
        elif opt in ("-E", "--expectile"):
            expectile = float(arg)
        elif opt in ("-s", "--expectile_start"):
            expectile_start = float(arg)
        elif opt in ("-n", "--expectile_end"):
            expectile_end = float(arg)
        elif opt in ("-g", "--expectile_step_size"):
            expectile_step_size = int(arg)
        elif opt in ("-S", "--seed"):
            seed = int(arg)
        elif opt in ("-t", "--tag"):
            tag = arg

    BASE_DATA_DIR = "/home/tbl0009/data"
    set_seed(seed)

    train_path = os.path.join(BASE_DATA_DIR, "walker2d_train.npz")
    val_path   = os.path.join(BASE_DATA_DIR, "walker2d_val.npz")

    # check
    for path in [train_path, val_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset file: {path}")

    print(f"Loading dataset from {train_path}")
    train_dataset = WalkerDataset(train_path)
    print(f"Loading dataset from {val_path}")
    val_dataset = WalkerDataset(val_path)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    observations, actions, rewards, next_observations, terminals = next(iter(train_loader))
    
    state_dim = observations.shape[1]
    action_dim = actions.shape[1]

    tag = f"{tag}_seed{seed}"

    model = RL.IQLNet(state_dim, action_dim, device=device, expectile=expectile, linear_expectile=linear_expectile, step_expectile=step_expectile, expectile_start=expectile_start, expectile_end=expectile_end, expectile_step_size=expectile_step_size)

    trainer = RL.Trainer(model, train_loader, val_loader, epochs, device, tag=tag)

    trainer.train()
    trainer.evaluation()
    trainer.save_results()


if __name__ == "__main__":
    main(sys.argv[1:])