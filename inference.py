#%%
import os
import torch
from torch import nn
import itertools
from collections import deque
import numpy as np
import random
from utilities.logger import logger
from utilities.mass_wrapper import MASS_Env
from tqdm import tqdm
import msgpack
from utilities.msgpack_numpy import patch as msgpck_numpy_patch
msgpck_numpy_patch()

#%%
def CNN(obs_space, spissitude, final_layer):
    n_input_channels = obs_space.shape[0]

    cnn = nn.Sequential(
        nn.Conv3d(n_input_channels, spissitude[0], kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv3d(spissitude[0], spissitude[1], kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv3d(spissitude[1], spissitude[2], kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten())
    
    # compute shape by doing one forward pass
    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(obs_space.sample()[None]).float()).shape[1]

    out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU())

    return out

class Network(nn.Module):
    def __init__(self, env, device) -> None:
        super().__init__()
        self.num_act = env.act_space_n
        self.device = device

        conv_net = CNN(env.obs_space, spissitude=(32, 64, 64), final_layer=512)

        self.net = nn.Sequential(conv_net, nn.Linear(512,self.num_act))

    def forward(self, x):
        return self.net(x)

    def act(self, obss, epsilon):
        obss_t = torch.as_tensor(obss, dtype=torch.float32, device=self.device)
        q_vals = self(obss_t)

        max_q_indices = torch.argmax(q_vals, dim=1)
        act = max_q_indices.detach().tolist()

        for i in range(len(act)):
            rnd_sample = random.random()
            if rnd_sample <= epsilon:
                act[i] = random.randint(0, self.num_act - 1)

        return act

    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)

        with open(load_path, 'rb') as f:
            params_numpy = msgpack.loads(f.read())
        
        params = {k: torch.as_tensor(v, device=self.device) for k, v in params_numpy.items()}

        self.load_state_dict(params)

#%%
PROJECT_NAME = 'MASS_DQN'
PROJECT_ID = '_0_12'
MODEL_PATH = './models/' + PROJECT_NAME + '/' + PROJECT_ID + '_Model.pack'
PLAYBACK_PATH = './playback/' + PROJECT_NAME + '/' + PROJECT_ID + '_Playback.pack'

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# TODO: add specifications
env = MASS_Env()

net = Network(env, device = device)
net.to(device)

net.load('./models/MASS_DQN/_0_12_Model.pack')

obs = env.reset()
frames = [obs]
for t in itertools.count():

    action = net.act(obs, 0.0)

    obs, rew, done, _ = env.step(action)
    frames.append(obs)

    if done[0]:
        obs = env.reset()

frames_data = msgpack.dumps(frames)
with open(PLAYBACK_PATH, 'wb') as f:
    f.write(frames_data)

# %%