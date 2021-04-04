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
from msgpack_numpy import patch as msgpck_numpy_patch
msgpck_numpy_patch()

#%%
GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = int(1e6)
MIN_REPLAY_SIZE = 50000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = int(1e6)
TARGET_UPDATE_FREQ = 10000
NUM_ENVS = 4
LR = 2.5e-4
SAVE_INTERVAL = 10000
LOG_INTERVAL = 1000 
PROJECT_NAME = 'MASS_DQN'
PROJECT_ID = '_0_12'
SAVE_PATH = './' + PROJECT_NAME + '/' + PROJECT_ID + '_Model.pack'


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
        self.num_actions = env.act_space.n
        self.device = device

        conv_net = CNN(env.obs_space, spissitude=(32, 64, 64), final_layer=512)

        self.net = nn.Sequential(conv_net, nn.Linear(512,self.num_actions))

    def forward(self, x):
        return self.net(x)

    def act(self, obses, epsilon):
        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        q_values = self(obses_t)

        max_q_indices = torch.argmax(q_values, dim=1)
        actions = max_q_indices.detach().tolist()

        for i in range(len(actions)):
            rnd_sample = random.random()
            if rnd_sample <= epsilon:
                actions[i] = random.randint(0, self.num_actions - 1)

        return actions
    
    def compute_loss(self, transitions, target_network):

        obses = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = np.asarray([t[4] for t in transitions])

        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device).unsqueeze(-1)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device=self.device).unsqueeze(-1)
 
        # Compute Targets
        target_q_values = target_network(new_obses_t)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

        targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

        # Compute Loss
        q_values = self(obses_t)

        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets) 

        return loss

    def save(self, save_path):
        params = {k: t.detach().cpu().numpy() for k, t in self.state_dict().items()}
        params_data = msgpack.dumps(params)

        os.mkdir(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(params_data)

    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)

        with open(load_path, 'rb') as f:
            params_numpy = msgpack.loads(f.read())
        
        params = {k: torch.as_tensor(v, device=self.device) for k, v in params_numpy.items()}

        self.load_state_dict(params)


#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# TODO: add specifications
env = MASS_Env()

replay_buffer = deque(maxlen=BUFFER_SIZE)
epinfos_buffer = deque([], maxlen=100)

episode_count = 0.0

# Initialize Networks
online_network = Network(env, device = device)
target_network = Network(env, device = device)
online_network.to(device)
target_network.to(device)
target_network.load_state_dict(online_network.state_dict())

optimizer = torch.optim.Adam(online_network.parameters(), lr=LR)

# Initialize the Replay Buffer
obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.act_space.sample()

    new_obs, rew, done, _ = env.step(action)

    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)

    obses = new_obs

#%%
# Main training loop
obses = env.reset()
log = logger(PROJECT_NAME, PROJECT_ID, ['WB', 'LO'])
tq = tqdm()
for step in itertools.count():
    # tq.update(1)
    epsilon = np.interp(step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    
    # TODO: convert to singluar step
    actions = online_network.act(obses, epsilon)

    new_obses, rews, dones, infos = env.step(actions)

    for obs, action, rew, done, new_obs, info in zip(obses, actions, rews, dones, new_obses, infos):
        transition = (obs, action, rew, done, new_obs)
        replay_buffer.append(transition)

        if done:
            epinfos_buffer.append(info['episode'])
            episode_count += 1

    obses = new_obses

    # Start Gradient Step
    transitions = random.sample(replay_buffer, BATCH_SIZE)
    loss = online_network.compute_loss(transitions, target_network)

    # Gradient Descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update target network
    if step % TARGET_UPDATE_FREQ ==0 :
        target_network.load_state_dict(online_network.state_dict())

    # Logging
    if step % LOG_INTERVAL == 0:
        rew_mean = np.mean([e['r'] for e in epinfos_buffer]) or 0
        len_mean = np.mean([e['l'] for e in epinfos_buffer]) or 0

        # TODO: Update tags
        log_dict = {
            'AvgRew': rew_mean,
            'AvgEpLen': len_mean,
            'Episodes': episode_count
        }

        log.update(log_dict, step)

    # Saving
    if step % SAVE_INTERVAL == 0 and step != 0:
        # print('Saving...')
        online_network.save(SAVE_PATH)

# %%
