import mass
import os
import topogenesis as tg
import numpy as np
import pandas as pd
import json

class MASS_Env():
    def __init__(self) -> None:
        self.data_path = os.path.relpath('data')

        # Init stencils
        stn_I = tg.create_stencil("von_neumann", 1, 1)
        stn_I.set_index([0,0,0], 0)
        stn_II = tg.create_stencil("von_neumann", 2, clip=1)
        stn_II.set_index([0,0,0], 0)
        self.stencils = {
            "stn_I": stn_I,
            "stn_II": stn_II
            }

        # Load program table
        prgm_path = os.path.join(self.data_path, 'program.csv')
        self.program = pd.read_csv(prgm_path, header=[0, 1, 2], index_col=[0], sep=";",converters={i:eval for i in range(1,100)})

        # Agents names and id
        agents_dict = mass.utilities.df_to_nested_dict(self.program)
        self.agent_names = np.array([a_name for a_name in agents_dict.keys()])

        self.act_space_n = self.agent_names.size * 1 
        self.obs_space = tg.lattice_from_csv(os.path.join(self.data_path, 'voxelized_envelope.csv'))
 
    def reset(self):
        # Load availability lattice
        avail_lattice = tg.lattice_from_csv(os.path.join(self.data_path, 'voxelized_envelope.csv'))

        # Load quality criteria lattices
        env_lattices = {l_name: tg.lattice_from_csv(os.path.join(self.data_path,l_name+'.csv')) for l_name in set(self.program["preferences"].columns.get_level_vals(0))}

        # Init agents/spaces dictionary
        agents_dict = mass.utilities.df_to_nested_dict(self.program)
        for aid, a in enumerate(agents_dict):
            agents_dict[a]["aid"] = aid

        # Init multi agent environment
        self.env = mass.environment(avail_lattice, env_lattices, agents_dict, self.stencils)

        return self.env.occ_lattice

    def step(self, action):
        a_name = self.agent_names[action]
        a = self.env.agents[a_name]

        # Neighbourhood Evaluation
        ##########
        neigh_lat = self.env.avail_lattice * 0.0
        for s, w in zip(
            a.behaviors["neighbourhood"]["stencils"], 
            a.behaviors["neighbourhood"]["weights"]):
            # find and aggregate neighs
            neigh_lat += a.bhv_find_neighbour(s, self.env) * w    

        # extract neigh vals for agent
        neigh_eval_lat = neigh_lat * a.eval_lat


        # Occupation
        ##########
        a.bhv_occupy(self.env, np.unravel_index(neigh_eval_lat.argmax(), neigh_eval_lat.shape))


        # obs, rew, don, inf 
        return (self.env.occ_lattice, a.satisfaction, a.satisfaction>0.95, a.shape)

    def act_space_sample(self):
        return np.random.randint(self.act_space_n)

class MASS_VecEnv():
    def __init__(self, num_envs) -> None:
        self.es = []
        for _ in range(num_envs):
            e = MASS_Env
            self.es.append(e)

        self.act_space_n = self.es[0].act_space_n
        self.obs_space = self.es[0].obs_space

    def step(self, actions):
        obss, rews, dons, infs = [], [], [], []

        for e, act in zip(self.es, actions):
            obs, rew, don, inf = e.step(act)
            obss.append(obs)
            rews.append(rew)
            dons.append(don)
            infs.append(inf)
        
        return obss, rews, dons, infs
    
    def act_space_sample(self):
        return [e.act_space_sample() for e in self.es]