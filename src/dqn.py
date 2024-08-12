from ray.rllib.algorithms.dqn.dqn import DQNConfig, DQN
from cliquetree_env import CliqueTreeEnv

config = DQNConfig()
replay_config = {
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 50000,
        "prioritized_replay_alpha": 0.5,
        "prioritized_replay_beta": 0.5,
        "prioritized_replay_eps": 3e-6,
    }

config = config.training(replay_buffer_config=replay_config)
config = config.resources(num_gpus=0)
config = config.env_runners(num_env_runners=1)
config = config.environment(env=CliqueTreeEnv)
config["env_config"] = {"nneighbor": 2, "graph_path": "case300_molzahn.txt"}

algo = DQN(config=config)
algo.train()
del algo
