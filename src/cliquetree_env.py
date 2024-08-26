import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from merge import get_clique_tree, compute_merge_cost, nintersect

def read_graph(path):
    return np.loadtxt(path, dtype=int, delimiter=",")

def edge_neighbors(adj, i, j):
    neighbors = []
    neighbors.extend([(i, k) for k in adj[i]])
    neighbors.extend([(j, k) for k in adj[j]])
    return neighbors

def set_node_id(graph):
    attributes = {i: str(i) for i in range(graph.number_of_nodes())}
    nx.set_node_attributes(graph, attributes, name="id")

def set_edge_id(graph):
    attributes = {i: str(i) for i in range(graph.number_of_edges())}
    nx.set_edge_attributes(graph, attributes, name="id")


class CliqueTreeState:
    def __init__(self, graph_path):
        # Retrieve cliques
        self.graph = read_graph(graph_path)
        self.nxgraph = nx.from_numpy_array(self.graph)
        set_node_id(self.nxgraph)
        self.cliques = list(nx.find_cliques(self.nxgraph))
        self.cliques = {str(i):c for i, c in enumerate(self.cliques)}

        # Setup cliquetree
        self.nxclique_tree = get_clique_tree(self.cliques)

        # Setup edge list (constant)
        self.edges_const = [(e[2]["src"], e[2]["dst"]) for e in self.nxclique_tree.edges(data=True)]
        self.true_id = {str(i):str(i) for i, c in enumerate(self.cliques)}

    def update(self, action):
        i, k = self.edges_const[action]
        clique = list(np.union1d(self.cliques[i], self.cliques[k]))
        if i < k:
            self.cliques[i].extend(self.cliques[k])
            self.true_id[k] = i
            del self.cliques[k]
        else:
            self.cliques[k].extend(self.cliques[i])
            self.true_id[i] = k
            del self.cliques[i]
        return clique




class CliqueTreeEnv(gym.Env):
    def __init__(self, env_config):
        self.nneighbor = env_config["nneighbor"]
        self.nfeature = 1

        self.state = CliqueTreeState(env_config["graph_path"])
        self.nedge = self.state.nxclique_tree.number_of_edges()
        self.stop_treshold = int(0.1 * len(self.state.cliques))
        self.action_done = []

        self.observation_space = spaces.Box(
            low=-1.0,
            high=2**63 - 2,
            shape=(self.nedge * (self.nneighbor + 1) * self.nfeature,),
            dtype=np.int64
        )
        self.observation = []
        for e in self.state.edges_const:
            self.observation.extend([
                len(self.state.cliques[e[0]]),
                len(self.state.cliques[e[1]]),
                nintersect(self.state.cliques[e[0]], self.state.cliques[e[1]])
            ])
        self.observation = np.array(self.observation)
        self.action_space = spaces.Discrete(self.nedge)
        self.action_done = []

    def step(self, action):
        if action in self.action_done:
            return 0, -1, True, False, {}
        self.action_done.append(action)
        i, k = self.state.edges_const[action]

        reward = compute_merge_cost(self.state.cliques[i], self.state.cliques[k])
        self.state.update(action)

        for i, e in enumerate(self.state.edges_const):
            if i in self.action_done:
                self.observation[i * 3] = -1
                self.observation[i * 3 + 1] = -1
                self.observation[i * 3 + 2] = -1
                continue
            c1 = self.state.cliques[self.state.true_id[e[0]]]
            c2 = self.state.cliques[self.state.true_id[e[1]]]
            self.observation[i * 3] = len(c1)
            self.observation[i * 3 + 1] = len(c2)
            self.observation[i * 3 + 2] = nintersect(c1, c2)

        terminated = len(self.state.cliques) <= self.stop_treshold
        return self.observation, reward, terminated, False, {}

    def reset(self, seed, options):
        return self.observation, {}
