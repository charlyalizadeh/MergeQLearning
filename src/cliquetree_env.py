import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from merge import get_clique_tree, merge_clique, compute_merge_cost

def read_graph(path):
    return np.loadtxt(path, dtype=int, delimiter=",")

def edge_neighbors(adj, i, j):
    neighbors = []
    neighbors.extend([(i, k) for k in adj[i]])
    neighbors.extend([(j, k) for k in adj[j]])
    return neighbors

class CliqueTreeState:
    def __init__(self, graph_path):
        self.graph = read_graph(graph_path)
        self.nxgraph = nx.from_numpy_array(self.graph)
        attributes = {i: str(i) for i in range(self.graph.shape[0])}
        nx.set_node_attributes(self.nxgraph, attributes, name="id")
        self.cliques = list(nx.find_cliques(self.nxgraph))
        self.nxclique_tree, self.clique_tree = get_clique_tree(self.cliques)
        self.edges = [e for e in self.nxclique_tree.edges.data("weight", default=-1)]
        self.edges2index = {f"{e[0]}, {e[1]}":i for i, e in enumerate(self.edges)}
        self.edges2index.update({f"{e[1]}, {e[0]}":i for i, e in enumerate(self.edges)})
        self.edges2edges = {f"{e[0]}, {e[1]}": (e[0], e[1]) for e in self.edges}
        self.edges2edges = {f"{e[1]}, {e[0]}": (e[0], e[1]) for e in self.edges}

    def update(self, i, k):
        clique = list(np.union1d(self.cliques[i], self.cliques[k]))
        self.cliques, self.clique_tree = merge_clique(i, k, clique, self.cliques, self.clique_tree)
        nxclique_tree = nx.from_numpy_array(self.clique_tree)
        self.edges = [e for e in nxclique_tree.edges.data("weight", default=-1)]
        self.nedge = nxclique_tree.number_of_edges()
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
        for e in self.state.edges:
            self.observation.extend([
                len(self.state.cliques[e[0]]),
                len(self.state.cliques[e[1]]),
                e[2]
            ])
        self.observation = np.array(self.observation)
        self.action_space = spaces.Discrete(self.nedge)
        self.action_unvalid = []

    def step(self, action):
        if action in self.action_done:
            return 0, -1, True, False, {}, False
        self.action_unvalid.append(action)
        i, k, w = self.state.edges[action]
        neighbors = edge_neighbors(self.state.nxclique_tree, i, k)

        # Update reward
        reward = compute_merge_cost(self.state.cliques[i], self.state.cliques[k])

        # Update state
        clique = self.state.update(i, k)

        # Update observation
        for n in neighbors:
            n0 = self.state.nxgraph.nodes[int(n[0])]["id"]
            n1 = self.state.nxgraph.nodes[int(n[1])]["id"]
            print(f"(i, k) = ({i}, {k})")
            for k in self.state.edges2edges.keys():
                print(k)
            print(f"({int(n[0])}, {int(n[1])}) => ({n0}, {n1})")


            index = self.state.edges2index[f"{n0}, {n1}"]
            edge = self.state.edges2edges[f"{n0}, {n1}"]
            if edge[0] == n0:
                self.observation[index] = len(clique)
            else:
                self.observation[index + 1] = len(clique)
            self.observation[index + 2] = self.state.clique_tree[edge[0], edge[1]]

        terminated =  len(self.state.cliques) <= self.stop_treshold
        return self.observation, reward, terminated, False, {}, False

    def reset(self, seed, options):
        return self.observation, {}
