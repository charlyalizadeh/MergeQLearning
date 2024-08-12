import numpy as np
import networkx as nx

def nintersect(c1, c2):
    return sum([c in c2 for c in c1])

def get_clique_tree(cliques):
    clique_graph = nx.Graph()
    for i in range(len(cliques) - 1):
        for j in range(i + 1, len(cliques)):
            weight = nintersect(cliques[i], cliques[j])
            if weight != 0:
                clique_graph.add_edge(i, j, weight=weight)
    clique_tree = nx.maximum_spanning_tree(clique_graph)
    return clique_tree, nx.to_numpy_array(clique_tree, nodelist=list(range(len(cliques))))

def neighbors(adj, v, exclude=[]):
	return [i for i in range(adj.shape[0]) if (adj[v, i] != 0 or adj[i, v] != 0) and not (i in exclude)]

def compute_merge_cost(c1, c2):
    di = len(c1)
    dk = len(c2)
    sik = len(np.intersect1d(c1, c2))
    dik = di + dk - sik
    delta_ik = dik * (2 * dik + 1) - di * (2 * di + 1) - dk * (2 * dk + 1) - sik * (2 * sik + 1)
    return delta_ik

def compute_merge_cost_all(maximal_cliques, clique_tree):
    costs = []
    for i in range(clique_tree.shape[0]-1):
        for k in range(i+1, clique_tree.shape[0]):
            if clique_tree[i, k] != 0:
                costs.append([i, k, compute_merge_cost(maximal_cliques[i], maximal_cliques[k])])
    return costs

def merge_clique(i, k, clique, maximal_cliques, clique_tree):
    _neighbors = neighbors(clique_tree, i, exclude=[i, k])
    _neighbors.extend([n for n in neighbors(clique_tree, k, exclude=[i, k]) if n not in _neighbors])
    _neighbors = np.array([n - (n > i) - (n > k) for n in _neighbors])

    new_maximal_cliques = maximal_cliques.copy()
    del new_maximal_cliques[i]
    del new_maximal_cliques[k]
    maximal_cliques.append(clique)
    clique_tree = np.delete(clique_tree, [i, k], 0)
    clique_tree = np.delete(clique_tree, [i, k], 1)
    clique_tree = np.vstack([clique_tree, np.zeros(clique_tree.shape[0])])
    clique_tree = np.hstack([clique_tree, np.zeros((clique_tree.shape[1] + 1, 1))])
    for n in _neighbors:
        weight = len(np.intersect1d(maximal_cliques[n], maximal_cliques[-1]))
        clique_tree[n, -1] = weight
        clique_tree[-1, n] = weight
    return new_maximal_cliques, clique_tree
