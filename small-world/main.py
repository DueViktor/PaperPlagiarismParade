import copy
import random
from math import log as ln
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm


def create_graph(n: int, k: int) -> nx.Graph:
    # validate n and k based on the paper
    assert n > k > ln(n) > 1

    # We start with a ring of n vertices, ...
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # ... each connected to its k nearest neighbours by undirected edges.
    for i in range(n):
        for j in range(i + 1, i + k // 2 + 1):
            G.add_edge(i, j % n)

    # minor extension assert that the degree of all nodes are equal to k
    assert all(G.degree[node] == k for node in G.nodes)

    return G


def rewire_with_probability(G: nx.Graph, p: float) -> nx.Graph:
    """Rewiring can be done in many ways. Based on figure 1 it seems
    they are only rewiring one of the endpoints of the edge. I'll do the same"""

    # assert valid probability
    assert p >= 0 and p <= 1, "p must be between 0 and 1, but was {}".format(p)

    # create a copy of the graph
    graph: nx.Graph = copy.deepcopy(G)

    def validate_new_edge(u: int, v: int, graph: nx.Graph) -> bool:
        """Validate that the new edge is not a duplicate self-loop"""
        return u != v and not graph.has_edge(u, v)

    # For each edge (u, v) in the graph, rewire the edge with probability p
    for u, v in graph.edges:
        rewire: bool = p > random.random()

        if rewire:
            while True:
                new_v: int = random.choice(list(graph.nodes))
                if validate_new_edge(u, new_v, graph):
                    break

            graph.remove_edge(u, v)
            graph.add_edge(u, new_v)

    return graph


def figure_1():
    G: nx.Graph = create_graph(20, 4)

    # generate a list of values with 0.1 increments
    ps = np.arange(0, 1.1, 0.25)
    fig, ax = plt.subplots(1, len(ps), figsize=(15, 5))

    for i, p in enumerate(ps):
        rewired: nx.Graph = rewire_with_probability(G, p)
        nx.draw(
            rewired,
            pos=nx.circular_layout(rewired),
            with_labels=True,
            connectionstyle="arc3,rad=0.1",
            arrows=True,
            ax=ax[i],
        )

        ax[i].set_title(f"p = {p}")

    plt.tight_layout()
    plt.savefig("assets/figure_1.png")
    plt.clf()

    return


def figure_2():
    def get_metrics(G: nx.Graph) -> Tuple[float, float]:
        L_g = nx.average_shortest_path_length(G)
        C_g = nx.average_clustering(G)
        return L_g, C_g

    ps = np.logspace(-4, 0, num=14)

    G: nx.Graph = create_graph(1000, 10)

    L_0, C_0 = get_metrics(G)

    Ls, Cs = [], []
    for p in tqdm(ps):
        Ls_p = []
        Cs_p = []
        for _ in range(20):
            G_rewired = rewire_with_probability(G, p)
            L_g, C_g = get_metrics(G_rewired)
            Ls_p.append(L_g / L_0)
            Cs_p.append(C_g / C_0)

        Ls.append(sum(Ls_p) / len(Ls_p))
        Cs.append(sum(Cs_p) / len(Cs_p))

    plt.scatter(ps, Ls, label="L(p)/L(0)")
    plt.scatter(ps, Cs, label="C(p)/C(0)")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/figure_2.png")
    plt.clf()


if __name__ == "__main__":
    figure_1()
    figure_2()
