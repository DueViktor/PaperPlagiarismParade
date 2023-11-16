"""
Exercise description:
Reproduce the main results of the paper "Community structure in social and biological networks". Specifically:
- Implement the edge-betweenness community discovery algorithm without using the corresponding networkx function (but you can use nx's helper functions such as one to calculate edge betwennesses and graph components). The function takes one parameter: the number of communities to return.
- Reproduce Figure 3 by generating graphs with the networkx function planted_partition_graph, setting number of nodes to 128 and communities of 32 nodes each. Vary p_out and p_in so that the average degree of a node is 16, and you can have an average number of edges per node going outside its communities going from 0 to 8. Record the fraction of correctly-classified nodes.
- Reproduce Figure 4b by applying the algorithm to the Zachary Katate networks (which you can get from the networkx function karate_club_graph) and reconsturcting the community dendogram.

"""


from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx


def create_artificial_graph(
    vertices: int = 128,
    num_of_communities: int = 4,
    p_out: float = 0,
    expected_degree: int = 16,
) -> nx.Graph:
    """Create the graph based on section "Computer-Generated Graphs

    Quote:
        Edges were placed between vertex pairs independently at random,
        with probability P_in for vertices belonging to the same community
        and P_out for vertices in different communities, with P_out < P_in.
        The probabilities were chosen so as to keep the average degree z of a vertex equal to 16.
    """
    size_of_each_community = vertices // num_of_communities

    p_in = (expected_degree - (vertices - size_of_each_community) * p_out) / (
        size_of_each_community - 1
    )
    assert p_in > p_out, f"p_in {p_in} must be greater than p_out {p_out}"

    """
    # my attempt to create the graph but the average degree is not 16...
    graph = nx.Graph()
    for node in range(vertices):
        graph.add_node(node, community=node // (vertices // num_of_communities))

    for u in graph.nodes:
        for v in graph.nodes:
            if u == v:
                continue
            if graph.nodes[u]["community"] == graph.nodes[v]["community"]:
                if random.random() < p_in:
                    graph.add_edge(u, v)
            else:
                if random.random() < p_out:
                    graph.add_edge(u, v)
    """

    # boring way to create the graph
    graph = nx.planted_partition_graph(
        l=num_of_communities, k=size_of_each_community, p_in=p_in, p_out=p_out, seed=42
    )

    # add community attribute to each node
    for node in graph.nodes:
        graph.nodes[node]["community"] = node // (vertices // num_of_communities)

    # plotting
    degress = [graph.degree(node) for node in graph.nodes]
    avg_degree = sum(degress) / len(degress)

    # assert that avg_degree is close to expected_degree
    assert (
        abs(avg_degree - expected_degree) < 1
    ), f"Average degree is {avg_degree} but expected is {expected_degree}"

    # # # # # # # #
    # # PLOTTING  #
    # # DEGREE DISTRIBUTION
    # plt.hist(degress)
    # plt.axvline(sum(degress) / len(degress), color="k", linestyle="dashed", linewidth=1)
    # plt.title(f"Average degree: {avg_degree}")
    # plt.savefig(f"assets/average_degree_{p_out}.png")
    # plt.clf()

    # # THE GRAPH
    # plt.figure(figsize=(10, 10))
    # nx.draw(
    #     graph,
    #     pos=nx.spring_layout(graph),
    #     node_color=[graph.nodes[node]["community"] for node in graph.nodes],
    #     node_size=100,
    # )
    # # add title to the graph
    # plt.savefig(f"assets/artificial_graph_{p_out}.png")
    # plt.clf()

    return graph


if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm

    theoretical_p_out_max = 0.083
    print(f"theoretical_p_out_max: {theoretical_p_out_max}")

    # generate 20 values between 0 and theoretical_p_out_max
    p_outs = [round(x, 3) for x in list(np.linspace(0, theoretical_p_out_max, 20))]
    print(f"p_outs: {p_outs}")
    inter_community_degrees = []
    correctly_classified_nodes = []

    for p_out in tqdm(p_outs):
        vertices = 128
        num_of_communities = 4
        expected_degree = 16
        graph = create_artificial_graph(
            vertices, num_of_communities, p_out, expected_degree
        )

        while True:
            edge_betweenness = nx.edge_betweenness_centrality(graph)

            # sort the edge betweenness
            sorted_edge_betweenness = sorted(
                edge_betweenness.items(), key=lambda x: x[1], reverse=True
            )

            # remove the edge with the highest betweenness
            edge_to_be_removed: Tuple = sorted_edge_betweenness[0][0]
            graph.remove_edge(*edge_to_be_removed)

            if nx.number_connected_components(graph) == 4:
                break

        # x-axis: average number of inter-comunity edges per vertex
        # This is not the true average number of inter-comunity edges per vertex but rather the expected value
        avg_inter_community_degree = p_out * (vertices - expected_degree)
        inter_community_degrees.append(avg_inter_community_degree)

        # y-axis: fraction of vertices classified correctly
        # if the community attribute of a node is the same as all its neighbors then it is classified correctly
        fraction_of_correctly_classified_nodes = 0
        for node in graph.nodes:
            neighbor_community_match = [
                graph.nodes[node]["community"] != graph.nodes[neighbor]["community"]
                for neighbor in graph.neighbors(node)
            ]
            if not any(neighbor_community_match):
                fraction_of_correctly_classified_nodes += 1

        fraction_of_correctly_classified_nodes /= len(graph.nodes)
        correctly_classified_nodes.append(fraction_of_correctly_classified_nodes)

    plt.plot(inter_community_degrees, correctly_classified_nodes)
    plt.ylim(0, 1.1)
    plt.xlabel("average number of inter-comunity edges per vertex")
    plt.ylabel("fraction of vertices classified correctly")
    plt.savefig("assets/fig-3.png")
