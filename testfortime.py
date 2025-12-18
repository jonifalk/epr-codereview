"""
This module uses the modules in pathfinder to create a random graph.
This graph has given number of nodes and a random number of cost and distraction values.
It also uses the timeit functionality to measure the time it takes in ms
to run a greedy or recursive algorithm for the given graph.
"""

__author__ = "8819989, Falk, 8746524, Hartmann"
__version__ = "0.3"

# Import Section
import timeit
import random
from pathfinder import path_finder_greedy as greedy
from pathfinder import path_finder_recursive as recursive
from pathfinder import path_finder_dijkstra as dijkstra
from pathfinder import opt_balanced as optimum


# Assuming the main logic is saved as pathfinder.py in the same directory.
# Takes the functions path_finder_greedy and path_finder_recursive
# Optimum linear path taken as logic, can be theoretically switched
# with the other two Optimizations


# Generates a random graph which can then be traversed.
# Change num_nodes for the total number of nodes in the graph.
# Change probability to
def generate_random_graph(num_nodes=15, probability=0.3):
    """
    Generates a directed graph with random weights.
    Nodes are named '0', '1', ... 'num_nodes-1'.
    """
    nodes = [str(i) for i in range(num_nodes)]
    graph = {n: {} for n in nodes}

    for u in nodes:
        for v in nodes:
            if u != v and random.random() < probability:
                # Ensure no immediate loops back, though algorithm handles it
                cost = random.randint(1, 10)
                distraction = random.randint(0, 10)
                graph[u][v] = {'c': cost, 'd': distraction}
    return graph


def run_timing_comparison():
    """
    Creates a random graph with node_count number of nodes.
    The start node is 0.
    To change the size of the graph: Edit the node_count, which is 15 by default.
    Uses timeit.timeit to determine the time for each algorithm to run.
    :Returns:
        time greedy : average greedy
        time recursive : average recursive
    """
    # Setup Data
    # Keep graph small (15 nodes) because Recursive will take way to long.
    # Any Node number over 16 causes the programm to take a very long time.
    node_count = 12
    graph = generate_random_graph(num_nodes=node_count, probability=0.4)
    start_node = '0'
    end_node = str(node_count - 1)

    print(f"\nTiming Analysis (Graph size: {node_count} nodes)")

    # Wrapper for Greedy
    def wrapper_greedy():
        greedy(graph, start_node, end_node, optimum)

    # Wrapper for Dijkstra
    def wrapper_dijkstra():
        dijkstra(graph, start_node, end_node, optimum)

    # Wrapper for Recursive
    def wrapper_recursive():
        recursive(graph, start_node, end_node, optimum)

    # Number of executions
    iterations = 1000

    # Measure Greedy
    # We use timeit.timeit which returns total time for 'number' iterations
    time_greedy = timeit.timeit(wrapper_greedy, number=iterations)
    avg_greedy = (time_greedy / iterations) * 1000      # to ms

    # Measure Dijkstra
    time_dijkstra = timeit.timeit(wrapper_dijkstra, number=iterations)
    avg_dijkstra = (time_dijkstra / iterations) * 1000  # to ms

    # Measure Recursive:
    # Recursive is much slower, so we run fewer iterations to save user time
    rec_iterations = 10
    time_recursive = timeit.timeit(wrapper_recursive, number=rec_iterations)
    avg_recursive = (time_recursive / rec_iterations) * 1000        # to ms

    print(f"\nResults over {iterations} iterations (Greedy) and "
          f"{rec_iterations} iterations (Recursive):")
    print(f"Greedy Average Time:    {avg_greedy:.4f} ms")
    print(f"Dijkstra Average Time:  {avg_dijkstra:.4f} ms")
    print(f"Recursive Average Time: {avg_recursive:.4f} ms")

    if avg_greedy > 0:
        ratio = avg_recursive / avg_greedy
        print(f"\nRecursive was approx {ratio:.1f}x slower than Greedy.")

    # Verification of Path Quality (Single run)
    g_res = greedy(graph, start_node, end_node, optimum)
    d_res = dijkstra(graph, start_node, end_node, optimum)
    r_res = recursive(graph, start_node, end_node, optimum)

    print("\n--- Quality Check (Random Sample) ---")

    # Calculation of the greedy score
    if g_res:
        g_score = optimum(g_res[0], g_res[1])
        print(f"Greedy Path Score:    {g_score} (Cost: {g_res[0]}, Dist: {g_res[1]})")
    else:
        print("Greedy found no path.")

    # Calculation of the Dijkstra score
    if d_res:
        d_score = optimum(d_res[0], d_res[1])
        print(f"Dijkstra Path Score:  {d_score} (Cost: {d_res[0]}, Dist: {d_res[1]})")
    else:
        print("Dijkstra found no path.")

    # Calculation of the recursive score
    if r_res:
        r_score = optimum(r_res[0], r_res[1])
        print(f"Recursive Path Score: {r_score} (Cost: {r_res[0]}, Dist: {r_res[1]})")
    else:
        print("Recursive found no path.")


if __name__ == "__main__":
    run_timing_comparison()
