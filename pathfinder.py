"""
Module that contains 3 Optimizations that each use a different approach to
give a way to quantify the optimum path through the graph given a cost and distraction value.
Contains functions for greedy and recursive path movement.
"""

__author__ = "8819989, Falk, 8746524, Hartmann"
__version__ = "0.3"

import heapq as hq


def opt_balanced_linear(cost, distraction):
    """
    Optimization 1: Balanced Linear Combination.
    Minimizes (Cost - Distraction).
    We want low cost and high distraction.

    Args:
        cost (int): The cost value of the edge.
        distraction (int): The distraction value of the edge.
    
    Returns:
        int: The computed score for the edge.

    >>> opt_balanced_linear(10, 5)
    5
    >>> opt_balanced_linear(5, 10)
    -5
    >>> opt_balanced_linear(10, 10)
    0
    """
    return cost - distraction


def opt_distraction_focused(cost, distraction):
    """
    Optimization 2: Weighted Linear (Distraction Priority).
    Heavily favors distraction (weight 2.0) over cost (weight 1.0).
    Formula: Cost - (2 * Distraction)

    Args:
        cost (int): The cost value of the edge.
        distraction (int): The distraction value of the edge.
    
    Returns:
        int: The computed score for the edge.

    >>> opt_distraction_focused(10, 5)
    0
    >>> opt_distraction_focused(10, 2)
    6
    >>> opt_distraction_focused(5, 10)
    -15
    """
    return cost - (2 * distraction)


def opt_efficiency_ratio(cost, distraction):
    """
    Optimization 3: Cost Efficiency.
    Minimizes Cost per unit of Distraction.
    Formula: Cost / (Distraction + 0.001) to avoid division by zero.

    Args:
        cost (int): The cost value of the edge.
        distraction (int): The distraction value of the edge.
    
    Returns:
        int: The computed score for the edge.

    >>> round(opt_efficiency_ratio(10, 2), 2)
    5.0
    >>> round(opt_efficiency_ratio(10, 5), 2)
    2.0
    >>> opt_efficiency_ratio(10, -1) # Negative test for logic (though graph values usually +)
    -10.01001001001001
    """
    # Small epsilon to prevent division by zero
    epsilon = 0.001
    return cost / (distraction + epsilon)

def opt_min_cost(cost, distraction):
    """
    Optimization 4: Minimize Cost.
    Focuses solely on minimizing cost.

    Args:
        cost (int): The cost value of the edge.
        distraction (int): The distraction value of the edge.
    
    Returns:
        int: The computed score for the edge.

    >>> opt_min_cost(10, 5)
    10
    >>> opt_min_cost(5, 10)
    5
    """
    return cost + 0 * distraction  # distraction has no effect

def opt_min_distraction(cost, distraction):
    """
    Optimization 5: Minimize Distraction.
    Focuses solely on minimizing distraction.

    Args:
        cost (int): The cost value of the edge.
        distraction (int): The distraction value of the edge.
    
    Returns:
        int: The computed score for the edge.

    >>> opt_min_distraction(10, 5)
    5
    >>> opt_min_distraction(5, 10)
    10
    """
    return distraction + 0 * cost  # cost has no effect

def opt_balanced(cost, distraction):
    """
    Optimization 6: Balanced Sum.
    Minimizes the sum and distraction equally.

    Args:
        cost (int): The cost value of the edge.
        distraction (int): The distraction value of the edge.
    
    Returns:
        int: The computed score for the edge.

    >>> opt_balanced(10, 5)
    7.5
    >>> opt_balanced(5, 10)
    7.5
    """
    return (cost + distraction) / 2

# Algorithms

def path_finder_dijkstra(graph, start, end, opt_func):
    """
    Dijkstra path finder with two edge weights: c, d
    Uses a priority queue to explore the graph based on cumulative scores.
    Considers the total path score using the provided optimization function.

    Args:
        graph (dict): The graph represented as an adjacency list with weights.
        start (str): The starting node.
        end (str): The destination node.
        opt_func (function): The optimization function to calculate edge scores.

    Returns:
        tuple: (total_effort, total_distraction, path_list) or None if no path found.

    >>> graph = {
    ...     'S': {'A': {'c': 1, 'd': 1}, 'B': {'c': 10, 'd': 100}},
    ...     'A': {'E': {'c': 1, 'd': 1}},
    ...     'B': {'E': {'c': 1, 'd': 1}},
    ...     'E': {}
    ... }

    # Greedy using balanced linear (1-1=0 vs 10-100=-90).
    # Logic: 1-1=0, 10-100=-90. -90 is lower, so greedy picks B.
    >>> path_finder_dijkstra(graph, 'S', 'E', opt_balanced_linear)
    (2, 2, ['S', 'A', 'E'])

    # Negative Test: Dead end
    >>> graph_trap = {'S': {'A': {'c': 1, 'd': 1}}, 'A': {}, 'E': {}}
    >>> path_finder_dijkstra(graph_trap, 'S', 'E', opt_balanced_linear) is None
    True

    # Test: Already at destination
    >>> path_finder_dijkstra(graph, 'S', 'S', opt_balanced_linear)
    (0, 0, ['S'])
    """
    queue = [(0, start)]  # (score, node, total_cost, total_distraction)
    hq.heapify(queue)

    scores = {node: float('inf') for node in graph}
    previous = {node: None for node in graph}

    scores[start] = 0

    while queue:
        current_score, current_node = hq.heappop(queue)

        # Break when we reach the end node
        if current_node == end:
            break

        # Skip if we have already found a better path
        if current_score > scores[current_node]:
            continue

        for neighbor, weights in graph[current_node].items():
            # Get edge weights to neighbor
            c = weights['c']
            d = weights['d']

            # Calculate new score using the optimization function
            edge_score = opt_func(c, d)
            new_score = current_score + edge_score

            # Dijksta cannot handle negative weights
            if new_score < 0:
                break # Results in invalid path if negative weights occur

            # Update if better path found
            if new_score < scores[neighbor]:
                scores[neighbor] = new_score
                previous[neighbor] = current_node
                # Add the new total score and neighbor to the priority queue
                # Total score is first element, because heapq gets
                # sorted by the first element of a tuple
                hq.heappush(queue, (new_score, neighbor))

    # Reconstruct path
    path = []
    visited = set()
    total_cost = 0
    total_distraction = 0

    current = end
    while current is not None and current not in visited:
        visited.add(current)
        path.append(current)
        prev = previous[current]
        if prev is not None:
            edge = graph[prev][current]
            total_cost += edge['c']
            total_distraction += edge['d']
        current = prev

    path.reverse()

    # Because path reconstruction always adds the end node,
    # we need to check if we actually reached the start node.
    if path[0] != start:
        return None # No path found

    return total_cost, total_distraction, path


def path_finder_greedy(graph, start, end, opt_func):
    """
    Approaches the destination by choosing the locally optimal edge at each step.
    Note: Does not backtrack and may fail to find the destination or a global optimum.

    Args:
        graph (dict): The graph represented as an adjacency list with weights.
        start (str): The starting node.
        end (str): The destination node.
        opt_func (function): The optimization function to calculate edge scores.

    Returns:
        tuple: (total_effort, total_distraction, path_list) or None if no path found.

    >>> graph = {
    ...     'S': {'A': {'c': 1, 'd': 1}, 'B': {'c': 10, 'd': 100}},
    ...     'A': {'E': {'c': 1, 'd': 1}},
    ...     'B': {'E': {'c': 1, 'd': 1}},
    ...     'E': {}
    ... }

    # Greedy using balanced linear (1-1=0 vs 10-100=-90).
    # Logic: 1-1=0, 10-100=-90. -90 is lower, so greedy picks B.
    >>> path_finder_greedy(graph, 'S', 'E', opt_balanced_linear)
    (11, 101, ['S', 'B', 'E'])

    # Negative Test: Dead end
    >>> graph_trap = {'S': {'A': {'c': 1, 'd': 1}}, 'A': {}, 'E': {}}
    >>> path_finder_greedy(graph_trap, 'S', 'E', opt_balanced_linear) is None
    True

    # Test: Already at destination
    >>> path_finder_greedy(graph, 'S', 'S', opt_balanced_linear)
    (0, 0, ['S'])
    """
    # Edge case: If we are already at the destination, cost/distraction is 0.
    if start == end:
        return 0, 0, [start]

    current_node = start
    path = [current_node]       # Keeps track of where we have been: ['S', 'B'...]
    total_cost = 0              # Accumulator for Effort
    total_distraction = 0       # Accumulator for Distraction

    # Loop Prevention:
    # We maintain a 'visited' set. If we didn't, the cat might go A -> B -> A -> B forever
    # if that edge was the "cheapest" one.
    visited = {current_node}

    # Main Loop:
    # Keep moving until we hit the 'end' node.
    while current_node != end:
        # Get all neighbors of the current location
        neighbors = graph.get(current_node, {})

        # Filter out visited neighbors to prevent loops
        valid_neighbors = [n for n in neighbors if n not in visited]

        # Dead end: If there are no new places to go, and we aren't at the house, we failed.
        if not valid_neighbors:
            # Dead end and we haven't reached the goal
            return None

        # Greedy Choice
        # Determine the best local step based on opt_func
        best_next_node = None
        best_score = float('inf')       # Start with infinity so the first real score is lower
        best_edge_data = None

        for neighbor in valid_neighbors:
            edge = neighbors[neighbor]
            # Score Calculation:
            # We apply the math function (e.g., Cost - Distraction)
            # ONLY to this specific edge (local view).
            score = opt_func(edge['c'], edge['d'])

            # If this edge is better than the previous best one we saw...
            if score < best_score:
                best_score = score
                best_next_node = neighbor
                best_edge_data = edge

        # Moving:
        # We commit to the best node we found. Rule for the algo:
        # We cannot "undo" this "step" later.
        if best_next_node:
            current_node = best_next_node
            visited.add(current_node)       # Mark as visited
            path.append(current_node)       # Add to path history
            total_cost += best_edge_data['c']
            total_distraction += best_edge_data['d']
        else:
            return None         # Should be caught by valid_neighbors check, but safety first.

    return total_cost, total_distraction, path


def path_finder_recursive(graph, start, end, opt_func, path=None, cost=0, dist=0):
    """
    Finds the optimal path from start to end using recursion and backtracking.
    Efficiency is improved by mutating a single path list rather than copying it.

    Args:
        graph (dict): The graph represented as an adjacency list with weights.
        start (str): The starting node.
        end (str): The destination node.
        opt_func (function): The optimization function to calculate edge scores.

    Returns:
        tuple: (total_effort, total_distraction, path_list) or None if no path exists.

    >>> graph = {
    ...     'S': {'A': {'c': 1, 'd': 0}, 'B': {'c': 5, 'd': 10}},
    ...     'A': {'E': {'c': 1, 'd': 0}},
    ...     'B': {'E': {'c': 5, 'd': 10}},
    ...     'E': {}
    ... }
    >>> # Path A: Score = 2 (1+1 cost, 0+0 dist)
    >>> # Path B: Score = -2 (5+5 cost, 10+10 dist) -> opt_balanced_linear(10, 20)
    >>> path_finder_recursive(graph, 'S', 'E', opt_balanced_linear)
    (10, 20, ['S', 'B', 'E'])

    >>> # Negative Test: No path exists
    >>> graph_iso = {'S': {'A': {'c': 1, 'd': 1}}, 'B': {}}
    >>> path_finder_recursive(graph_iso, 'S', 'B', opt_balanced_linear) is None
    True

    >>> # Test: Start is End
    >>> path_finder_recursive(graph, 'S', 'S', opt_balanced_linear)
    (0, 0, ['S'])
    """
    # 1. Initialization: On the first call, create the tracking list
    if path is None:
        path = [start]

    # 2. Base Case: Destination reached
    if start == end:
        # We return a copy of the path [:] because the original list
        # will continue to be mutated (popped) during backtracking.
        return cost, dist, path[:]

    best_result = None
    min_score = float('inf')

    # 3. Exploration: Safely get neighbors to avoid KeyErrors on leaf nodes
    neighbors = graph.get(start, {})

    for neighbor, attr in neighbors.items():
        # Prevent cycles (infinite loops) by checking if node is already visited
        if neighbor not in path:

            # Backtracking Step:
            # We modify the existing list to save memory.
            path.append(neighbor)

            # Recursive call: Move deeper into the graph
            result = path_finder_recursive(
                graph, neighbor, end, opt_func,
                path, cost + attr['c'], dist + attr['d']
            )

            # 4. Evaluation: If a valid path to 'end' was returned
            if result:
                res_cost, res_dist, _ = result
                score = opt_func(res_cost, res_dist)

                # Keep track of the path with the lowest optimization score
                if score < min_score:
                    min_score = score
                    best_result = result

            # Backtracking Step:
            # Remove the neighbor before trying the next branch
            # so the 'path' list is clean for the next iteration.
            path.pop()

    return best_result


if __name__ == "__main__":
    import doctest

    print("Running Doctests...")
    results = doctest.testmod()
    if results.failed == 0:
        print("All doctests passed successfully!")
    else:
        print(f"{results.failed} doctests failed.")
