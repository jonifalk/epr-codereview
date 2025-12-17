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

    >>> opt_min_cost(10, 5)
    10
    >>> opt_min_cost(5, 10)
    5
    """
    return cost

def opt_min_distraction(cost, distraction):
    """
    Optimization 5: Minimize Distraction.
    Focuses solely on minimizing distraction.

    >>> min_distraction(10, 5)
    5
    >>> min_distraction(5, 10)
    10
    """
    return distraction

def opt_balanced(cost, distraction):
    """
    Optimization 6: Balanced Sum.
    Minimizes the sum of cost and distraction.

    >>> opt_balanced(10, 5)
    15
    >>> opt_balanced(5, 10)
    15
    """
    return cost + distraction

# Algorithms

def path_finder_dijkstra(graph, start, end, opt_func):
    """
    Dijkstra path finder with two edge weights: c, d
    
    :param graph: Beschreibung
    :param start: Beschreibung
    :param end: Beschreibung
    :param opt_func: Beschreibung
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
    total_cost = 0
    total_distraction = 0

    current = end
    while current is not None:
        path.append(current)
        prev = previous[current]
        if prev is not None:
            edge = graph[prev][current]
            total_cost += edge['c']
            total_distraction += edge['d']
        current = prev

    path.reverse()

    print(f"Path found: {path} with total cost {total_cost}"
          f" and total distraction {total_distraction}")
    return total_cost, total_distraction, path


def path_finder_greedy(graph, start, end, opt_func):
    """
    Approaches the destination by choosing the locally optimal edge at each step.
    Note: Does not backtrack and may fail to find the destination or a global optimum.

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


def path_finder_recursive(graph, start, end, opt_func, path=None, current_cost=0, current_dist=0):
    """
    Recursively finds the global optimum path based on
    the opt_func applied to the TOTAL path values.

    Returns:
        tuple: (total_effort, total_distraction, path_list)

    >>> graph = {
    ...     'S': {'A': {'c': 1, 'd': 0}, 'B': {'c': 5, 'd': 10}},
    ...     'A': {'E': {'c': 1, 'd': 0}},
    ...     'B': {'E': {'c': 5, 'd': 10}},
    ...     'E': {}
    ... }

    # Balanced Linear:
    # Path via A: Cost 2, Dist 0 -> Score 2
    # Path via B: Cost 10, Dist 20 -> Score -10 (Better)
    >>> path_finder_recursive(graph, 'S', 'E', opt_balanced_linear)
    (10, 20, ['S', 'B', 'E'])

    # Negative Test: No path exists
    >>> graph_iso = {'S': {'A': {'c': 1, 'd': 1}}, 'B': {}}
    >>> path_finder_recursive(graph_iso, 'S', 'B', opt_balanced_linear) is None
    True

    # Multiple paths logic check
    >>> path_finder_recursive(graph, 'S', 'S', opt_balanced_linear)
    (0, 0, ['S'])
    """
    # Set up the path list (On the first call, it's empty)
    if path is None:
        path = [start]

    # If we reached the house (end), we stop exploring this branch.
    # We return the totals we have accumulated so far.
    if start == end:
        return current_cost, current_dist, path

    # 3. Tracking the best path
    # Since we might split into 3 different paths here, we need to know which
    # of those 3 returns the best score.
    best_result = None
    best_path_score = float('inf')      # Infinity ensures the first valid path replaces this.

    # Exploration
    if start in graph:
        for neighbor, attr in graph[start].items():

            # Do not go back to a node already in our current history ('path')
            if neighbor not in path:

                # Calculation: Add this step's cost/distraction
                # to our running total BEFORE passing it down.
                new_cost = current_cost + attr['c']
                new_dist = current_dist + attr['d']
                new_path = path + [neighbor]

                # For recursion: "Pause" this function and
                # call a fresh version of it for the neighbor.
                # This will go all the way to the end before returning.
                result = path_finder_recursive(graph, neighbor, end, opt_func,
                                               new_path, new_cost, new_dist)

                # Backtracking and evaluating
                # We are now "back" at the current node. We look at what the neighbor returned.
                if result:
                    r_cost, r_dist, r_path = result

                    # Apply the optimization function to the TOTAL journey returned by the child.
                    # Note the difference: Greedy checked the edge score.
                    # Recursive checks the total path score.
                    score = opt_func(r_cost, r_dist)

                    # If this path is better than previous neighbor paths we checked...
                    if score < best_path_score:
                        best_path_score = score
                        best_result = result

    # Return the single best path found among all neighbors to the node that called us.
    return best_result


if __name__ == "__main__":
    if True:
        # Test graph with different routes for Dijkstra
        g = {
            'A': {'B': {'c': 1, 'd': 10}, 'E': {'c': 10, 'd': 1}},
            'B': {'C': {'c': 1, 'd': 10}, 'F': {'c': 100, 'd': 100}},
            'C': {'D': {'c': 1, 'd': 10}},
            'D': {},
            'E': {'C': {'c': 100, 'd': 100}, 'G': {'c': 10, 'd': 10}},
            'F': {'D': {'c': 10, 'd': 1}},
            'G': {'F': {'c': 10, 'd': 1}},
        }
        print("\nBalanced Linear Dijkstra:")
        path_finder_dijkstra(g, 'A', 'D', opt_balanced_linear)
        print("\nEfficiency Ratio Dijkstra:")
        path_finder_dijkstra(g, 'A', 'D', opt_efficiency_ratio)
        print("\nDistraction Focused Dijkstra:")
        path_finder_dijkstra(g, 'A', 'D', opt_distraction_focused)
        print("\nMin Cost Dijkstra:")
        path_finder_dijkstra(g, 'A', 'D', opt_min_cost)
        print("\nMin Distraction Dijkstra:")
        path_finder_dijkstra(g, 'A', 'D', opt_min_distraction)
        print("\nBalanced Dijkstra:")
        path_finder_dijkstra(g, 'A', 'D', opt_balanced)
    else:
        import doctest

        print("Running Doctests...")
        results = doctest.testmod()
        if results.failed == 0:
            print("All doctests passed successfully!")
        else:
            print(f"{results.failed} doctests failed.")
