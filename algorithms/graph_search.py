from heapq import heappush, heappop  # Recommended.
import numpy as np

from flightsim.world import World

from .occupancy_map import OccupancyMap # Recommended.

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index  = tuple(occ_map.metric_to_index(goal))

    # STUDENT CODE HERE
    # Infeasible endpoints.
    if occ_map.is_occupied_index(start_index) or occ_map.is_occupied_index(goal_index):
        return None, 0

    # Start and goal share a single voxel.
    if start_index == goal_index:
        path = np.vstack((np.asarray(start), np.asarray(goal)))
        return path, 0

    # Search state: open heap, best cost-to-come, and parent links.
    open_heap = []
    g_best    = {start_index: 0.0}
    parent    = {start_index: None}

    nodes_expanded = 0
    tie_break      = 1

    # Seed the open heap with the start node.
    gx, gy, gz = goal_index
    if astar:
        dx = (start_index[0] - gx) * resolution[0]
        dy = (start_index[1] - gy) * resolution[1]
        dz = (start_index[2] - gz) * resolution[2]
        h_start = np.sqrt(dx*dx + dy*dy + dz*dz)
        heappush(open_heap, (h_start, 0, 0.0, start_index))
    else:
        heappush(open_heap, (0.0, 0, 0.0, start_index))

    while open_heap:
        _, _, g, node = heappop(open_heap)

        # Stale entry, superseded by a cheaper path to the same node.
        if g != g_best.get(node, float("inf")):
            continue

        if node == goal_index:
            # Backtrack parent links from goal to start.
            path_indices = []
            k = node
            while parent[k] is not None:
                path_indices.append(k)
                k = parent[k]
            path_indices.append(start_index)
            path_indices.reverse()

            # Voxel centers, with the exact start and goal at the ends.
            path_pts = []
            for index in path_indices:
                path_pts.append(occ_map.index_to_metric_center(index))
            path = np.vstack(path_pts)

            path[0]  = start
            path[-1] = goal
            return path, nodes_expanded

        nodes_expanded += 1

        # Expand the 26-connected neighborhood.
        nx, ny, nz = node

        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for oz in (-1, 0, 1):
                    if ox == 0 and oy == 0 and oz == 0:
                        continue

                    neighbor = (nx + ox, ny + oy, nz + oz)
                    if occ_map.is_occupied_index(neighbor):
                        continue

                    # Metric edge cost of one voxel step.
                    step_cost = np.sqrt((ox*resolution[0])**2
                                        + (oy*resolution[1])**2
                                        + (oz*resolution[2])**2)
                    g_new = g + step_cost

                    # Relax the neighbor when this path is cheaper.
                    g_old = g_best.get(neighbor)
                    if g_old is None or g_new < g_old:
                        g_best[neighbor] = g_new
                        parent[neighbor] = node

                        tie_break += 1
                        if astar:
                            dx = (neighbor[0] - gx) * resolution[0]
                            dy = (neighbor[1] - gy) * resolution[1]
                            dz = (neighbor[2] - gz) * resolution[2]
                            h = np.sqrt(dx*dx + dy*dy + dz*dz)
                            heappush(open_heap, (g_new + h, tie_break, g_new, neighbor))
                        else:
                            heappush(open_heap, (g_new, tie_break, g_new, neighbor))

    # Return a tuple (path, nodes_expanded)
    return None, nodes_expanded
