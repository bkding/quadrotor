import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation

from flightsim.animate import _decimate_index
from flightsim.axes3ds import Axes3Ds, world_transformation
from flightsim.crazyflie_params import quad_params
from flightsim.sensors.stereo_utils import StereoUtils
from flightsim.sensors.vio_utils import Vio
from flightsim.shapes import Quadrotor as QuadrotorShape
from flightsim.simulate import Quadrotor, simulate, ExitStatus
from flightsim.world import World

#######################################################################
from algorithms.se3_control import SE3Control
from algorithms.world_traj import WorldTraj
#######################################################################


class FlightAxes(Axes3Ds):
    # Fit the map at its initial view and keep that scale while dragging.
    view_span = None

    def apply_aspect(self, position=None):
        if position is None:
            position = self.get_position(original=True)
        self.set_position(position, 'active')

    def get_proj(self):
        limits = np.array([self.get_xlim3d(), self.get_ylim3d(), self.get_zlim3d()])
        spans = np.diff(limits, axis=1).ravel()
        aspect = spans / spans.max()
        world = world_transformation(*limits.ravel(), pb_aspect=aspect)
        center = aspect / 2
        elev, azim = np.radians([self.elev, self.azim])
        direction = np.array([np.cos(elev) * np.cos(azim),
                              np.cos(elev) * np.sin(azim), np.sin(elev)])
        self.eye = center + self.dist * direction
        self.vvec = -direction
        up = (0, 0, 1 if np.cos(elev) >= 0 else -1)
        view = proj3d.view_transformation(self.eye, center, up)
        matrix = self._projection(-self.dist, self.dist) @ view @ world

        if self.view_span is None:
            corners = np.array(np.meshgrid(*limits)).reshape(3, -1)
            projected = np.array(proj3d.proj_transform(*corners, matrix))[:2]
            self.view_span = 2 * np.max(np.abs(projected), axis=1)

        pixels = np.array([self.bbox.width, self.bbox.height])
        size = np.array([self.viewLim.width, self.viewLim.height])
        scale = np.min(0.80 * pixels / self.view_span) * size / pixels
        middle = np.array([self.viewLim.x0, self.viewLim.y0]) + size / 2
        matrix[:2] = scale[:, None] * matrix[:2] + middle[:, None] * matrix[3]
        return matrix


def draw_world(title, world, start, goal, filename):
    # Use the actual map bounds to keep the obstacles and flight easy to see.
    fig = plt.figure(title, figsize=(6.4, 4.8))
    ax = FlightAxes(fig, rect=[0.035, 0.035, 0.93, 0.91])
    world.draw(ax)
    xmin, xmax, ymin, ymax, zmin, zmax = world.world['bounds']['extents']
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax),
           xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.tick_params(labelsize=11, pad=0)
    ax.xaxis.labelpad = 10
    if filename == 'test_maze.json':
        ax.view_init(elev=75, azim=-60)
        ax.set_zticks([zmin, zmax])
    elif filename == 'test_window.json':
        ax.view_init(elev=40, azim=-135)
    elif filename == 'test_over_under.json':
        ax.view_init(elev=10, azim=100)
        ax.set_yticks([])
        ax.set_ylabel('')

    ax.plot([start[0]], [start[1]], [start[2]], 'go',
            markersize=8, markeredgewidth=1.6, markerfacecolor='none')
    ax.plot([goal[0]], [goal[1]], [goal[2]], 'ro',
            markersize=8, markeredgewidth=1.6, markerfacecolor='none')
    return fig, ax


if __name__ == '__main__':
    np.random.seed(0)

    # Load one test example. Change filename to test_window.json or test_over_under.json.
    filename = 'test_maze.json'
    file = Path(__file__).resolve().parent / 'tests' / filename
    world = World.from_file(file)          # World boundary and obstacles.
    start = world.world['start']          # Start point, shape=(3,).
    goal = world.world['goal']            # Goal point, shape=(3,).

    # Set up the quadrotor model, controller, and trajectory.
    quadrotor = Quadrotor(quad_params)
    my_se3_control = SE3Control(quad_params)
    planning_start_time = time.perf_counter()
    my_world_traj = WorldTraj(world, start, goal)
    planning_time = time.perf_counter() - planning_start_time

    # Set simulation parameters.
    t_final = 150
    initial_state = {'x': start,
                     'v': (0, 0, 0),
                     'q': (0, 0, 0, 1),  # [i,j,k,w]
                     'w': (0, 0, 0)}
    vio = Vio()
    max_num_features = 150
    sample_resolution = 0.75             # Stereo feature spacing, in meters.
    stereo = StereoUtils(world, vio.camera_matrix, sample_resolution=sample_resolution,
                         visualization=False, max_num_features=max_num_features)

    # Perform simulation.
    print()
    print('Simulate.')
    (sim_time, state, _, _, flat, exit_status, _) = simulate(
        initial_state, quadrotor, my_se3_control, my_world_traj,
        t_final, stereo=stereo, vio=vio)
    print(exit_status.value)

    # Print results.
    position = state['x']
    collision_pts = world.path_collisions(position, 0.25)
    stopped_at_goal = (exit_status == ExitStatus.COMPLETE
                       and np.linalg.norm(position[-1] - goal) <= 2.0)
    distance = np.linalg.norm(np.diff(position, axis=0), axis=1)
    position_error = np.linalg.norm(position - flat['x'], axis=1)
    rms = np.sqrt(np.mean(position_error ** 2))
    print()
    print('Results:')
    print(f"  No Collision:    {'pass' if collision_pts.size == 0 else 'FAIL'}")
    print(f"  Stopped at Goal: {'pass' if stopped_at_goal else 'FAIL'}")
    print(f'  Flight time:     {sim_time[-1]:.2f} seconds')
    print(f'  Flight distance: {distance.sum():.2f} meters')
    print(f'  Planning time:   {planning_time:.2f} seconds')
    print(f'  Tracking RMS:    {rms:.3f} meters')

    # Visualize the dense path, sparse waypoints, and smooth trajectory.
    fig, ax = draw_world('A* Path, Waypoints, and Trajectory', world, start, goal, filename)
    if my_world_traj.path is not None:
        world.draw_line(ax, my_world_traj.path, color='#d62728', linewidth=1.7)
    if my_world_traj.points is not None:
        world.draw_points(ax, my_world_traj.points, color='#762a83', markersize=5.2)
    if np.any(np.diff(flat['x'], axis=0)):
        world.draw_line(ax, flat['x'], color='black', linewidth=2.7)
    fig.legend(handles=[
        Line2D([], [], color='#d62728', linewidth=1.7, label='A* path'),
        Line2D([], [], color='#762a83', marker='o', linestyle='none', label='Waypoints'),
        Line2D([], [], color='black', linewidth=2.7, label='Min. snap')],
        loc='upper center', ncol=3, frameon=False, fontsize=14,
        bbox_to_anchor=(0.5, 0.984))

    # Position Tracking Error vs. Time
    fig, ax = plt.subplots(num='Position Tracking Error', figsize=(6.4, 4.8))
    fig.subplots_adjust(left=0.19, bottom=0.19, right=0.95, top=0.83)
    ax.set_title(f'RMS = {rms:.3f} m', fontsize=19, weight='bold', pad=14)
    ax.plot(sim_time, position_error, color='#176895', linewidth=2.4)
    ax.axhline(rms, color='#6b7177', linestyle='--', linewidth=1.2)
    ax.set(xlabel='Time (s)', ylabel='Position error (m)', xlim=(0, sim_time[-1]),
           ylim=(0, max(0.20, 1.05 * position_error.max())))
    ax.xaxis.label.set_size(17)
    ax.yaxis.label.set_size(17)
    ax.tick_params(labelsize=16)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Animation
    #
    # Update the vehicle and flown path at 30 frames per second.
    # The view can be rotated by dragging the mouse in either 3D window.
    fig, ax = draw_world('Flight', world, start, goal, filename)
    vehicle = QuadrotorShape(ax, color='#156d9b')
    rotation = Rotation.from_quat(state['q']).as_matrix()
    title = fig.text(0.5, 0.95, '', ha='center', va='center', fontsize=13)
    flight_distance = np.r_[0.0, np.cumsum(distance)]
    boundaries = np.r_[np.searchsorted(sim_time, np.arange(0, sim_time[-1], 0.4)),
                       len(sim_time) - 1]
    trail = []
    for first, last in zip(boundaries[:-1], boundaries[1:]):
        line = Line3DCollection([], colors='#1d77a5', linewidths=2.4,
                                capstyle='round', joinstyle='round')
        ax.add_collection3d(line)
        trail.append((first, last, line))

    def update(index):
        vehicle.transform(position[index].copy(), rotation[index])
        title.set_text(f't = {sim_time[index]:.2f} s')

        # Leave a small gap behind the vehicle so its attitude stays visible.
        limit = np.searchsorted(flight_distance, flight_distance[index] - 0.24,
                                side='right') - 1
        for first, last, line in trail:
            stop = min(last, limit)
            line.set_segments([position[first:stop + 1]] if stop > first else [])

    sample_time = np.r_[np.arange(0, sim_time[-1], 1 / 30), sim_time[-1]]
    frames = _decimate_index(sim_time, sample_time)
    ani = FuncAnimation(fig, update, frames=frames, interval=1000 / 30,
                        blit=False, repeat=False)
    plt.show()
