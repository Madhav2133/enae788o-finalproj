'''
This script is used to implement the Standard CSA approach where the agents try to move toawards a target and aim to circle around it.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# ---------- Parameters ----------
NUM_AGENTS = 10
WORLD_SIZE = 100
CELL_SIZE = 10
DT = 0.1
STEPS = 1000

SEP_RADIUS = 10
ALI_RADIUS = 8
COH_RADIUS = 10
MAX_FORCE = 0.5
MAX_SPEED = 2.0

W_SEP = 1.5
W_ALI = 1.0
W_COH = 1.2
W_GOAL = 1.5

DRAG_COEFF = 0.95
CURRENT_STRENGTH = 0.05
GOAL_POINT = np.array([50, 50])

# ---------- Force Functions ----------
def separation(i, positions):
    diffs = positions - positions[i]
    dists = np.linalg.norm(diffs, axis=1)
    neighbors = np.where((dists < SEP_RADIUS) & (dists > 0))[0]
    force = np.zeros(2)
    for j in neighbors:
        diff = positions[i] - positions[j]
        norm = np.linalg.norm(diff) + 1e-5
        force += diff / norm
    return force


def alignment(i, positions, velocities):
    dists = np.linalg.norm(positions - positions[i], axis=1)
    neighbors = np.where((dists < ALI_RADIUS) & (dists > 0))[0]
    if neighbors.size > 0:
        avg_velocity = np.mean(velocities[neighbors], axis=0)
        return avg_velocity - velocities[i]
    return np.zeros(2)

def cohesion(i, positions):
    dists = np.linalg.norm(positions - positions[i], axis=1)
    neighbors = np.where((dists < COH_RADIUS) & (dists > 0))[0]
    if neighbors.size > 0:
        center = np.mean(positions[neighbors], axis=0)
        return center - positions[i]
    return np.zeros(2)

def goal_force(pos):
    vec = GOAL_POINT - pos
    dist = np.linalg.norm(vec)
    return vec / dist if dist > 5 else np.zeros(2)

def water_current(pos, t):
    return CURRENT_STRENGTH * np.array([np.sin(0.1 * pos[1] + 0.01 * t), 0])

# ---------- Simulation Setup ----------
def run_simulation():
    # Initialization
    y = np.linspace(10, WORLD_SIZE - 10, NUM_AGENTS)
    positions = np.vstack([np.zeros(NUM_AGENTS), y]).T
    velocities = (np.random.rand(NUM_AGENTS, 2) - 0.5) * 2
    grid_dim = WORLD_SIZE // CELL_SIZE
    visited = np.zeros((grid_dim, grid_dim))

    # Plot setup
    cmap = colors.ListedColormap(['white', 'gray'])
    bounds = [0, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    heatmap = ax.imshow(visited, cmap=cmap, norm=norm, origin='lower',
                        extent=[0, WORLD_SIZE, 0, WORLD_SIZE])
    scatter = ax.scatter(positions[:, 0], positions[:, 1], color='blue')
    ax.plot(GOAL_POINT[0], GOAL_POINT[1], 'r*', markersize=10)

    for x in range(0, WORLD_SIZE+1, CELL_SIZE):
        ax.axvline(x, linestyle=':', color='k', linewidth=0.5)
    for y in range(0, WORLD_SIZE+1, CELL_SIZE):
        ax.axhline(y, linestyle=':', color='k', linewidth=0.5)

    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(0, WORLD_SIZE)
    ax.set_title('CSA Underwater Simulation')
    ax.set_aspect('equal')

    # Simulation loop
    for t in range(STEPS):
        new_vel = np.zeros_like(velocities)

        # Update coverage map
        for i in range(NUM_AGENTS):
            gx = int(positions[i, 0] // CELL_SIZE)
            gy = int(positions[i, 1] // CELL_SIZE)
            if 0 <= gx < grid_dim and 0 <= gy < grid_dim:
                visited[gy, gx] = 1

        # Update agent velocities
        for i in range(NUM_AGENTS):
            acc = (
                W_SEP * separation(i, positions) +
                W_ALI * alignment(i, positions, velocities) +
                W_COH * cohesion(i, positions) +
                W_GOAL * goal_force(positions[i])
            )

            if np.linalg.norm(acc) > MAX_FORCE:
                acc = acc / np.linalg.norm(acc) * MAX_FORCE

            vel = velocities[i] + acc + water_current(positions[i], t)
            if np.linalg.norm(vel) > MAX_SPEED:
                vel = vel / np.linalg.norm(vel) * MAX_SPEED

            new_vel[i] = vel

        # Update positions and apply drag
        velocities = new_vel * DRAG_COEFF
        positions += velocities * DT
        positions = np.clip(positions, 0, WORLD_SIZE)

        # Update plot
        heatmap.set_data(visited)
        scatter.set_offsets(positions)
        plt.pause(0.01)

    plt.show()

if __name__ == "__main__":
    run_simulation()
