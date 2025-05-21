'''
This script is an extension of Modified CSA script by implementing the Search and Rescue (SAR) mission.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from modified_csa import separation, alignment, cohesion, water_current

# ---------- Parameters ----------
NUM_AGENTS = 10
WORLD_SIZE = 100
CELL_SIZE = 10
DT = 0.1
STEPS = 1000
DRAG_COEFF = 0.95
MAX_FORCE = 0.5
MAX_SPEED = 2.0
NUM_TARGETS = 3
W_INFO = 2.0

def goal_force(pos, targets, found):
    unfound = [t for i, t in enumerate(targets) if not found[i]]
    if not unfound:
        return np.zeros(2)
    dists = [np.linalg.norm(t * CELL_SIZE + CELL_SIZE / 2 - pos) for t in unfound]
    nearest = unfound[np.argmin(dists)]
    vec = nearest * CELL_SIZE + CELL_SIZE / 2 - pos
    dist = np.linalg.norm(vec)
    return vec / dist if dist > 5 else np.zeros(2)

# ---------- SAR Simulation ----------
def run_modified_csa_sar(target_cells=None):
    # Initialization
    y = np.linspace(10, WORLD_SIZE - 10, NUM_AGENTS)
    positions = np.vstack([np.zeros(NUM_AGENTS), y]).T
    velocities = (np.random.rand(NUM_AGENTS, 2) - 0.5) * 2
    grid_dim = WORLD_SIZE // CELL_SIZE
    visited = np.zeros((grid_dim, grid_dim))

    if target_cells is None:
        target_cells = np.random.randint(0, grid_dim, size=(NUM_TARGETS, 2))

    found_targets = [False] * NUM_TARGETS
    rescuer_ids = [None] * NUM_TARGETS

    coverage_over_time = []
    agent_tracks = [[] for _ in range(NUM_AGENTS)]

    # Plot setup
    fig, ax = plt.subplots()
    cmap = colors.ListedColormap(['white', 'gray'])
    norm = colors.BoundaryNorm([0, 0.5, 1], cmap.N)
    heatmap = ax.imshow(visited, cmap=cmap, norm=norm, origin='lower',
                        extent=[0, WORLD_SIZE, 0, WORLD_SIZE])
    scatter = ax.scatter(positions[:, 0], positions[:, 1], c='blue')
    [ax.plot(tx * CELL_SIZE + CELL_SIZE / 2, ty * CELL_SIZE + CELL_SIZE / 2, 'ro')[0]
     for tx, ty in target_cells]

    for x in range(0, WORLD_SIZE + 1, CELL_SIZE):
        ax.axvline(x, linestyle=':', color='k', linewidth=0.5)
    for y in range(0, WORLD_SIZE + 1, CELL_SIZE):
        ax.axhline(y, linestyle=':', color='k', linewidth=0.5)

    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(0, WORLD_SIZE)
    ax.set_title("SAR with Modified CSA (Information Bias)")
    ax.set_aspect('equal')

    # Simulation 
    for t in range(STEPS):
        new_vel = np.zeros_like(velocities)

        for i in range(NUM_AGENTS):
            agent_tracks[i].append(positions[i].copy())

            gx = int(positions[i, 0] // CELL_SIZE)
            gy = int(positions[i, 1] // CELL_SIZE)
            if 0 <= gx < grid_dim and 0 <= gy < grid_dim:
                visited[gy, gx] = 1
                for idx, (tx, ty) in enumerate(target_cells):
                    if not found_targets[idx] and gx == tx and gy == ty:
                        found_targets[idx] = True
                        rescuer_ids[idx] = i
                        print(f"Target {idx+1} found by Agent {i} at step {t}")

        coverage_percent = np.sum(visited) / visited.size * 100
        coverage_over_time.append(coverage_percent)

        unexplored = 1.0 - visited
        Gy, Gx = np.gradient(unexplored)

        # Velocities
        for i in range(NUM_AGENTS):
            if i in rescuer_ids:
                new_vel[i] = np.zeros(2)
                continue

            acc = (
                1.5 * separation(i, positions) +
                1.0 * alignment(i, positions, velocities) +
                1.2 * cohesion(i, positions) +
                1.5 * goal_force(positions[i], target_cells, found_targets)   # ✅ new
            )

            gx_idx = int(positions[i, 0] // CELL_SIZE)
            gy_idx = int(positions[i, 1] // CELL_SIZE)
            if 0 <= gx_idx < grid_dim and 0 <= gy_idx < grid_dim:
                grad = np.array([Gx[gy_idx, gx_idx], Gy[gy_idx, gx_idx]])
                if np.linalg.norm(grad) > 1e-3:
                    info_force = grad / (np.linalg.norm(grad) + 1e-5)
                    acc += W_INFO * info_force

            if np.linalg.norm(acc) > MAX_FORCE:
                acc = acc / np.linalg.norm(acc) * MAX_FORCE

            vel = velocities[i] + acc + water_current(positions[i], t)
            if np.linalg.norm(vel) > MAX_SPEED:
                vel = vel / np.linalg.norm(vel) * MAX_SPEED

            new_vel[i] = vel

        velocities = new_vel * DRAG_COEFF
        positions += velocities * DT
        positions = np.clip(positions, 0, WORLD_SIZE)

        colors_live = ['green' if i in rescuer_ids else 'blue' for i in range(NUM_AGENTS)]
        scatter.set_offsets(positions)
        scatter.set_color(colors_live)
        heatmap.set_data(visited)
        plt.pause(0.01)

        if all(found_targets):
            print("All targets found.")
            break

    plt.show()
    return agent_tracks, coverage_over_time

# ---------- Plot Function ----------
def show_final_plots(agent_tracks, coverage_over_time):
    import matplotlib.pyplot as plt2

    plt2.figure()
    plt2.plot(coverage_over_time)
    plt2.xlabel("Time Step")
    plt2.ylabel("Coverage (%)")
    plt2.title("Coverage Over Time (Modified CSA)")
    plt2.grid(True)
    plt2.savefig("mod_csa_coverage_over_time.png")

    plt2.figure()
    for path in agent_tracks:
        path = np.array(path)
        plt2.plot(path[:, 0], path[:, 1])
    plt2.xlabel("X")
    plt2.ylabel("Y")
    plt2.title("Agent Trajectories (Modified CSA)")
    plt2.xlim(0, WORLD_SIZE)
    plt2.ylim(0, WORLD_SIZE)
    plt2.gca().set_aspect('equal')
    plt2.grid(True)
    plt2.savefig("mod_csa_agent_trajectories.png")

    plt2.show()

if __name__ == "__main__":
    tracks, coverage = run_modified_csa_sar()
    show_final_plots(tracks, coverage)