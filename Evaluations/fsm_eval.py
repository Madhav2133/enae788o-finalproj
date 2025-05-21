'''
This script is introduced to evaluate the approach with various swarm sizes and number of targets.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from fsm_sim import run_fsm_sar, show_final_plots

# Range of agents and targets
agent_counts = [5, 10, 15, 20]
target_counts = [1, 3, 5]

'''
Store final results
Format: {(n_agents, n_targets): (final_coverage, steps_taken)}
'''
results = {}

for n in agent_counts:
    for k in target_counts:
        print(f"\nRunning FSM SAR with {n} agents and {k} targets")

        # Set CSA weights and radii based on n
        from modified_csa import SEP_RADIUS, ALI_RADIUS, COH_RADIUS, W_SEP, W_ALI, W_COH
        if n == 5:
            SEP_RADIUS, ALI_RADIUS, COH_RADIUS = 15, 12, 15
            W_SEP, W_ALI, W_COH = 2.0, 1.2, 1.5
        elif n == 10:
            SEP_RADIUS, ALI_RADIUS, COH_RADIUS = 10, 8, 10
            W_SEP, W_ALI, W_COH = 1.5, 1.0, 1.2
        elif n == 15:
            SEP_RADIUS, ALI_RADIUS, COH_RADIUS = 8, 7, 9
            W_SEP, W_ALI, W_COH = 1.3, 0.9, 1.1
        elif n == 20:
            SEP_RADIUS, ALI_RADIUS, COH_RADIUS = 7, 6, 8
            W_SEP, W_ALI, W_COH = 1.2, 0.8, 1.0

        # Override agent and target parameters
        import fsm_sim
        fsm_sim.NUM_AGENTS = n
        fsm_sim.NUM_TARGETS = k

        # Run simulation
        tracks, coverage = run_fsm_sar()

        # Show and save plots
        show_final_plots(tracks, coverage)

        # Determine how many steps it took to find all targets
        steps_taken = len(coverage)
        final_coverage = coverage[-1]
        results[(n, k)] = (final_coverage, steps_taken)

'''
# ----- Tabulated Results -----
print("\nSummary Table (Coverage %, Steps Taken)")
print("{:<10} {:<10} {:<15} {:<15}".format("Agents", "Targets", "Coverage (%)", "Steps"))
for (n, k), (coverage, steps) in sorted(results.items()):
    print("{:<10} {:<10} {:<15.2f} {:<15}".format(n, k, coverage, steps))
'''

# ----- Comparison Plot -----
plt.figure()
for k in target_counts:
    y = [results[(n, k)][0] for n in agent_counts]
    plt.plot(agent_counts, y, marker='o', label=f'{k} Targets')

plt.xlabel("Number of Agents")
plt.ylabel("Final Coverage (%)")
plt.title("FSM SAR: Final Coverage vs Agents")
plt.legend()
plt.grid(True)
plt.savefig("fsm_final_coverage_comparison.png")
plt.show()

# Steps comparison plot
plt.figure()
for k in target_counts:
    y = [results[(n, k)][1] for n in agent_counts]
    plt.plot(agent_counts, y, marker='s', linestyle='--', label=f'{k} Targets')

plt.xlabel("Number of Agents")
plt.ylabel("Steps to Find All Targets")
plt.title("FSM SAR: Steps vs Agents")
plt.legend()
plt.grid(True)
plt.savefig("fsm_final_steps_comparison.png")
plt.show()
