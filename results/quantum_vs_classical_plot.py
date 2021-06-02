import time

import numpy as np
from matplotlib import pyplot as plt

import results.classical_8width as classical
import results.quantum_8width as quantum

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 16,
    "font.size": 12,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.titlesize": 11
}
plt.rcParams.update(tex_fonts)

q1 = quantum.results_sim_circ_14_half_null_AF_statevector()
q2 = quantum.results_abbas_3_layers_null_AF_statevector()
q3 = quantum.results_sim_circ_14_half_null_AF_qasm_100_shots()
q4 = quantum.results_sim_circ_14_half_partial_measurement_half_AF_qasm_100_shots()

c_sigmoid_fixed = classical.results_sigmoid_bounded_fixed_2()

q1_unzipped = [list(zip(*run)) for run in q1]
q2_unzipped = [list(zip(*run)) for run in q2]
q3_unzipped = [list(zip(*run)) for run in q3]
q4_unzipped = [list(zip(*run)) for run in q4]

c1_unzipped = [list(zip(*run)) for run in c_sigmoid_fixed]

ax = plt.subplot(111)
c1_accs = np.mean([run[0][:-1] for run in c1_unzipped], 0)
c1_std = np.std([run[0][:-1] for run in c1_unzipped], 0)

q1_accs = np.mean([run[0] for run in q1_unzipped], 0)
q1_std = np.std([run[0] for run in q1_unzipped], 0)

q2_accs = np.mean([run[0][:-1] for run in q2_unzipped], 0)
q2_std = np.std([run[0][:-1] for run in q2_unzipped], 0)

plt.errorbar(np.array([checkpoint[0] * 97 + checkpoint[1] for checkpoint in c1_unzipped[0][1][:-1]]) + 1,
             c1_accs, c1_std, fmt="--o",
             markersize=4, mew=2, label='Classical', capsize=2, alpha=1.0)
print(c1_accs[-1], c1_std[-1])

plt.errorbar(np.array([checkpoint[0] * 97 + checkpoint[1] for checkpoint in q1_unzipped[0][1]]) + 1,
             q1_accs, q1_std, fmt="--o",
             markersize=4, mew=2, label='Quantum, ring ansatz', capsize=2, alpha=1.0)
print(q1_accs[-1], q1_std[-1])

plt.errorbar(np.array([checkpoint[0] * 97 + checkpoint[1] for checkpoint in q2_unzipped[0][1][:-1]]) + 1,
             q2_accs, q2_std, fmt="--o",
             markersize=4, mew=2, label='Quantum, all-to-all ansatz', capsize=2)
print(q2_accs[-1], q2_std[-1])

next(ax._get_lines.prop_cycler)['color']
next(ax._get_lines.prop_cycler)['color']

# q3_accs = np.mean([run[0][:-1] for run in q3_unzipped], 0)
# q3_std = np.std([run[0][:-1] for run in q3_unzipped], 0)
# plt.errorbar(np.array([checkpoint[0] * 97 + checkpoint[1] for checkpoint in q3_unzipped[0][1][:-1]]) + 1,
#              q3_accs, q3_std, fmt="--o",
#              markersize=4, mew=2, label='Quantum, 100 shots', capsize=2)
# print(q3_accs[-1], q3_std[-1])

# print(np.mean(q3_std))

plt.xlabel('Training batches')
plt.ylabel('Accuracy ($\%$)')
plt.ylim(27, 48)
# plt.xlim(0, 97)
ax.tick_params(axis='both', which='major')
plt.grid(alpha=0.5)
plt.legend(title='Repr. Network', loc=4)
plt.savefig('figures/{}.pdf'.format(time.time()), dpi=3000, bbox_inches='tight')
# plt.show()
