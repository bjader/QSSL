import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator

import results.quantum_width_ablation.quantum_width_data as quantum_width
import results.classical_ablation.classical_width_data as classical_width
import results.classical_8width as classical
from results.quantum_8width import results_sim_circ_14_half_null_AF_statevector

logging.basicConfig(level=logging.INFO)

quantum_widths, quantum_accs1 = quantum_width.results()

classical_8width = classical.results_sigmoid_bounded_fixed_2()

classical_widths, classical_accs1 = classical_width.results_updated_nobatchnorm()

quantum_accs_8width = results_sim_circ_14_half_null_AF_statevector

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

fig, [[w2_ax, w4_ax], [w6_ax, w8_ax]] = plt.subplots(2, 2, sharex=True, sharey=True)
colors = []

colours = {"2": 'tab:blue', "4": 'tab:orange', "6": 'tab:red', "8": 'tab:green'}

for ax, width_to_plot in zip([w2_ax, w4_ax, w6_ax, w8_ax], [2, 4, 6, 8]):

    for i, width_results in enumerate(quantum_accs1):

        if quantum_widths[i] == width_to_plot:
            unzipped = [list(zip(*run)) for run in width_results]

            completed_data_points = min([len(run[0]) for run in unzipped])

            ax.errorbar(
                np.array(
                    [checkpoint[0] * 97 + checkpoint[1] for checkpoint in unzipped[0][1][:completed_data_points]]) + 1,
                np.nanmean([run[0][:completed_data_points] for run in unzipped], 0),
                np.nanstd([run[0][:completed_data_points] for run in unzipped], 0), fmt="--o",
                markersize=4, mew=2, color=colours[str(quantum_widths[i])], capsize=2)

            ax.plot([], 'o--', color=colours[str(quantum_widths[i])], label="Quantum")

    for i, width_results in enumerate(classical_accs1):
        if classical_widths[i] == width_to_plot:
            if width_to_plot == 8:
                unzipped = [list(zip(*run)) for run in classical_8width]
            else:
                unzipped = [list(zip(*run)) for run in width_results]

            completed_data_points = min([len(run[0]) for run in unzipped])

            ax.errorbar(
                np.array(
                    [checkpoint[0] * 97 + checkpoint[1] for checkpoint in unzipped[0][1][:completed_data_points]]) + 1,
                np.nanmean([run[0][:completed_data_points] for run in unzipped], 0),
                np.nanstd([run[0][:completed_data_points] for run in unzipped], 0), fmt="x--",
                markersize=8, mew=2, color=colours[str(classical_widths[i])], capsize=2)
            ax.plot([], 'x--', color=colours[str(classical_widths[i])], label="Classical")

    ax.set_xlim(-5, 180)
    # ax.set_ylim(18, 52)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, bottom=True, left=True, right=True)
    ax.grid(alpha=0.5)

w2_ax.legend(fancybox=False, loc="upper right", title=r"$W=2$")
w4_ax.legend(fancybox=False, loc="upper right", title=r"$W=4$")
w6_ax.legend(fancybox=False, loc="lower right", title=r"$W=6$")
w8_ax.legend(fancybox=False, loc="lower right", title=r"$W=8$")

fig.supxlabel('Training batches')
fig.supylabel('Accuracy ($\%$)')
fig.subplots_adjust(wspace=0.05, hspace=0.05)
plt.savefig('../figures/quantum_classical_ablation.pdf', dpi=3000, bbox_inches='tight')
plt.show()
