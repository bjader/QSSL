import logging

import numpy as np
from matplotlib import pyplot as plt

import results.classical_ablation.classical_epochsize_data as classical_epochsize
import results.classical_ablation.classical_width_data as classical_width

logging.basicConfig(level=logging.INFO)

widths, accs1 = classical_width.results_updated_nobatchnorm()
epochsizes, accs3 = classical_epochsize.results_low_tepochs()

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

ax = plt.subplot(111)
colors = []

# for i, width_results in enumerate(accs1):
#     unzipped = list(zip(*width_results))
#     color = next(ax._get_lines.prop_cycler)['color']
#     plt.plot(np.array(unzipped[1]) + 1, unzipped[0], "--x", markersize=10, mew=2, color=color,
#              label="Width={}".format(widths[i]))
#     colors.append(color)

for i, width_results in enumerate(accs1):
    unzipped = [list(zip(*run)) for run in width_results]

    plt.errorbar(np.array([checkpoint[0] * 97 + checkpoint[1] for checkpoint in unzipped[0][1]]) + 1,
                 np.mean([run[0] for run in unzipped], 0), np.std([run[0] for run in unzipped], 0), fmt="--x",
                 markersize=10, mew=2, label="Width={}".format(widths[i]))

# for i, epoch_results in enumerate(accs3):
#     unzipped = list(zip(*epoch_results))
#     plt.plot(np.array(np.array(unzipped[1])+1)*epochsizes[i], unzipped[0], "--x", markersize=10, mew=2,
#              label="Training data={}%".format(epochsizes[i] * 100 / 25000))

# plt.vlines(100000, 25, 90, linestyles='dashed', lw=0.3)

# plt.plot(np.array(baseline.results()[0])+1, baseline.results()[1], "--x", markersize=10, mew=2, label="Baseline")

plt.xlabel('Number of training batches')
plt.ylabel('Accuracy ($\%$)')
ax.tick_params(axis='both', which='major')
# plt.ylim(25, 55)
# plt.xlim(0, 100)
plt.legend()
plt.grid(alpha=0.5)
# plt.title('Without Batchnorm')
# plt.savefig('figures/{}.pdf'.format(time.time()), dpi=3000, bbox_inches='tight')
plt.show()
