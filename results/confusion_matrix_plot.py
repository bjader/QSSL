import time

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

q_real_device = np.load('../model/sup/simclr/quantum_pretrained_checkpoint_0001_0092_1/accuracy_metrics.npy',
                        allow_pickle=True)
classical = np.load('../model/sup/simclr/classical_pretrained_checkpoint_0001_0092_1/accuracy_metrics.npy',
                    allow_pickle=True)

samples = 900

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

for result in [q_real_device]:
    labels = [s for i in result[1] for s in i]
    predictions = [s for i in result[2] for s in i]
    m = confusion_matrix(labels[:samples], predictions[:samples])
    yticklabels = ['aeroplane', 'automobile', 'bird', 'cat', 'deer']
    xticklabels = ['aero.', 'auto.', 'bird', 'cat', 'deer']

    ax = plt.subplot(111)

    sns.heatmap(m, annot=True, cmap='Blues', cbar=False, annot_kws={"size": 25}, xticklabels=xticklabels,
                yticklabels=yticklabels, fmt='d', square=True)
    # plt.xlabel('Predicted', fontsize=30)
    # plt.ylabel('True', fontsize=30)
    ax.xaxis.set_label_position('top')
    plt.tick_params(axis='both', which='major', labelsize=25, labelbottom=False, bottom=False, top=False, labeltop=True,
                    left=False, labelleft=False)
    # plt.subplots_adjust(left=0.2)
    plt.tight_layout()
    plt.savefig('figures/confusion_matrix_{}.pdf'.format(time.time()), dpi=3000, bbox_inches='tight')
    plt.show()
    # plt.close()
