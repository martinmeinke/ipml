import matplotlib.pyplot as plt
import numpy as np

def heatmap(title, scores, xlabels, ylabels):
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.90)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.gist_heat, vmin=0.5, vmax=.75)
    plt.xlabel('sigma')
    plt.ylabel('c')
    plt.colorbar()
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=45)
    plt.yticks(np.arange(len(ylabels)), ylabels)
    plt.suptitle(title)
    plt.show()
