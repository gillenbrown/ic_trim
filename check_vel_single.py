import sys, os
import numpy as np
import betterplotlib as bpl
bpl.presentation_style()

sys.path.append("/u/home/gillenb/code/mine/ic_trim/")

import utils

ic_dir = os.path.abspath(sys.argv[1])

# ==============================================================================
# 
# check whether we have baryons and read data
# 
# ==============================================================================
# Check if we have baryons or not. If we do not, the files will be of the form
# music_D.extension, while if we do, they will be of the form 
# "music_H.extension"
# see if the baryon file is in the directory
ic_files = os.listdir(ic_dir)
has_baryons = "music_H.md" in ic_files

data = utils.ARTData(ic_dir, has_baryons)

# ==============================================================================
#
# make the plot
# 
# ==============================================================================
for kind in ["particle"] + ["gas"] * has_baryons:

    fig, axs = bpl.subplots(ncols=3, figsize=[20, 7])
    axs = axs.flatten()

    max_vel = np.max([np.max(data.get_data("v{}".format(dim), "all", kind))
                      for dim in ["x", "y", "z"]])
    min_vel = np.min([np.min(data.get_data("v{}".format(dim), "all", kind))
                      for dim in ["x", "y", "z"]])
    n_bins = 50
    bin_size = (max_vel - min_vel) / n_bins

    for idx in range(10):
        try:
            level = "zoom_{}".format(idx)
            vx = data.get_data("vx", level, kind)
            vy = data.get_data("vy", level, kind)
            vz = data.get_data("vz", level, kind)
        except ValueError:  # ran out of levels
            break

        axs[0].hist(vx, histtype="step", rel_freq=True, bin_size=bin_size, color=bpl.color_cycle[idx], label="Level {}".format(idx))
        axs[1].hist(vy, histtype="step", rel_freq=True, bin_size=bin_size, color=bpl.color_cycle[idx], label="Level {}".format(idx))
        axs[2].hist(vz, histtype="step", rel_freq=True, bin_size=bin_size, color=bpl.color_cycle[idx], label="Level {}".format(idx))


    axs[0].add_labels("{} Velocity X [code units]".format(kind.title()), "Relative Frequency")
    axs[1].add_labels("{} Velocity Y [code units]".format(kind.title()), "Relative Frequency")
    axs[2].add_labels("{} Velocity Z [code units]".format(kind.title()), "Relative Frequency")

    axs[1].legend(loc="upper left")

    plot_base = ic_dir.split(os.sep)[-1]
    plot_name = os.path.abspath("./plots/velocity_check_single_{}_{}.png".format(kind, plot_base))
    print(" - saving plot:\n   - {}".format(plot_name))
    fig.savefig(plot_name)