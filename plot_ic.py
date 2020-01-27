import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from celluloid import Camera
import betterplotlib as bpl
from matplotlib import cm
from matplotlib import colors
import cmocean
import colorcet as cc
from tqdm import tqdm

bpl.presentation_style()

def trim_box(density, new_length):
    # This will take the central `new_length` values for all directions
    # calculate how much to trim off each end
    # This assumes we have a cube
    trim_end = (density.shape[0] - new_length) / 2.0
    trim_end_int = int(trim_end)
    assert trim_end == trim_end_int  # check for non-integer values

    idx_start = trim_end_int
    idx_end = density.shape[0] - trim_end_int

    return density[idx_start:idx_end,idx_start:idx_end,idx_start:idx_end]

# parse the command line arguments
ic_file_loc = os.path.abspath(sys.argv[1])
user_field = sys.argv[2]
level = int(sys.argv[3])
if len(sys.argv) == 4:
    central_power = level
else:
    central_power = int(sys.argv[4])
cut_length = 2**central_power

if len(sys.argv) > 5:
    raise ValueError("Too many arguments")

# determine if the dataset has baryons or not
hf = h5py.File(ic_file_loc)
baryons = "level_000_BA_rho" in hf.keys()
hf.close()

if baryons:
    print(" - Baryons detected")
else:
    print(" - No baryons detected")

if user_field == "all":
    if baryons:
        fields = ["BA_potential", "BA_rho", "BA_vx", "BA_vy", "BA_vz"]
    else:
        fields = []
    fields += ["DM_rho", "DM_dx", "DM_dy", "DM_dz", "DM_vx", "DM_vy", "DM_vz"]
else:
    fields = [user_field]

for field in fields:
    print("\n - Reading field {} on level {}".format(field, level))
    # Read the DM density at the level requested
    hf = h5py.File(ic_file_loc)
    values = np.array(hf['level_{:03d}_{}'.format(level, field)])
    hf.close()

    print(" - Trimming cube from  2^{} to 2^{}".format(level, central_power))
    # Trim it to the size requested
    plot_values = trim_box(values, cut_length)
    # normalize the positions and velocities. They are all in units where the box
    # size is one, so multiplying by the grid size will get it into code units
    if "DM_d" in field or "DM_v" in field:
        plot_values *= 2**level

    print(" - Making visualization")

    # Then make the visualization
    fig, axs = bpl.subplots(nrows=1, ncols=2, gridspec_kw={"width_ratios":[20, 1]})
    ax = axs[0]
    cax = axs[1]
    ax.equal_scale()
    ax.add_labels("X [code units]", "Y [code units]")
    
    # vmax determined experimentally with several outputs, although I don't 
    # understand what baryons are doing
    if "DM_d" in field:
        cmap = cmocean.cm.delta
        vmax = 0.2 * 2**(central_power - 5)
    elif "DM_v" in field:
        cmap = cmocean.cm.balance
        vmax = 125 * 2**(central_power - 5)
    elif "BA_v" in field:
        cmap = cmocean.cm.curl
        vmax = np.max(np.abs(plot_values))
    elif "DM_rho" == field:
        cmap = cmocean.cm.tarn_r
        vmax = 0.35
    elif "BA_rho" == field:
        cmap = cmocean.cm.diff
        vmax = np.max(np.abs(plot_values))
    elif "BA_potential" == field:
        cmap = cc.m_gwv
        vmax = np.max(np.abs(plot_values))
    else:
        raise ValueError("field {} not recognized!".format(field))
        
    vmin = -vmax  # ensure 0 is in the middle, which is nice for diverging colormaps
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label(field.replace("_", " "))

    camera = Camera(fig)

    for z in tqdm(range(plot_values.shape[2])):
        ax.pcolormesh(plot_values[:,:,z], vmin=vmin, vmax=vmax, cmap=cmap)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        ax.easy_add_text("Z = {} code units".format(z), "upper left")
        cbar.set_label(field.replace("_", " "))
        camera.snap()
        
    anim = camera.animate()

    # then get the name to save it. We'll use the name of the output file, since 
    # it is descriptive
    plot_name = ic_file_loc.split(os.sep)[-1].replace(".hdf5", "")
    if level != central_power:
        savename = "../plots/" + plot_name + "_viewlevel_{:02d}_trim_{:02d}_{}.mp4".format(level, central_power, field)
    else:
        savename = "../plots/" + plot_name + "_viewlevel_{:02d}_{}.mp4".format(level, field)
    savename = os.path.abspath(savename)

    print(" - Saving visualization to:\n   {}".format(savename))
    anim.save(savename, codec="mpeg4")
