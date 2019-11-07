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
from tqdm import tqdm

bpl.presentation_style()

# parse the command line arguments
ic_file_loc = os.path.abspath(sys.argv[1])
level = int(sys.argv[2])
if len(sys.argv) == 3:
    central_power = level
else:
    central_power = int(sys.argv[3])
cut_length = 2**central_power

if len(sys.argv) > 4:
    raise ValueError("Too many arguments")

# Read the DM density at the level requested
hf = h5py.File(ic_file_loc)
dm_density = np.array(hf['level_{:03d}_DM_rho'.format(level)])
hf.close()

# Trim it to the size requested
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

dm_density = trim_box(dm_density, cut_length)

# Then make the visualization
fig, axs = bpl.subplots(nrows=1, ncols=2, gridspec_kw={"width_ratios":[20, 1]})
ax = axs[0]
cax = axs[1]
ax.equal_scale()
ax.add_labels("X [code units]", "Y [code units]")

camera = Camera(fig)

vmax = np.max(np.abs(dm_density.flatten()))
vmin = -vmax
norm = colors.Normalize(vmin=vmin, vmax=vmax)
cmap = cmocean.cm.curl  
fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

for z in tqdm(range(dm_density.shape[2])):
    ax.pcolormesh(dm_density[:,:,z], vmin=vmin, vmax=vmax, cmap=cmap)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    ax.easy_add_text("Z = {} code units".format(z), "upper left")
    camera.snap()
    
anim = camera.animate()

# then get the name to save it. We'll use the name of the directory holding 
# the output file, since it is descriptive
dirname = os.path.dirname(ic_file_loc).split(os.sep)[-1]
if level != central_power:
    savename = "./plots/" + dirname + "_level_{:02d}_trim_{:02d}.mp4".format(level, central_power)
else:
    savename = "./plots/" + dirname + "_level_{:02d}.mp4".format(level)
savename = os.path.abspath(savename)

anim.save(savename, codec="mpeg4")
