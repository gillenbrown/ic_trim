import sys, os
import numpy as np
from collections import defaultdict
from tqdm import tqdm as tqdm
import betterplotlib as bpl
bpl.presentation_style()

sys.path.append("/u/home/gillenb/code/mine/ic_trim/")

import utils

root_ic_dir = os.path.abspath(sys.argv[1])
original_ic_dir = os.path.abspath(sys.argv[2])
replica_ic_dir = os.path.abspath(sys.argv[3])
final_ic_dir = os.path.abspath(sys.argv[4])

ic_files = os.listdir(final_ic_dir)
has_baryons = "music_H.md" in ic_files

plot_base = final_ic_dir.split(os.sep)[-1]

kinds = ["particle"] + ["gas"] * has_baryons

# assign colors to each of these 
color_root = bpl.color_cycle[1]
color_original = bpl.color_cycle[3]
color_replica = bpl.color_cycle[0]
color_final = bpl.almost_black
# ==============================================================================
#
# Read data
# 
# ==============================================================================
print()
data_root = utils.ARTData(root_ic_dir, has_baryons)

print()
data_original = utils.ARTData(original_ic_dir, has_baryons)
header_original = data_original.header

print()
data_replica = utils.ARTData(replica_ic_dir, has_baryons)

print()
data_final = utils.ARTData(replica_ic_dir, has_baryons)

# ==============================================================================
#
# convenience functions
# 
# ==============================================================================
def cell_size(header, level): 
    if level == "root":
        n = header.NGRIDC
    elif level == "max_zoom":
        n = header.NROWC
    elif type(level) == int:
        n = header.NGRIDC * 2**level
    else:
        raise ValueError("level not recognized.")
    return header.extras[74] / n

def get_data_wrap(data_obj, field, subset, kind):
    data = data_obj.get_data(field, subset, kind)
    # positions need to be shifted to the new coordinate system, velocities dont
    if "v" not in field and field != "pma":
        return to_final_root(data, data_obj.header)
    else:
        return data

def to_final_root(x, header): 
    l_this = header.extras[74]
    n_this = header.NGRIDC
    l_ref = data_root.header.extras[74]
    n_ref = data_root.header.NGRIDC
    return utils.arbitrary_shift(x, l_this, l_ref, n_this, n_ref)

# ==============================================================================
# 
# figure out the new refinement structure
# 
# ==============================================================================
# next we need to figure out how many levels of refinement we need. This is to
# figure out how many levels of the refined particles to keep. To do this, we
# find the level at which the cells size from the original zoom IC match the
# cell size of the new root grid
root_cell_size_new_root = cell_size(data_root.header, "root")
min_cell_size_original_zoom = cell_size(data_original.header, "max_zoom")
# here i will be the number of refined levels to keep, meaning i=0 is the most
# refined level
for zoom_levels in range(10):
    factor = 2**zoom_levels
    new_cell_size_original_zoom = min_cell_size_original_zoom * factor
    if np.isclose(new_cell_size_original_zoom, root_cell_size_new_root):
        # we have found the level that matches the new root grid size. 
        break
else: # no break
    raise ValueError("The grids of old and new ICs don't align! Check that there"
                     "is a factor of two between the two sim sizes.")
# we want to keep refined particles that have the same cell size as the 
# new root grid. This means that the number of levels to keep is always
# one more than the difference in levels. However, the one exception to this is
# if the last level to keep is the same as the old root grid. Then we don't want
# to keep those old root grid particles. So check for that. Print some 
# debugging info to the user while we're at it
root_cell_size_original_zoom = cell_size(data_original.header, "root")
if np.isclose(new_cell_size_original_zoom, root_cell_size_original_zoom):
    levels_to_keep = zoom_levels
    print("\n - Keeping {} levels of refined particles: ".format(levels_to_keep))
    print("   - The root grids of the old and new are the same size, and we do not keep the old root cells.")
    print("   - {} levels of particles on more refined levels".format(levels_to_keep))
else:
    levels_to_keep = zoom_levels + 1
    print(" - Keeping {} levels of refined particles: ".format(levels_to_keep))
    print("   - 1 for the particles on the same level as the new root grid.")
    print("   - {} for the particles on more refined levels".format(levels_to_keep - 1))

# ==============================================================================
#
# Correcting offset
# 
# ==============================================================================
# First get the median location of the zoom particles
median_original = [np.median(get_data_wrap(data_original, x, "zoom_0", "particle"))
                   for x in ["x", "y", "z"]]
median_replica = [np.median(get_data_wrap(data_replica, x, "zoom_0", "particle"))
                  for x in ["x", "y", "z"]] 

print("   - Original zoom region median:           {:.5f}, {:.5f}, {:.5f}".format(*median_original))
print("   - Replica  zoom region median:           {:.5f}, {:.5f}, {:.5f}".format(*median_replica))

diffs = [median_replica[i] - median_original[i] for i in range(3)]

# these diffs are in units of the new root grid, rather than the original root
# grid. The ratio between those two is the ratio of the cell sizes
scale = cell_size(data_root.header, "root") / cell_size(data_original.header, "root")
diffs = [diff * scale for diff in diffs]
# then apply this offset to the original particles
data_original.fix_offset(*diffs)

# then compare the new median
original_median_corrected = [np.median(get_data_wrap(data_original, x, "zoom_0", "particle"))
                             for x in ["x", "y", "z"]] 
print("   - Corrected original zoom region median: {:.5f}, {:.5f}, {:.5f}".format(*original_median_corrected))

# ==============================================================================
#
# making the mask
# 
# ==============================================================================
# print(" - making the particle masks")
# level_masks = defaultdict(dict)

n_levels_original = np.log2(header_original.NROWC) - np.log2(header_original.NGRIDC)
assert int(n_levels_original) == n_levels_original
n_levels_original = int(n_levels_original)
# iterate through all the levels in the original, other than the max zoom,
# and inluding the root level. Here idx 0 is the max zoom level, and idx
# n_levels_original is the root level
n_plot_levels = range(1, n_levels_original+1)

# for kind in kinds:
#     
#     for idx in range(1, n_levels_original+1):
#         level = "zoom_{}".format(idx)
#         level_masks[kind][idx] = utils.make_mask(mask_size, 0, 
#                                                  get_data_wrap(data_original, "x", level, "particle"), 
#                                                  get_data_wrap(data_original, "y", level, "particle"), 
#                                                  get_data_wrap(data_original, "z", level, "particle"),
#                                                  fail=False)

# ==============================================================================
#
# checking particles against it
# 
# ==============================================================================
# print(" - checking final particles against the masks")

# root_in_mask_idx = defaultdict(lambda: defaultdict(list))  #weird syntax
# for kind in kinds:
#     f_x = get_data_wrap(data_final, "x", "zoom_from_1", "particle")
#     f_y = get_data_wrap(data_final, "y", "zoom_from_1", "particle")
#     f_z = get_data_wrap(data_final, "z", "zoom_from_1", "particle")
#     for idx in level_masks[kind]:
#         for idx_f in tqdm(range(len(f_x))):
#             cell_edges = utils.find_cell_edges(f_x[idx_f], f_y[idx_f], f_z[idx_f], 0)
#             idx_i = int(round(cell_edges[0], 0))
#             idx_j = int(round(cell_edges[2], 0))
#             idx_k = int(round(cell_edges[4], 0))

#             if level_masks[kind][idx][idx_i][idx_j][idx_k] > 0.5:
#                 root_in_mask_idx[kind][idx].append(idx_f)

mask_size = int(np.ceil(np.max(get_data_wrap(data_final, "x", "all", "particle"))))
def do_level_masking(x_o, y_o, z_o, x_f, x_y, z_f, kind):
    # this will only be called on the new root grid or less refined
    # so we can make the mask on the root grid level
    mask = np.zeros((mask_size, mask_size, mask_size))
    
    # for the root grid, we only want to compare the original ones that are 
    # actually in the new box, not outside it. We can do this by seeing if 
    # they're in the mask box as we create it
    idx_orig = []
    for idx_o in tqdm(range(len(x_o))):
        cell_edges = find_cell_edges(x_o[idx_o], y_o[idx_o], z_o[idx_o], 0)
        
        idx_i = int(round(cell_edges[0], 0))
        idx_j = int(round(cell_edges[2], 0))
        idx_k = int(round(cell_edges[4], 0))
        try:
            mask[idx_i][idx_j][idx_k] = 1.0
            idx_orig.append(idx_o)
        except IndexError:
            # this particle is outside the box
            continue

    # then check against this mask
    idx_final = []
    for idx_f in tqdm(range(len(x_f))):
        cell_edges = utils.find_cell_edges(x_f[idx_f], y_f[idx_f], z_f[idx_f], 0)
        idx_i = int(round(cell_edges[0], 0))
        idx_j = int(round(cell_edges[2], 0))
        idx_k = int(round(cell_edges[4], 0))

        if mask[idx_i][idx_j][idx_k] > 0.5:
            final_idxs.append(idx_f)

    return idx_orig, idx_final

# ==============================================================================
#
# start comparing velocities
# 
# ==============================================================================
def bin_size(data_1, data_2, n_bins):
    overall_max = np.max([np.max(data_1), np.max(data_2)])
    overall_min = np.max([np.min(data_1), np.min(data_2)])
    return (overall_max - overall_min) / n_bins

for kind in kinds:
    fig, axs = bpl.subplots(ncols=3, nrows=len(n_plot_levels), 
                            figsize=[25, 5*len(n_plot_levels)])

    for idx, ax_row in zip(n_plot_levels, axs):
        level = "zoom_{}".format(idx)
        orig_vx = get_data_wrap(data_original, "vx", level, kind)
        orig_vy = get_data_wrap(data_original, "vy", level, kind)
        orig_vz = get_data_wrap(data_original, "vz", level, kind)

        # see if we can just compare the same level
        try:
            # will raise ValueError if this level is not present
            this_vx = get_data_wrap(data_final, "vx", level, kind)
            this_vy = get_data_wrap(data_final, "vy", level, kind)
            this_vz = get_data_wrap(data_final, "vz", level, kind)
            # but if we're on the root grid, we want to do the masking
            if len(this_vx) == len(get_data_wrap(data_final, "vx", "root", kind)):
                raise ValueError
        except ValueError:
            print(" - making mask for zoom level {}".format(idx))
            o_x = get_data_wrap(data_original, "x", level, kind)
            o_y = get_data_wrap(data_original, "y", level, kind)
            o_z = get_data_wrap(data_original, "z", level, kind)
            f_x = get_data_wrap(data_final, "x", "zoom_from_1", kind)
            f_y = get_data_wrap(data_final, "y", "zoom_from_1", kind)
            f_z = get_data_wrap(data_final, "z", "zoom_from_1", kind)

            f_vx = get_data_wrap(data_final, "vx", "zoom_from_1", kind)
            f_vy = get_data_wrap(data_final, "vy", "zoom_from_1", kind)
            f_vz = get_data_wrap(data_final, "vz", "zoom_from_1", kind)

            idx_orig, idx_final = do_level_masking(o_x, o_y, o_z, f_x, f_y, f_z, kind)

            this_vx = f_vx[idx_final]
            this_vy = f_vy[idx_final]
            this_vz = f_vz[idx_final]

            orig_vx = orig_vx[idx_orig]
            orig_vy = orig_vy[idx_orig]
            orig_vz = orig_vz[idx_orig]
        
        n_bins = 50
        bin_size_x = bin_size(this_vx, orig_vx, n_bins)
        bin_size_y = bin_size(this_vy, orig_vy, n_bins)
        bin_size_z = bin_size(this_vz, orig_vz, n_bins)
        
        ax_row[0].hist(this_vx, histtype="step", rel_freq=True, bin_size=bin_size_x, hatch="\\", color=color_final,    label="Final")
        ax_row[0].hist(orig_vx, histtype="step", rel_freq=True, bin_size=bin_size_x, hatch="/",  color=color_original, label="Original")
        ax_row[1].hist(this_vy, histtype="step", rel_freq=True, bin_size=bin_size_y, hatch="\\", color=color_final,    label="Final")
        ax_row[1].hist(orig_vy, histtype="step", rel_freq=True, bin_size=bin_size_y, hatch="/",  color=color_original, label="Original")
        ax_row[2].hist(this_vz, histtype="step", rel_freq=True, bin_size=bin_size_z, hatch="\\", color=color_final,    label="Final")
        ax_row[2].hist(orig_vz, histtype="step", rel_freq=True, bin_size=bin_size_z, hatch="/",  color=color_original, label="Original")
        
        if idx == 0:
            title = "Max Refined Level"
        elif len(orig_vx) == len(get_data_wrap(data_original, "vx", "root", "particle")):
            title = "Old Root Level"
        else:
            title = "Intermediate Layer {}".format(idx)
        
        ax_row[0].add_labels("Velocity X [code units]", "Relative Frequency")
        ax_row[1].add_labels("Velocity Y [code units]", "Relative Frequency", title)
        ax_row[2].add_labels("Velocity Z [code units]", "Relative Frequency")
        
        ax_row[0].easy_add_text("Original N={:,}\nFinal N={:,}".format(len(orig_vx), len(this_vx)), 
                                "upper left", fontsize=16)
        
        ax_row[1].legend(loc="upper left")

    
    plot_name = os.path.abspath("./plots/velocity_level_comparison_{}_{}.png".format(kind, plot_base))
    fig.savefig(plot_name)
    print(" - saving debug plots:")
    print("   - {}".format(plot_name))
