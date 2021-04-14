import os
import sys
import copy
import utils
import numpy as np
from tqdm import tqdm as tqdm

import betterplotlib as bpl
bpl.presentation_style()

# This code takes 4 command line arguments, all directories. In order, they are:
# 1 - Directory containing the initial condition files (in ART format) that has
#     the particles to become the new root grid particles in the new IC.
#     Will be referred to as "root" throughout this code
# 2 - Directory containing the initial condition files (in ART format) that has
#     the zoom particles that will become the zoom particles in the new IC.
#     Will be referred to as "original" throughout this code
# 3 - Directory containing the initial condition files (in ART format) of the
#     zoom IC that was used to create the new root grid. Another way of 
#     describing this is: the new root grid particles were created by taking a 
#     white noise cube and trimming it, then running that through MUSIC. This
#     white noise cube was made by running a hacked version of MUSIC on a 
#     IC config file. If we run the regular version of MUSIC on that same IC 
#     config file, it would create an IC that is what we pass in here. This is
#     needed because there is an offset in the refined region between ICs with
#     different root grid levels, and we need to shift the particles from the 
#     IC in argument 2 to match the coordinate system of the IC in argument 1.
#     Will be referred to as "replica" throughout this code.
# 4 - The directory to write the new hybrid IC to.
#     Will be referred to as "final" throughout this code

root_ic_dir = os.path.abspath(sys.argv[1])
original_ic_dir = os.path.abspath(sys.argv[2])
replica_ic_dir = os.path.abspath(sys.argv[3])
final_ic_dir = os.path.abspath(sys.argv[4])

if len(sys.argv) > 5:
    raise ValueError("Too many arguments")

# assign colors to each of these 
color_root = bpl.color_cycle[1]
color_original = bpl.color_cycle[3]
color_replica = bpl.color_cycle[0]
color_final = bpl.almost_black
# ==============================================================================
# 
# check whether we have baryons
# 
# ==============================================================================
# Check if we have baryons or not. If we do not, the files will be of the form
# music_D.extension, while if we do, they will be of the form 
# "music_H.extension"
# I'll check that all ICs either have baryons or don't
has_baryons = None
for input_dir in [root_ic_dir, original_ic_dir, replica_ic_dir]:
    # see if the baryon file is in the directory
    ic_files = os.listdir(input_dir)
    this_baryons = "music_H.md" in ic_files
    # I have an initial unset flag, that will get set the first time
    if has_baryons is None:
        has_baryons = this_baryons
    # then we check that this directory matches the others
    if has_baryons != this_baryons:
            raise ValueError("All ICs need to be hydro or N-body")

kinds = ["particle"] + ["gas"] * has_baryons
# ==============================================================================
# 
# read headers
# 
# ==============================================================================
print(" - reading headers")
header_root = utils.ARTHeader(root_ic_dir, has_baryons)
header_original = utils.ARTHeader(original_ic_dir, has_baryons)
header_replica = utils.ARTHeader(replica_ic_dir, has_baryons)

# ==============================================================================
# 
# Validate headers
# 
# ==============================================================================
# check some things about the headers to verify the users choices
# the root grid cell size of new_root and new_zoom should be the same
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

if cell_size(header_root, "root") != cell_size(header_replica, "root"):
    raise ValueError("The new root grid IC and corresponding zoom IC need the "
                     "same root grid cell size!")

print("\n - comparing root header to original header")
utils.compare_headers(header_root, header_original)
print("\n - comparing root header to replica header")
utils.compare_headers(header_root, header_replica)
print("\n - comparing replica header to original header")
utils.compare_headers(header_replica, header_original)
print()

# do a couple checks to validate the data
# the new root grid should be equal to or smaller than the old root grid
if cell_size(header_root, "root") > cell_size(header_original, "root"):
    raise ValueError("The new root grid is larger than the old one! What are you doing?")

# ==============================================================================
# 
# figure out the new refinement structure
# 
# ==============================================================================
# next we need to figure out how many levels of refinement we need. This is to
# figure out how many levels of the refined particles to keep. To do this, we
# find the level at which the cells size from the original zoom IC match the
# cell size of the new root grid
root_cell_size_new_root = cell_size(header_root, "root")
min_cell_size_original_zoom = cell_size(header_original, "max_zoom")
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
root_cell_size_original_zoom = cell_size(header_original, "root")
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
original_subset = "zoom_to_{}".format(levels_to_keep-1)

# ==============================================================================
#
# Some wrapper functions
# 
# ==============================================================================

# wrapper function for arbitrary_shift to turn coordinates into the same system
# as the new root grid coordinate particles. We use deepcopy so we can delete the
# object and hopefully have it leave memory, since there won't be any references
# to it.
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
    l_ref = header_root.extras[74]
    n_ref = header_root.NGRIDC
    return utils.arbitrary_shift(x, l_this, l_ref, n_this, n_ref)

# ==============================================================================
#
# Read data
# 
# ==============================================================================
print()
data_root = utils.ARTData(root_ic_dir, has_baryons)
# we need to keep everything here, since we will use the particles and gas 
# in the final root region

print()
data_original = utils.ARTData(original_ic_dir, has_baryons)
# we need to keep everything here, since we will use the particles and gas 
# in the final zoom region

print()
data_replica = utils.ARTData(replica_ic_dir, has_baryons)
# all we need here are the particles, so we can delete all the baryon data
# But we'll do a debugging plot first, so keep them for now

# ==============================================================================
#
# Fix offset between old and new zoom region
# 
# ==============================================================================
print("\n - calculating offset between old and new zoom region")
def plot_offset(ax, center, label, x1, y1, z1, x2, y2, z2):
    """1 is original, 2 is new"""
    # first determine what the plots limits should be. To do this I'll find the 
    # location of the maximal zoom particle
    dz = 2.0 / (zoom_levels**2)
    dx = 20.0 / (zoom_levels**2)
    
    # our particle of interest is t
    limits = [center[0] - dx, center[0] + dx,
              center[1] - dx, center[1] + dx, 
              center[2] - dz, center[2] + dz]

    utils.plot_one_sim(x1, y1, z1, *limits, ax, s=25, zorder=1, 
                       c=color_original, label="Original")
    utils.plot_one_sim(x2, y2, z2, *limits, ax, s=5, zorder=2, 
                       c=color_replica, label="Replica")

    ax.equal_scale()
    ax.add_labels("X [New root grid code units]", "Y [New root grid code units]",
                  "{:.3f} < Z < {:.3f}".format(limits[4], limits[5]))
    ax.set_limits(limits[0], limits[1], limits[2], limits[3])
    ax.legend(bbox_to_anchor=(0.97,0.5), loc="center left", title=label)

# First get the median location of the zoom region. Use cells if we have them
if has_baryons:
    median_original = [np.median(get_data_wrap(data_original, x, "zoom_0", "gas"))
                       for x in ["x", "y", "z"]]
    median_replica = [np.median(get_data_wrap(data_replica, x, "zoom_0", "gas"))
                      for x in ["x", "y", "z"]]    
else: 
    median_original = [np.median(get_data_wrap(data_original, x, "zoom_0", "particle"))
                       for x in ["x", "y", "z"]]
    median_replica = [np.median(get_data_wrap(data_replica, x, "zoom_0", "particle"))
                      for x in ["x", "y", "z"]] 

print("   - Original zoom region median:           {:.5f}, {:.5f}, {:.5f}".format(*median_original))
print("   - Replica  zoom region median:           {:.5f}, {:.5f}, {:.5f}".format(*median_replica))

diffs = [median_replica[i] - median_original[i] for i in range(3)]

# start the plot before we correct the data
fig, axs = bpl.subplots(ncols=2, figsize=[15, 8])
idx = np.argmin(get_data_wrap(data_replica, "x", "zoom_0", "particle"))
plot_center = [get_data_wrap(data_replica, x, "zoom_0", "particle")[idx] 
               for x in ["x", "y", "z"]]
plot_offset(axs[0], plot_center, "Before",
            get_data_wrap(data_original, "x", "all", "particle"), 
            get_data_wrap(data_original, "y", "all", "particle"), 
            get_data_wrap(data_original, "z", "all", "particle"),
            get_data_wrap(data_replica,  "x", "all", "particle"), 
            get_data_wrap(data_replica,  "y", "all", "particle"), 
            get_data_wrap(data_replica,  "z", "all", "particle"))

# these diffs are in units of the new root grid, rather than the original root
# grid. The ratio between those two is the ratio of the cell sizes
scale = cell_size(header_root, "root") / cell_size(header_original, "root")
diffs = [diff * scale for diff in diffs]
# then apply this offset to the original particles
data_original.fix_offset(*diffs)

# then compare the new median
if has_baryons:
    original_median_corrected = [np.median(get_data_wrap(data_original, x, "zoom_0", "gas"))
                                 for x in ["x", "y", "z"]] 
else:
    original_median_corrected = [np.median(get_data_wrap(data_original, x, "zoom_0", "particle"))
                                 for x in ["x", "y", "z"]] 
print("   - Corrected original zoom region median: {:.5f}, {:.5f}, {:.5f}".format(*original_median_corrected))

# then plot the corrected values
plot_offset(axs[1], plot_center, "After",
            get_data_wrap(data_original, "x", "all", "particle"), 
            get_data_wrap(data_original, "y", "all", "particle"), 
            get_data_wrap(data_original, "z", "all", "particle"),
            get_data_wrap(data_replica,  "x", "all", "particle"), 
            get_data_wrap(data_replica,  "y", "all", "particle"), 
            get_data_wrap(data_replica,  "z", "all", "particle"))

# and save our plot
# # parse the final directory to get a plot name
# abspath was used to get final_ic_dir, and it does not add os.sep to the end,
# and strips it if present, so we do not need to worry about a trailing os.sep
plot_base = final_ic_dir.split(os.sep)[-1]
plot_name = os.path.abspath("./plots/shift_fix_{}.png".format(plot_base))
print("   - making debug plot:\n     - {}".format(plot_name))
fig.savefig(plot_name)

# ==============================================================================
#
# Plot positions of two sims to validate transformation
# 
# ==============================================================================
# then make a plot showing the positions of the new and old particles, so I
# can validate the coordinate transformation
plot_name = os.path.abspath("./plots/locations_debug_{}.png".format(plot_base))
print("   - making debug plot:\n     - {}".format(plot_name))
fig, ax = bpl.subplots()
z_center = int(header_root.NGRIDC / 2.0)
x_limit_low = 0
x_limit_high = header_root.NGRIDC + 2
limits = [x_limit_low, x_limit_high, x_limit_low, x_limit_high, 
          z_center, z_center + 2]

utils.plot_one_sim(get_data_wrap(data_original, "x", original_subset, "particle"), 
                   get_data_wrap(data_original, "y", original_subset, "particle"),
                   get_data_wrap(data_original, "z", original_subset, "particle"), 
                   *limits, ax, s=1, zorder=2, c=color_original, label="Zoom")
utils.plot_one_sim(get_data_wrap(data_root, "x", "all", "particle"), 
                   get_data_wrap(data_root, "y", "all", "particle"),
                   get_data_wrap(data_root, "z", "all", "particle"), 
                   *limits, ax, s=5, zorder=1, c=color_root, label="Root")
ax.equal_scale()
ax.add_labels("X [New Root code units]", "Y [New Root code units]")
ax.set_limits(limits[0], limits[1], limits[2], limits[3])
ax.easy_add_text("{} < Z < {}".format(limits[4], limits[5]), "upper left")
ax.legend(bbox_to_anchor=(0.97,0.5), loc="center left")
fig.savefig(plot_name)

# ==============================================================================
#
# Correct velocities or original header too
# 
# ==============================================================================
print("\n - calculating velocity scaling between old and replica")
# this will find to correct the velocities from the original dataset to be like
# the replica. They are different because of different root grid sizes, since 
# velocity is distance / time. The root grid size is the distance unit, while 
# time units are just based on cosmology. So if the original cell sizes are 
# twice as big, then the current velocity is a factor of smaller than the 
# replica velocities.
scale_vel = cell_size(header_original, "root") / cell_size(header_replica, "root")

for kind in kinds:
    # plot the velocities first, so we can compare them later 
    fig, axs = bpl.subplots(ncols=3, figsize=[20, 7])
    axs = axs.flatten()
    bin_size = 0.002

    # make some print statements
    original_median = [np.median(get_data_wrap(data_original, v, "zoom_0", kind))
                       for v in ["vx", "vy", "vz"]] 
    replica_median = [np.median(get_data_wrap(data_replica, v, "zoom_0", kind))
                         for v in ["vx", "vy", "vz"]] 
    print("   - Original zoom region velocity median {:<9}           {:.5f}, {:.5f}, {:.5f}".format(kind, *original_median))
    print("   - Replica zoom region velocity median {:<9}            {:.5f}, {:.5f}, {:.5f}".format(kind, *replica_median))

    for dim, ax in zip(["x", "y", "z"], axs):
        v = "v{}".format(dim)
        ax.hist(data_replica.get_data(v, "zoom_0", kind), color=color_replica, 
                bin_size=bin_size, hatch="o", histtype="step", label="Replica")
        ax.hist(data_original.get_data(v, "zoom_0", kind), color=color_original, 
                bin_size=bin_size, hatch="\\", histtype="step", label="Original")

        # then correct the velocity units
        if kind == "gas":
            data_original.gas_data[v] *= scale_vel
        else:
            data_original.dm_data[v] *= scale_vel

        # then plot the corrected values
        ax.hist(data_original.get_data(v, "zoom_0", kind), color=color_original, 
                bin_size=bin_size, hatch="//", histtype="step", label="Original - Corrected")
        ax.add_labels("{} Velocity {} [code units]".format(kind.title(), dim), "Number")
        
    axs[1].legend()

    # make some print statements
    original_median_corrected = [np.median(get_data_wrap(data_original, v, "zoom_0", kind))
                         for v in ["vx", "vy", "vz"]] 
    print("   - Corrected original zoom region velocity median {:<9} {:.5f}, {:.5f}, {:.5f}".format(kind, *original_median_corrected))

    plot_name = os.path.abspath("./plots/velocity_fix_{}_{}.png".format(kind, plot_base))
    print("   - making debug plot:\n   - {}".format(plot_name))
    fig.savefig(plot_name)
# ==============================================================================
#
# And correct the baryon cell masses
# 
# ==============================================================================
if has_baryons:
    print("\n - calculating baryon cell mass scaling between old and replica")
    # pma is the mass in a cell - mass scales as length^3
    scale_pma = scale_vel**3
    
    # plot pma first, so we can compare them later 
    fig, ax = bpl.subplots()
    # the expected value is the minimum paricle mass divided by about 5 
    # (for baryon to DM ratio), we divide it further for resolution in the plot
    bin_size_replica = header_replica.wpart[0] / 500
    bin_size_original = header_original.wpart[0] / 500

    ax.hist(data_replica.get_data("pma", "zoom_0", "gas"), color=color_replica, 
            bin_size=bin_size_replica, hatch="o", histtype="step", label="Replica")
    ax.hist(data_original.get_data("pma", "zoom_0", "gas"), color=color_original, 
            bin_size=bin_size_original, hatch="\\", histtype="step", label="Original")

    # then correct the velocity units
    data_original.gas_data["pma"] *= scale_pma

    # then plot the corrected values. Use the replica bin size, since it should
    # be on that scale now
    ax.hist(data_original.get_data("pma", "zoom_0", "gas"), color=color_original, 
            bin_size=bin_size_replica, hatch="//", histtype="step", label="Original - Corrected")
    ax.add_labels("Cell Mass [code units]", "Number")
        
    ax.legend()
    plot_name = os.path.abspath("./plots/gas_mass_fix_{}.png".format(plot_base))
    print("   - making debug plot:\n   - {}".format(plot_name))
    fig.savefig(plot_name)

# Now that we don't need the replica baryon data, we can delete it
for field in list(data_replica.gas_data.keys()):
    data_replica.delete_data("gas", field)
# ==============================================================================
#
# Make the mask
# 
# ==============================================================================
# make one mask for gas cells, one for particles. I think this is the best thing
# to do because large DM displacements can take particles outside their host 
# cells (at small grid sizes). Doing it separately will help make sure both the
# gas and particles from the original sim are included.
print(" - making the particle mask")
mask_size = int(np.ceil(np.max(get_data_wrap(data_root, "x", "all", "particle"))))
mask_part = utils.make_mask(mask_size, 0, 
                            get_data_wrap(data_original, "x", original_subset, "particle"), 
                            get_data_wrap(data_original, "y", original_subset, "particle"), 
                            get_data_wrap(data_original, "z", original_subset, "particle"))
if has_baryons:
    print(" - making the gas mask")
    mask_gas = utils.make_mask(mask_size, 0, 
                               get_data_wrap(data_original, "x", original_subset, "gas"), 
                               get_data_wrap(data_original, "y", original_subset, "gas"), 
                               get_data_wrap(data_original, "z", original_subset, "gas"))

# ==============================================================================
#
# Check new root against the mask
# 
# ==============================================================================
print(" - checking new root particles against the mask")
# get the particles first so we don't have to access each iteration
p_x = get_data_wrap(data_root, "x", "all", "particle")
p_y = get_data_wrap(data_root, "y", "all", "particle")
p_z = get_data_wrap(data_root, "z", "all", "particle")
g_x = get_data_wrap(data_root, "x", "all", "gas")
g_y = get_data_wrap(data_root, "y", "all", "gas")
g_z = get_data_wrap(data_root, "z", "all", "gas")
idxs_write_part = []
idxs_replace_part = []
for idx_p in tqdm(range(len(p_x))):
    cell_edges = utils.find_cell_edges(p_x[idx_p], p_y[idx_p], p_z[idx_p], 0)
    idx_i = int(round(cell_edges[0], 0))
    idx_j = int(round(cell_edges[2], 0))
    idx_k = int(round(cell_edges[4], 0))
    
    if mask_part[idx_i][idx_j][idx_k] < 0.5:
        idxs_write_part.append(idx_p)
    else:
        idxs_replace_part.append(idx_p)

if has_baryons:
    print(" - checking new root gas cells against the mask")
    idxs_write_gas = []
    idxs_replace_gas = []
    for idx_g in tqdm(range(len(g_x))):
        cell_edges = utils.find_cell_edges(g_x[idx_g], g_y[idx_g], g_z[idx_g], 0)
        idx_i = int(round(cell_edges[0], 0))
        idx_j = int(round(cell_edges[2], 0))
        idx_k = int(round(cell_edges[4], 0))
        
        if mask_gas[idx_i][idx_j][idx_k] < 0.5:
            idxs_write_gas.append(idx_g)
        else:
            idxs_replace_gas.append(idx_g)

# ==============================================================================
#
# calculate the velocity offset between the old and new particles
# 
# ==============================================================================
print(" - Calculating the velocity offset")
for vel in ["vx", "vy", "vz"]:
    root_replace_vels = get_data_wrap(data_root, vel, "all", "particle")[idxs_replace_part]
    average_root_vel = np.mean(root_replace_vels)
    # Calculate the mass-weighted average velocity of the particles in the zoom 
    # region that we want to keep. Each successive layer is 8 times more massive 
    mv = 0
    weights = 0
    for level in range(levels_to_keep):
        weight = 8**level
        zoom_vel = get_data_wrap(data_original, vel, "zoom_{}".format(level), "particle")
        mv += np.sum(zoom_vel * weight)
        weights += weight * len(zoom_vel)
        print(level, weight)

    average_zoom_vel = mv / weights

    print("particle {}: new root={:.3f}, original={:.3f}".format(vel, average_root_vel, average_zoom_vel))
    vel_offset = average_zoom_vel - average_root_vel

    data_original.dm_data[vel] -= vel_offset

if has_baryons:
    for vel in ["vx", "vy", "vz"]:
        # Calculate the mass-weighted average velocity of the cells in the zoom 
        # region that we want to keep. Here we can just use the cell mass
        root_replace_vels = get_data_wrap(data_root, vel, "all", "gas")[idxs_replace_gas]
        root_replace_mass = get_data_wrap(data_root, "pma", "all", "gas")[idxs_replace_gas]
        
        zoom_vels = get_data_wrap(data_original, vel, original_subset, "gas")
        zoom_mass = get_data_wrap(data_original, "pma", original_subset, "gas")

        average_root_vel = np.average(root_replace_vels, weights=root_replace_mass)
        average_zoom_vel = np.average(zoom_vels, weights=zoom_mass)

        print("gas {}: new root={:.3f}, original={:.3f}".format(vel, average_root_vel, average_zoom_vel))
        vel_offset = average_zoom_vel - average_root_vel

        data_original.gas_data[vel] -= vel_offset

# ==============================================================================
#
# combine data for final output 
# 
# ==============================================================================
# Add the data to an ARTData class
data_final = utils.ARTData(None, None)

print(" - Creating final list of particles")
for field in data_final.dm_data:
    good_root = get_data_wrap(data_root, field,  "all", "particle")[idxs_write_part]
    good_zoom = get_data_wrap(data_original, field,  original_subset, "particle")
    data_final.dm_data[field] = np.concatenate([good_zoom, good_root])

if has_baryons:
    print(" - Creating final list of cells")
    for field in data_final.gas_data:
        good_root = get_data_wrap(data_root, field,  "all", "gas")[idxs_write_gas]
        good_zoom = get_data_wrap(data_original, field,  original_subset, "gas")
        data_final.gas_data[field] = np.concatenate([good_zoom, good_root])

# ==============================================================================
#
# Create final header
# 
# ==============================================================================
print(" - Creating final output files")
header_final = copy.deepcopy(header_root)
header_final.directory = final_ic_dir
# then add the header to the data object
data_final.header = header_final
# I need to change a few fields here
# NGRIDC is the root grid size. This is the same as the new root grid.
# NROWC is the effective resolution of the high res region. This is based on the
# number of refined levels
header_final.NROWC = header_final.NGRIDC * 2**zoom_levels

# nspecies is the number of DM species. This is also based on the levels
header_final.nspecies = 1 + zoom_levels

# partw is the mass of the highest resolution particle. I can figure this out
# by the levels. I take the maximum particle mass (root grid), and decrease it
# by 8^num_zoom_levels
header_final.partw = max(header_final.wpart) / (8**zoom_levels)

# wpart is the mass of particles from high res to low_res, (i.e. is starts at partw)
# It's a tuple, so I have to copy it to change it...
temp_wpart = list(header_final.wpart)
temp_wpart[0] = header_final.partw
for idx in range(1, len(temp_wpart)):
    if idx < header_final.nspecies:
        temp_wpart[idx] = 8 * temp_wpart[idx - 1]
    else:
        temp_wpart[idx] = 0
header_final.wpart = tuple(temp_wpart)

# lpart is the number of particles at each level. I kept track of the number of
# zoom particles
temp_lpart = list(header_final.lpart)
for idx in range(len(temp_lpart)):
    if idx == (header_final.nspecies - 1):  # root level
        # can't use get_data_wrap, since the underlying function depends on
        # lpart here, so we can't use it yet
        temp_lpart[idx] = len(data_final.dm_data["x"])
    elif idx < header_final.nspecies - 1:  # zoom particles
        # since we're using the ones from the old IC we can just use those lpart
        temp_lpart[idx] = header_original.lpart[idx]
    else:
        temp_lpart[idx] = 0

header_final.lpart = tuple(temp_lpart)

# extras[74] is the box size, which is already in the new_root header

print("\n - comparing final header to replica header")
utils.compare_headers(header_final, header_replica)
print("\n - comparing final header to root header")
utils.compare_headers(header_final, header_root)
print("\n - comparing final header to original header")
utils.compare_headers(header_final, header_original)
print()

# then write the particles and header
print(" - Writing final particle list")
if has_baryons:
    header_final.write_header(final_ic_dir + "/music_H.mdh")
    data_final.write_data(final_ic_dir + "/music_H.mdxv", "particle")
    data_final.write_data(final_ic_dir + "/music_H.md", "gas")
else:
    header_final.write_header(final_ic_dir + "/music_D.mdh")
    data_final.write_data(final_ic_dir + "/music_D.mdxv", "particle")

# ==============================================================================

# Make debugging plot

# ==============================================================================
def mask_plot(kind, savename):
    idx = np.argmin(get_data_wrap(data_original, "x", original_subset, kind))
    center = [get_data_wrap(data_original, "x", original_subset, kind)[idx], 
              get_data_wrap(data_original, "y", original_subset, kind)[idx], 
              get_data_wrap(data_original, "z", original_subset, kind)[idx]]
    dx = 10.0
    limits = [np.floor(center[0] - dx), np.floor(center[0] + dx),
              np.floor(center[1] - dx), np.floor(center[1] + dx), 
              np.floor(center[2]), np.ceil(center[2])]  # ensure integer z range

    fig, ax = bpl.subplots()

    zorder_final = 5
    zorder_original = 6
    zorder_root = 7
    if kind == "gas":
        marker = "s"
        size_final = 30
        size_original = 10
        size_root = 3
        label = " Cells" 
    else:
        marker = "o"
        size_final = 50
        size_original = 20
        size_root = 10
        label = " Particles"

    utils.plot_one_sim(get_data_wrap(data_final, "x", "all", kind),
                       get_data_wrap(data_final, "y", "all", kind),
                       get_data_wrap(data_final, "z", "all", kind),
                       *limits, ax, s=size_final, zorder=zorder_final, marker=marker, 
                       c=color_final, label="Final" + label)

    utils.plot_one_sim(get_data_wrap(data_original, "x", "all", kind),
                       get_data_wrap(data_original, "y", "all", kind),
                       get_data_wrap(data_original, "z", "all", kind),
                       *limits, ax, s=size_original, zorder=zorder_original, marker=marker, 
                       c=color_original, label="Original" + label)

    utils.plot_one_sim(get_data_wrap(data_root, "x", "all", kind),
                       get_data_wrap(data_root, "y", "all", kind),
                       get_data_wrap(data_root, "z", "all", kind),
                       *limits, ax, s=size_root, zorder=zorder_root, marker=marker, 
                       c=color_root, label="New Root" + label)

    for x in np.arange(limits[0], limits[1], 1):
        ax.axvline(x, lw=0.5)
    for y in np.arange(limits[2], limits[3], 1):
        ax.axhline(y, lw=0.5)

    if kind == "gas":
        mask = mask_gas
    else:
        mask = mask_part
    d_check = 0.1
    eps = 1E-10
    mask_color = "0.92"
    mask_labeled = False
    for x_min in tqdm(np.arange(limits[0], limits[1], 1)):
        x_max = x_min + 1
        for y_min in np.arange(limits[2], limits[3], 1):
            y_max = y_min + 1
            
            cs = []
            for x in np.arange(x_min+eps, x_max, d_check):
                for y in np.arange(y_min+eps, y_max, d_check):
                    for z in np.arange(limits[-2]+eps, limits[-1], d_check):
                        cell_edges = utils.find_cell_edges(x, y, z, 0)
                        idx_i = int(round(cell_edges[0], 0))
                        idx_j = int(round(cell_edges[2], 0))
                        idx_k = int(round(cell_edges[4], 0))

                        if idx_i < 0 or idx_j < 0 or idx_k < 0 or mask[idx_i][idx_j][idx_k] < 0.5:
                            cs.append("white")
                        else:
                            cs.append(mask_color)

            # we checked several points within one root cell. If we did the masking
            # right these should all be the same
            assert len(set(cs)) == 1
            
            if not mask_labeled and cs[0] == mask_color:
                label = "Mask Region"
                mask_labeled = True
            else:
                label = None
            
            ax.fill_between(x=[x_min, x_max], y1=y_min, y2=y_max, color=cs[0], 
                            zorder=-1, label=label)
 
    ax.equal_scale()
    ax.add_labels("X [code units]", "Y [code units]", 
                  "{} < Z < {}".format(limits[4], limits[5]))
    ax.set_limits(limits[0], limits[1], limits[2], limits[3])
    ax.legend(bbox_to_anchor=(0.97,0.5), loc="center left")

    fig.savefig(savename)

plot_p_name = os.path.abspath("./plots/final_locations_particles_{}.png".format(plot_base))
plot_g_name = os.path.abspath("./plots/final_locations_gas_{}.png".format(plot_base))

print(" - making debug plots:")
print("   - {}".format(plot_p_name))
mask_plot("particle", plot_p_name)

if has_baryons:
    print("   - {}".format(plot_g_name))
    mask_plot("gas", plot_g_name)

# ==============================================================================
#
# make a plot showing the velocity at each level in the final IC
# 
# ==============================================================================
print(" - Comparing velocity on each level")
for kind in kinds:
    fig, axs = bpl.subplots(ncols=3, figsize=[20, 7])
    axs = axs.flatten()

    max_vel = np.max([np.max(get_data_wrap(data_final, "v{}".format(dim), "all", kind))
                      for dim in ["x", "y", "z"]])
    min_vel = np.min([np.min(get_data_wrap(data_final, "v{}".format(dim), "all", kind))
                      for dim in ["x", "y", "z"]])
    n_bins = 50
    bin_size = (max_vel - min_vel) / n_bins

    for idx in range(10):
        try:
            level = "zoom_{}".format(idx)
            vx = get_data_wrap(data_final, "vx", level, kind)
            vy = get_data_wrap(data_final, "vy", level, kind)
            vz = get_data_wrap(data_final, "vz", level, kind)
        except ValueError:  # ran out of levels
            break

        axs[0].hist(vx, histtype="step", rel_freq=True, bin_size=bin_size, color=bpl.color_cycle[idx], label="Level {}".format(idx))
        axs[1].hist(vy, histtype="step", rel_freq=True, bin_size=bin_size, color=bpl.color_cycle[idx], label="Level {}".format(idx))
        axs[2].hist(vz, histtype="step", rel_freq=True, bin_size=bin_size, color=bpl.color_cycle[idx], label="Level {}".format(idx))


    axs[0].add_labels("{} Velocity X [code units]".format(kind.title()), "Relative Frequency")
    axs[1].add_labels("{} Velocity Y [code units]".format(kind.title()), "Relative Frequency")
    axs[2].add_labels("{} Velocity Z [code units]".format(kind.title()), "Relative Frequency")

    axs[1].legend(loc="upper left")

    plot_name = os.path.abspath("./plots/velocity_check_levels_{}_{}.png".format(kind, plot_base))
    print("   - saving plot:\n   - {}".format(plot_name))
    fig.savefig(plot_name)

# ==============================================================================
#
# Then show the comparison between the old and new velocities
# 
# ==============================================================================
print(" - Comparing old and new velocities on each level")

n_levels_original = np.log2(header_original.NROWC) - np.log2(header_original.NGRIDC)
assert int(n_levels_original) == n_levels_original
n_levels_original = int(n_levels_original)
# iterate through all the levels in the original, other than the max zoom,
# and inluding the root level. Here idx 0 is the max zoom level, and idx
# n_levels_original is the root level
n_plot_levels = range(1, n_levels_original+1)

def do_level_masking(x_o, y_o, z_o, x_f, y_f, z_f, kind, level):
    # this will only be called on the new root grid or less refined
    # so we can make the mask on the root grid level
    mask = np.zeros((mask_size, mask_size, mask_size))
    
    print(" - making mask for zoom level {}".format(level))
    # for the root grid, we only want to compare the original ones that are 
    # actually in the new box, not outside it. We can do this by seeing if 
    # they're in the mask box as we create it
    idx_orig = []
    for idx_o in tqdm(range(len(x_o))):
        cell_edges = utils.find_cell_edges(x_o[idx_o], y_o[idx_o], z_o[idx_o], 0)
        
        idx_i = int(round(cell_edges[0], 0))
        idx_j = int(round(cell_edges[2], 0))
        idx_k = int(round(cell_edges[4], 0))
        try:
            if (idx_i < 0 or idx_j < 0 or idx_k < 0):
                raise IndexError
            mask[idx_i][idx_j][idx_k] = 1.0
            idx_orig.append(idx_o)
        except IndexError:
            # this particle is outside the box
            continue

    print(" - Checking against mask for zoom level {}".format(level))
    # then check against this mask
    idx_final = []
    for idx_f in tqdm(range(len(x_f))):
        cell_edges = utils.find_cell_edges(x_f[idx_f], y_f[idx_f], z_f[idx_f], 0)
        idx_i = int(round(cell_edges[0], 0))
        idx_j = int(round(cell_edges[2], 0))
        idx_k = int(round(cell_edges[4], 0))

        if mask[idx_i][idx_j][idx_k] > 0.5:
            idx_final.append(idx_f)

    return idx_orig, idx_final

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
            o_x = get_data_wrap(data_original, "x", level, kind)
            o_y = get_data_wrap(data_original, "y", level, kind)
            o_z = get_data_wrap(data_original, "z", level, kind)
            f_x = get_data_wrap(data_final, "x", "zoom_from_1", kind)
            f_y = get_data_wrap(data_final, "y", "zoom_from_1", kind)
            f_z = get_data_wrap(data_final, "z", "zoom_from_1", kind)

            f_vx = get_data_wrap(data_final, "vx", "zoom_from_1", kind)
            f_vy = get_data_wrap(data_final, "vy", "zoom_from_1", kind)
            f_vz = get_data_wrap(data_final, "vz", "zoom_from_1", kind)

            idx_orig, idx_final = do_level_masking(o_x, o_y, o_z, f_x, f_y, f_z, kind, idx)

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
        elif len(get_data_wrap(data_original, "vx", level, kind)) == len(get_data_wrap(data_original, "vx", "root", "particle")):
            title = "Old Root Level"
        else:
            title = "Intermediate Layer {}".format(idx)
            
        if idx == zoom_levels:
            title += " - New Root Level"
        
        ax_row[0].add_labels("Velocity X [code units]", "Relative Frequency")
        ax_row[1].add_labels("Velocity Y [code units]", "Relative Frequency", title)
        ax_row[2].add_labels("Velocity Z [code units]", "Relative Frequency")
        
        ax_row[0].easy_add_text("Original N={:,}\nFinal N={:,}".format(len(orig_vx), len(this_vx)), 
                                "upper left", fontsize=16)
        
        ax_row[1].legend(loc="upper left")

    
    plot_name = os.path.abspath("./plots/velocity_level_comparison_{}_{}.png".format(kind, plot_base))
    fig.savefig(plot_name)
    print("   - {}".format(plot_name))

print(" - Done!")

