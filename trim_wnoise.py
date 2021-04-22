import sys
import os
import h5py
import numpy as np
import struct
import betterplotlib as bpl
bpl.set_style()

# parse the command line arguments
wnoise_file_loc = os.path.abspath(sys.argv[1])
base_level = int(sys.argv[2])
if len(sys.argv) == 3:
    central_power = base_level
else:
    central_power = int(sys.argv[3])
cut_length = 2**central_power

if len(sys.argv) > 4:
    raise ValueError("Too many arguments")

print(" - Reading white noise cube")
# Read the DM density at the level requested
hf = h5py.File(wnoise_file_loc, "r")
wnoise = np.array(hf['level_{:03d}_DM_rho'.format(base_level)])
hf.close()

print(" - Trimming white noise cube")

# Then trim it. 
def trim_wnoise(wnoise, new_length):
    # This will take the central `new_length` values for all directions.
    # calculate how much to trim off each end
    # This assumes we have a cube
    wnoise_new = wnoise.copy()

    trim_end = (wnoise_new.shape[0] - new_length) / 2.0
    trim_end_int = int(trim_end)
    assert trim_end == trim_end_int

    idx_start = trim_end_int
    idx_end = wnoise_new.shape[0] - trim_end_int

    return wnoise_new[idx_start:idx_end,idx_start:idx_end,idx_start:idx_end]

# first trim off the zero edges
wnoise = trim_wnoise(wnoise, 2**base_level)
# then get the smaller cube
cut_wnoise = trim_wnoise(wnoise, cut_length)

# Then normalize it
# center at zero
# cut_wnoise -= np.mean(cut_wnoise.flatten())
# give a mean of zero
cut_wnoise /=  np.std(cut_wnoise.flatten())

# Print some key properties
print(" - White noise properties:")
print("     - Original cube size: {:,d}".format(wnoise.size))
print("     - New cube size: {:,d}".format(cut_wnoise.size))
print("     - Expected cube size: {:d}^3 = {:,d}".format(cut_length, cut_length**3))
print("     - Original cube mean: {:.5f}".format(np.mean(wnoise.flatten())))
print("     - Original cube std: {:.5f}".format(np.std(wnoise.flatten())))
print("     - New cube mean: {:.5f}".format(np.mean(cut_wnoise.flatten())))
print("     - New cube std: {:.5f}".format(np.std(cut_wnoise.flatten())))

# Plot the white noise to validate its properties
# the plot name will be based on the white noise file name
print(" - Plotting white noise histogram")
plot_base = ".".join(os.path.basename(wnoise_file_loc).split(".")[:-1])
if base_level != central_power:
    savename = "../plots/" + plot_base + "_readlevel{:02d}_trim{:02d}.png".format(base_level, central_power)
else:
    savename = "../plots/" + plot_base + "_readlevel{:02d}.png".format(base_level)
savename = os.path.abspath(savename)
print("     - Will be saved to " + savename)

fig, ax = bpl.subplots()
ax.hist(cut_wnoise.flatten(), bin_size=0.05, rel_freq=True)
ax.easy_add_text("White Noise\n$\sigma$ = {:.2f}\nmean = {:.2f}".format(np.std(cut_wnoise.flatten()), np.mean(cut_wnoise.flatten())),
                 "upper left")
ax.add_labels("Value", "Relative Frequency")
fig.savefig(savename)

# Then write the white noise to the file. 
# We need to get the new name for the file, based on the new size
cut_ratio = 2**(base_level - central_power)
old_length = float(plot_base.split("_")[1].replace("mpc", ""))
new_length = old_length / cut_ratio
out_filename = "./trimmed_" + plot_base + "_readlevel{:02d}_trimlength{:.2f}mpc_trimlevel{:02d}.dat".format(base_level, new_length, central_power) 
out_filename = os.path.abspath(out_filename)
print(" - Writing white noise to " + out_filename)

# convenience function 
def pack_and_write(fmt, value, out_file):
    try:
        len(value)
        s = struct.pack(fmt, *value)
    except TypeError: # regular integers
        s = struct.pack(fmt, value)
    out_file.write(s)

# all based on random.cc of MUSIC
# See https://docs.python.org/3/library/struct.html#format-characters
# for how to use struct

nx = cut_wnoise.shape[0]
ny = cut_wnoise.shape[1]
nz = cut_wnoise.shape[2]

int_size = 4
double_size = 8

with open(out_filename, "wb") as out_file:
    
    # lines 317 - 330 identify what kind of data we use
    # We use the sizes specified in lines 325-326, to use 64 bit
    # After this check is done, the file is reset (line 331) then
    # the actual reading starts on line 336 with blocksize, then 
    # the dimensions
    pack_and_write("N", 4*int_size, out_file)  # blksz64, line 339. 
    # This is the number of bytes in the coming set - nx ny nz iseed
    pack_and_write("I", nx, out_file) # nx, line 341
    pack_and_write("I", ny, out_file) # ny, line 342
    pack_and_write("I", nz, out_file) # nz, line 343
    pack_and_write("i", 0, out_file)   #iseed, line 344
    # not sure if the seed should be what's in the original file or not.
    # I'll ignore it for now
    
    # I don't understand what's happening on lines 355-358
    # This value doesn't appear to be used. I'll write a blank value for now
#         pack_and_write("N", 0, out_file)   #blksz64, line 358
    
    # Then we tell the number of bytes in the coming chunk, which has
    # nx*ny = 128*128 entries. We want doubles, so this is
    # 8 bytes each, so we have 128*128*8
    pack_and_write("N", nx*ny*double_size, out_file)  # blksz64, line 373
    # This is also rewound at line 386 then reread on line 410
    
    # then we do the actual writing of the random values
    count = 0
    for z_idx in range(nz):
        # then another statement of the slice size
        pack_and_write("N", nx*ny*double_size, out_file)  # blksz64, line 454
        
        for y_idx in range(ny):
            for x_idx in range(nx):
                # -1 is needed for some reason
                pack_and_write("d", -1 * cut_wnoise[x_idx][y_idx][z_idx], out_file)

        # then another statement of the slice size
        pack_and_write("N", nx*ny*double_size, out_file)  # blksz64, line 478

print(" - Done!")

