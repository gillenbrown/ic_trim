import struct
import os
import numpy as np
from tqdm import tqdm as tqdm

# convenience function 
def pack_and_write(value, fmt, out_file):
    try:
        len(value)
        s = struct.pack(fmt, *value)
    except TypeError: # regular integers
        s = struct.pack(fmt, value)
    out_file.write(s)

class ARTHeader(object):
    def __init__(self, directory, has_baryons):
        self.directory = directory
        self.has_baryons = has_baryons

        if directory is None:
            header_filename = None
        elif has_baryons:
            header_filename = directory + os.sep + "music_H.mdh"
        else:
            header_filename = directory + os.sep + "music_D.mdh"

        if header_filename is not None:
            self.read_header(header_filename)
        else:
            # set up blank header
            self.header_data = None
            self.idx = None
            self.blocksize = None
            self.head = None
            self.aexpN = None
            self.aexp0 = None
            self.amplt = None
            self.astep = None
            self.istep = None
            self.partw = None
            self.TINTG = None
            self.EKIN = None
            self.EKIN1 = None
            self.EKIN2 = None
            self.AU0 = None
            self.AEU0 = None
            self.NROWC = None
            self.NGRIDC = None
            self.nspecies = None
            self.Nseed = None
            self.Om0 = None
            self.Oml0 = None
            self.hubble = None
            self.Wp5 = None
            self.Ocurv = None
            self.Omb0 = None
            self.wpart = None
            self.lpart = None  # cumulative number of DM particles from high to low res!!
            self.magic1 = None
            self.DelDC = None
            self.abox = None
            self.Hbox = None
            self.magic2 = None
            self.extras = None
    
    def _read_bytes(self, fmt):
        """Read bytes starting at idx and increment idx appropriately"""
        new_idx = self.idx + struct.calcsize(fmt)
        item = struct.unpack(fmt, self.header_data[self.idx:new_idx])
        self.idx = new_idx
        # struct always returns a tuple. If we know we have only
        # one item, just get that item from the one-item tuple
        if len(fmt) == 1:
            item = item[0] 
        return item

    def read_header(self, header_filename):
        # first just open the file and get all the data
        with open(header_filename, "rb") as in_file:
            self.header_data = in_file.read()
        
        # then parse it. This is all based on lines 208-241 of 
        # output_cart.cc in MUSIC's plugin directory. The types 
        # of all these things are given on lines 52-87 of the
        # same file.
        self.idx = 0  # will be incremented by _read_bytes()
        self.blocksize = self._read_bytes("i")
        self.head = self._read_bytes("c"*45)
        self.aexpN = self._read_bytes("f")
        self.aexp0 = self._read_bytes("f")
        self.amplt = self._read_bytes("f")
        self.astep = self._read_bytes("f")
        self.istep = self._read_bytes("i")
        self.partw = self._read_bytes("f")
        self.TINTG = self._read_bytes("f")
        self.EKIN = self._read_bytes("f")
        self.EKIN1 = self._read_bytes("f")
        self.EKIN2 = self._read_bytes("f")
        self.AU0 = self._read_bytes("f")
        self.AEU0 = self._read_bytes("f")
        self.NROWC = self._read_bytes("i")
        self.NGRIDC = self._read_bytes("i")
        self.nspecies = self._read_bytes("i")
        self.Nseed = self._read_bytes("i")
        self.Om0 = self._read_bytes("f")
        self.Oml0 = self._read_bytes("f")
        self.hubble = self._read_bytes("f")
        self.Wp5 = self._read_bytes("f")
        self.Ocurv = self._read_bytes("f")
        self.Omb0 = self._read_bytes("f")
        self.wpart = self._read_bytes("f"*10) # mass of DM particles!!
        self.lpart = self._read_bytes("i"*10)  # cumulative number of DM particles from high to low res!!
        self.magic1 = self._read_bytes("f")
        self.DelDC = self._read_bytes("f")
        self.abox = self._read_bytes("f")
        self.Hbox = self._read_bytes("f")
        self.magic2 = self._read_bytes("f")
        self.extras = self._read_bytes("f"*75)
        # Some notes on extras:
        # These are written on line 630 of output_cart.cc
        # item 13 is Omega_b
        # 14 is sigma_8
        # 15 is nspec of the power spectrum
        # and the last item is the box size
        # the rest are zeros
        # No idea why these particular ones
        self.blocksize2 = self._read_bytes("i")
        
    def write_header(self, output_file):
        # first just open the file
        with open(output_file, "wb") as out_file:
            # then write all the data, in the same way we read it in
            pack_and_write(self.blocksize, "i", out_file)
            pack_and_write(self.head, "c"*45, out_file)
            pack_and_write(self.aexpN, "f", out_file)
            pack_and_write(self.aexp0, "f", out_file)
            pack_and_write(self.amplt, "f", out_file)
            pack_and_write(self.astep, "f", out_file)
            pack_and_write(self.istep, "i", out_file)
            pack_and_write(self.partw, "f", out_file)
            pack_and_write(self.TINTG, "f", out_file)
            pack_and_write(self.EKIN, "f", out_file)
            pack_and_write(self.EKIN1, "f", out_file)
            pack_and_write(self.EKIN2, "f", out_file)
            pack_and_write(self.AU0, "f", out_file)
            pack_and_write(self.AEU0, "f", out_file)
            pack_and_write(self.NROWC, "i", out_file)
            pack_and_write(self.NGRIDC, "i", out_file)
            pack_and_write(self.nspecies, "i", out_file)
            pack_and_write(self.Nseed, "i", out_file)
            pack_and_write(self.Om0, "f", out_file)
            pack_and_write(self.Oml0, "f", out_file)
            pack_and_write(self.hubble, "f", out_file)
            pack_and_write(self.Wp5, "f", out_file)
            pack_and_write(self.Ocurv, "f", out_file)
            pack_and_write(self.Omb0, "f", out_file)
            pack_and_write(self.wpart, "f"*10, out_file) # mass of DM particles!!
            pack_and_write(self.lpart, "i"*10, out_file)  # cumulative number of DM particles from high to low res!!
            pack_and_write(self.magic1, "f", out_file)
            pack_and_write(self.DelDC, "f", out_file)
            pack_and_write(self.abox, "f", out_file)
            pack_and_write(self.Hbox, "f", out_file)
            pack_and_write(self.magic2, "f", out_file)
            pack_and_write(self.extras, "f"*75, out_file)
            # Some notes on extras:
            # These are written on line 630 of output_cart.cc
            # item 13 is Omega_b
            # 14 is sigma_8
            # 15 is nspec of the power spectrum
            # and the last item is the box size
            # the rest are zeros
            # No idea why these particular ones
            pack_and_write(self.blocksize2, "i", out_file)

def compare_headers(header_a, header_b):
    for item in vars(header_a):
        if not item.startswith("_") and item not in ["header_data", "extras"]:
            a_attr = header_a.__getattribute__(item)
            b_attr = header_b.__getattribute__(item)
            if a_attr != b_attr:
                print("   -", item, a_attr, b_attr)
    for idx in range(75):
        a_attr = header_a.extras[idx]
        b_attr = header_b.extras[idx]
        if a_attr != b_attr:
            print("   - extra[{}]".format(idx), a_attr, b_attr)

class ARTData(object):
    def __init__(self, directory, has_baryons):
        self.header = ARTHeader(directory, has_baryons)

        self.directory = directory
        self.has_baryons = has_baryons

        if directory is None:
            particle_filename = None
            gas_filename = None
        elif has_baryons:
            particle_filename = directory + os.sep + "music_H.mdxv"
            gas_filename = directory + os.sep + "music_H.md"
        else:
            particle_filename = directory + os.sep + "music_D.mdxv"
            gas_filename = None

        if particle_filename is not None:
            self.read_data(particle_filename, "particle")
        else:
            self.dm_data = {"x": np.zeros(0),
                            "y": np.zeros(0),
                            "z": np.zeros(0),
                            "vx": np.zeros(0),
                            "vy": np.zeros(0),
                            "vz": np.zeros(0)}
        if gas_filename is not None:
            self.read_data(gas_filename, "gas")
        else:
            self.gas_data = {"x": np.zeros(0),
                             "y": np.zeros(0),
                             "z": np.zeros(0),
                             "vx": np.zeros(0),
                             "vy": np.zeros(0),
                             "vz": np.zeros(0),
                             "pma": np.zeros(0)}
    

    def read_data(self, filename, kind):
        # things are different for gas and particle datasets, but only slightly,
        # so I can do both in one function
        gas = kind == "gas"

        print(" - reading {} data from {}".format(kind, filename))

        with open(filename, "rb") as in_file:
            data_raw = in_file.read()
            
        # When writing (see assemble_DM_file, starting line 286 of 
        # output_cart.cc, or assemble_gas_file, starting line 411 of the same),
        # the DM particles (gas) are written in blocks of `block_buf_size_` 
        # values. This can be seen from the code starting at line 336 (465), 
        # where the actual writing starts on line 364 (495). We write 6 (7) 
        # fields that are all `block_buf_size_` items, and the fields are x, y, 
        # z, vx, vy, vz (plus pma for baryons) read from the temporary files on 
        # line 350-355 (479-485). `block_buf_size_` is defined on line 558 as 
        # (2**max_level)**2. So the entire file is written in blocks of this 
        # size, regardless of whether or not the zoom fills an entire block.
        # Another way of saying this is that there is only one block size in 
        # the entire file, even when writing particles (cells) on different 
        # levels. Since this means that the number of particles (cells) likely 
        # won't divide evenly into blocks, lines 342-349 (471-478) write zeros
        # in the leftover spots.
        # some notation here: block = one set of data for one attribute 
        # (i.e. x, y, z)
        # page = set of 6 blocks
        fields = ["x", "y", "z", "vx", "vy", "vz"]
        if gas:
            fields.append("pma")

        block_items = self.header.NROWC**2  # NROWC is 2**max_level
        block_fmt = "d" * block_items
        block_size = struct.calcsize(block_fmt)

        n_fields = len(fields)
        page_size = block_size * n_fields
        
        # check out math, since the block size should evenly divide
        # the file size
        file_size = len(data_raw)
        n_pages = file_size / page_size
        assert int(n_pages) == n_pages
        n_pages = int(n_pages)
        n_blocks = n_pages * n_fields
        # I need to store things as numpy arrays, so I need to figure out how long
        # the arrays will be. We take the number of pages times the items in each
        # block since this variable is the number of total values per attribute, 
        # and there is one block of this attribute per page
        total_n_items = n_pages * block_items
        
        # then read the file, going one block at at a time
        if gas:
            self.gas_data = {field: np.zeros(total_n_items)
                             for field in fields}
            obj_data = self.gas_data  # placeholder
        else:
            self.dm_data = {field: np.zeros(total_n_items)
                             for field in fields}
            obj_data = self.dm_data  # placeholder
        
        pbar = tqdm(total=n_blocks)
        idx_file = 0
        idx_array = 0
        while idx_file < file_size:
            # each page has all fields
            for field in fields:
                this_data = struct.unpack(block_fmt, data_raw[idx_file:idx_file+block_size])
                
                obj_data[field][idx_array:idx_array+block_items] = this_data
                
                idx_file += block_size # move to next field
                
                pbar.update(1) # update progress bar
            
            idx_array += block_items
        pbar.close()
        
        del data_raw
                
    
    def debug(self):
        print("max refine expected length:", self.header.NROWC**3)
            
        x = self.get_particles("x")
        y = self.get_particles("y")
        z = self.get_particles("z")
        vx = self.get_particles("vx")
        vy = self.get_particles("vy")
        vz = self.get_particles("vz")

        print("x", np.min(x), np.max(x), len(x))
        print("y", np.min(y), np.max(y), len(y))
        print("z", np.min(z), np.max(z), len(z))
        print("vx", np.min(vx), np.max(vx), len(vx))
        print("vy", np.min(vy), np.max(vy), len(vy))
        print("vz", np.min(vz), np.max(vz), len(vz))
        
    def plot_residuals(self, subset):
        fig, axs = bpl.subplots(ncols=3, figsize=[15, 7])
        for ax, dim in zip(axs, ["x", "y", "z"]):
            values = self.get_particles(dim, subset)
            values = values % 1
            ax.hist(values, bin_size=0.05, rel_freq=True)

            ax.add_labels(dim, "Relative Frequency")
        
    def get_data(self, field, subset, kind):
        # the lpart attribute has the cumulativce number of particles/cells
        # at each refinement level. We can use this to slice the data
        # and exclude any zeros if present in a zoom sim. We can also use
        # this to select only the zoom particles/cells, since the writing (and 
        # therefore reading) were done from high res to low res

        for idx, value in enumerate(self.header.lpart):
            if value == 0:
                # Subtracting 1 from the index of the first zero gets us
                # the index for the total number of particles, subtracting
                # 2 gets us to the total number of zoom particles, subtracting
                # 3 gets us to the total number of zoom other than the 
                # least refined zoom level, etc. Each step back cuts off one
                # refinement level
                if subset == "zoom":
                    idx_lpart_1 = idx - 2
                    
                    idx_data_0 = 0
                    idx_data_1 = self.header.lpart[idx_lpart_1]

                elif subset.startswith("zoom_to"):
                    zoom_level = int(subset.split("_")[-1])
                    idx_data_0 = 0  # always start at the beginning
                    idx_lpart_1 = zoom_level  

                    idx_data_1 = self.header.lpart[idx_lpart_1]

                    if idx_data_1 == 0:
                        raise ValueError("zoom level {} does not exist".format(zoom_level))

                elif subset.startswith("zoom_from"):
                    zoom_level = int(subset.split("_")[-1])
                    # start at the beginning of this idx, go to end (include root)
                    if zoom_level == 0:
                        idx_data_0 = 0
                    else:
                        idx_lpart_0 = zoom_level - 1
                        idx_data_0 = self.header.lpart[idx_lpart_0]
                    # end is the one before where we are now
                    idx_lpart_1 = idx - 1
                    idx_data_1 = self.header.lpart[idx_lpart_1]

                    if idx_data_1 == 0:
                        raise ValueError("zoom level {} does not exist".format(zoom_level))

                    
                elif subset.startswith("zoom"):
                    zoom_level = int(subset.split("_")[1])
                    
                    if zoom_level == 0:
                        idx_lpart_1 = zoom_level

                        idx_data_0 = 0
                    else:
                        idx_lpart_0 = zoom_level - 1
                        idx_lpart_1 = idx_lpart_0 + 1
                    
                        idx_data_0 = self.header.lpart[idx_lpart_0]

                    idx_data_1 = self.header.lpart[idx_lpart_1]

                    if idx_data_1 == 0:
                        raise ValueError("zoom level {} does not exist".format(zoom_level))
                    
                elif subset == "root":
                    idx_lpart_0 = idx - 2
                    idx_lpart_1 = idx - 1
                    
                    idx_data_0 = self.header.lpart[idx_lpart_0]
                    idx_data_1 = self.header.lpart[idx_lpart_1]
                elif subset == "all":
                    idx_lpart_1 = idx - 1
                    
                    idx_data_0 = 0
                    idx_data_1 = self.header.lpart[idx_lpart_1]
                else:
                    raise ValueError("Wrong name")
                
                if kind == "gas":
                    return self.gas_data[field][idx_data_0:idx_data_1]
                elif kind == "particle":
                    return self.dm_data[field][idx_data_0:idx_data_1]
                else:
                    raise ValueError("Wrong kind!")

    def delete_data(self, kind, field):
        if kind == "gas":
            del self.gas_data[field]
        elif kind == "particle":
            del self.dm_data[field]

    def fix_offset(self, dx, dy, dz):
        self.gas_data["x"] += dx
        self.gas_data["y"] += dy
        self.gas_data["z"] += dz
        self.dm_data["x"] += dx
        self.dm_data["y"] += dy
        self.dm_data["z"] += dz
                
    def write_data(self, filename, kind):
        # This is very similar to reading, see that documentation
        out_file = open(filename, "wb")
        # some notation here: block = one set of data for one attribute (i.e. x, y, z)
        # page = set of 6 blocks
        fields = ["x", "y", "z", "vx", "vy", "vz"]
        if kind == "gas":
            fields.append("pma")
            obj_data = self.gas_data  # placeholder
        else:
            obj_data = self.dm_data  # placeholder

        block_items = self.header.NROWC**2  # NROWC is 2**max_level
        block_fmt = "d" * block_items
        block_size = struct.calcsize(block_fmt)
        n_fields = len(fields)
        page_size = block_size * n_fields
        
        n_items = len(self.get_data("x", "all", kind))
        pbar = tqdm(total=n_items)
        idx_array = 0
        while idx_array <= n_items:
            # each page has all fields
            for field in fields:
                # check for the last block, where we have to pad with zeros
                idx_end = idx_array+block_items
                if idx_end >= n_items:  # last page
                    block_data = obj_data[field][idx_array:]
                    n_to_pad = block_items - len(block_data)
                    # pad can add zeros before and after, but we only want them after
                    block_data = np.pad(block_data, (0, n_to_pad), 
                                        "constant", constant_values=0.0)
                else:  #normal writing
                    block_data = obj_data[field][idx_array:idx_end]
                
                pack_and_write(block_data, block_fmt, out_file)
            
            idx_array += block_items
            pbar.update(block_items)
            
        pbar.close()
        out_file.close()

def restrict_1d_idxs(x, x_min, x_max):
    """Get the indices where x_min < x < x_max """
    idx_x_above = np.where(x > x_min)
    idx_x_below = np.where(x < x_max)
    return np.intersect1d(idx_x_above, idx_x_below)
                          
def restrict_to_xyz(x, y, z, x_min, x_max, y_min, y_max, z_min, z_max):
    """Get the x,y,z values that satisy all of:
    x_min < x < x_max
    y_min < y < y_max
    z_min < z < z_max
    """
    # first find all values where the z values are satisfied.
    # Do the z value first since it's often the strictest cut
    idxs_z = restrict_1d_idxs(z, z_min, z_max)
    # restrict to only the good ones
    x = x[idxs_z]
    y = y[idxs_z]
    z = z[idxs_z]
    
    # then among those, find the values where the y value works
    idxs_y = restrict_1d_idxs(y, y_min, y_max)
    x = x[idxs_y]
    y = y[idxs_y]
    z = z[idxs_y]
    
    # same things with x
    idxs_x = restrict_1d_idxs(x, x_min, x_max)
    x = x[idxs_x]
    y = y[idxs_x]
    z = z[idxs_x]
    
    return x, y, z

def plot_one_sim(x, y, z, x0, x1, y0, y1, z0, z1, ax, **kwargs):
    # plot all DM particles in the box bounded
    # by the coordinates selected
    x, y, z = restrict_to_xyz(x, y, z, x0, x1, y0, y1, z0, z1)  
    ax.scatter(x, y, alpha=1, **kwargs)

def find_cell_edges(x, y, z, refinement_level):
    """Find the cell edges for the cell that the point x,y,z is in. The 
    refinement level can be specified, where 0 is the root grid, where the cell
    boundaries are at integer values, and each level is a factor of 2 refinement
    in length"""

    cell_size = 1 / (2**refinement_level)
    # First just start at integer values for the minimum, we'll 
    # iterate on this later
    x_min = np.floor(x)
    x_max = x_min + cell_size
    
    y_min = np.floor(y)
    y_max = y_min + cell_size
    
    z_min = np.floor(z)
    z_max = z_min + cell_size
    
    # The move the boundary until it contains the point
    while not x_min <= x <= x_max:
        x_min += cell_size
        x_max += cell_size
    while not y_min <= y <= y_max:
        y_min += cell_size
        y_max += cell_size
    while not z_min <= z <= z_max:
        z_min += cell_size
        z_max += cell_size
        
    return x_min, x_max, y_min, y_max, z_min, z_max

def find_restrict_idxs(x, y, z, x_min, x_max, y_min, y_max, z_min, z_max):
    """Get the indicies that satisy all of:
    x_min < x[idx] < x_max
    y_min < y[idx] < y_max
    z_min < z[idx] < z_max

    Very similar to restrict_to_xyz, just for only the indices, not 
    x,y,z locations. This is less efficient than that function
    """
    idxs_x = restrict_1d_idxs(x, x_min, x_max)
    idxs_y = restrict_1d_idxs(y, y_min, y_max)
    idxs_z = restrict_1d_idxs(z, z_min, z_max)
    
    idxs_xy = np.intersect1d(idxs_x, idxs_y)
    idxs_xyz = np.intersect1d(idxs_xy, idxs_z)
    
    return idxs_xyz

def distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

def make_mask(cube_size, level, xs, ys, zs):
    """Make a mask, where cells that contain any particles (specified by 
    x, y, z) are 1, and empty cells are 0. 

    :param cube_size: the number of cells on the side of the mask cube
    :param level: Level at which the mask will be created (see find_cell_edges)
    """
    mask = np.zeros((cube_size, cube_size, cube_size))
    
    cell_size = 1 / (2**level)
    
    for idx_z in tqdm(range(len(zs))):
        cell_edges = find_cell_edges(xs[idx_z], ys[idx_z], zs[idx_z], level)
        
        idx_i = int(round(cell_edges[0] / cell_size, 0))
        idx_j = int(round(cell_edges[2] / cell_size, 0))
        idx_k = int(round(cell_edges[4] / cell_size, 0))

        # we have to be careful about negative indices, which can wrap around 
        # when that doesn't make physical sense
        if (idx_i < 0 or idx_j < 0 or idx_k < 0):
            raise IndexError
        mask[idx_i][idx_j][idx_k] = 1.0 # can raise index error too

        
    return mask


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

# have a few functions to make the coordinate systems match. These are 
# determined by the geometry of each box. These equations look a little 
# complicated, but can be derived by drawing the geometry of the two boxes
def arbitrary_shift(x_old, l_old, l_reference, N_old, N_reference):
    cell_size_old = l_old / N_old
    cell_size_reference = l_reference / N_reference
    delta_n = (l_old - l_reference) / (2 * cell_size_old)
    return (cell_size_old / cell_size_reference)  * (x_old - delta_n)
