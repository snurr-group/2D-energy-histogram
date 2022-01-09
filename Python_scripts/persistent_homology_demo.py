# ----------------------------------------------
# This Demo script provides a workflow to get persistance image from a single CIF structure
# All length units are in [Angstrom]
# 
# How to use: type the following command in command-line
# python persistent_homology_demo.py xxx.cif
# 
# Author: Kaihang Shi
# Email : kaihangshi0@gmail.com
# Last update: 4/15/2021
# ----------------------------------------------

import pymatgen.core as pmcore
import pymatgen.io.cif as pmcif
import diode as dd
import dionysus as dy
import numpy as np  
import matplotlib.pyplot as plt
import os 
import sys                
import re 
import math
from persim import PersistenceImager
from persim.images_weights import linear_ramp

# ------------------------- Input parameters ------------------------
# Make a super cell with abc_norm^3 to normalize the data set 
# 250 A is at least twice as large as the abc of the largest MOFs in Tobacco 1.0
abc_norm = 125.0 

# exactness of alpha shape calculations (set to False by default, for quality run set it to True)
exactness = True

# dimension of homology to output (0-edge, 1-loop, 2-void)
dim_homology = 2

# pixel size of persistence image ()
pixel_size = 1.0

# Gaussian kernel sigma
sigma = 0.15


# ------------------------- Preparation ------------------------
# command-line argument
if len(sys.argv) >2:
	print('Only one CIF is allowed!')
	exit()
# sys.argv[0] is the program ie. script name.
cif_file = sys.argv[1]

# Read the coordinates from the cif file using pymatgen
struct = pmcore.Structure.from_file(cif_file, primitive=False)
# struct belongs to class 'Structure'
# https://pymatgen.org/pymatgen.io.cif.html
aa = struct.lattice.a
bb = struct.lattice.b
cc = struct.lattice.c

# calculate the minimum cell
n_a = round(abc_norm/aa)
n_b = round(abc_norm/bb)
n_c = round(abc_norm/cc)

# Make a supercell to normalize every MOF
struct.make_supercell([n_a,n_b,n_c])

# write to cif file 
sc_cif = pmcif.CifWriter(struct)
#sc_cif.write_file('supercell.cif')

# a numpy array containing all cartesian coordiantes 
# (points for persistent homology)
points = struct.cart_coords


# ------------------------- Main body ------------------------
# Calculate all simplices based on given points
simplices = dd.fill_alpha_shapes(points,exact=exactness)

# Perform alpha-shape filtration (ie nested alpha-complex)
filtr = dy.Filtration(simplices)

# Compute persistent homology
ph = dy.homology_persistence(filtr)

# initialize persistent diagram
persdgm_all = dy.init_diagrams(ph,filtr)

# Convert persistence diagram into a numpy array
# initialize numpy array
persdgm = np.empty(shape=(0,2))

# loop over all birth-death point in nD-homology
for pt in persdgm_all[dim_homology]:
	# convert alpha value to more intuitive radius of ball
	r_birth = math.sqrt(pt.birth)
	r_death = math.sqrt(pt.death)
	persdgm = np.append(persdgm,np.array([[r_birth,r_death]]), axis=0)

	
# initialize persistent image
persimg = PersistenceImager(pixel_size = pixel_size)
# determine birth and persistence range to capture all persistence pair; 
# The option `skew=True` specifies that the diagram is currently in birth-death-
# coordinates and must first be transformed to birth-persistence coordinates
persimg.fit(persdgm,skew=True)

# Customize parameters of persistent image
persimg.pers_range = (0, persimg.pers_range[1]+5)
persimg.birth_range = (0, persimg.birth_range[1]+5)

# change weight function to a uniform one
persimg.weight = linear_ramp
# standard linear ramp
#persimg.weight_params = {'low':0.0, 'high':1.0, 'start':0.0, 'end':10.0}
# no weighting (uniform function)
persimg.weight_params = {'low':1.0, 'high':1.0, 'start':0.0, 'end':1.0}

# set Gaussian sigma
persimg.kernel_params = {'sigma': sigma}

# convert from birth-death diagram to persistent image (birth-persistence)
pimgs = persimg.transform(persdgm,skew=True)

# make plot
fig, axs = plt.subplots(1, 3, figsize=(10,5))

axs[0].set_title("Original Diagram")
persimg.plot_diagram(persdgm, skew=False, ax=axs[0])

axs[1].set_title("Birth-Persistence\nCoordinates")
persimg.plot_diagram(persdgm, skew=True, ax=axs[1])

axs[2].set_title("Persistence Image")
persimg.plot_image(pimgs, ax=axs[2])

plt.tight_layout()
plt.savefig("figure.png")
plt.show()




