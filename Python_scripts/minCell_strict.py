# ----------------------------------------------
# Get the minimum number of unit cells in each direction 
# This version is based on the most strict constraints propsed by William Smith to satisfy the minimum image convension
# 
# Command-line argument input: txt file containing [MOF number id] 
# Output: txt file, each row contains [MOF number id] [n_cell_x] [n_cell_y] [n_cell_z]
# 
# 5/17/2021 (adapted from Yamil Colon's code)
# ----------------------------------------------

# Needs Pymatgen installed 
from pymatgen.core import Structure
import numpy as np  
import os 
import sys                
import re 
import math

# cut-off radius [A]
rcut = 12.8 

# MOF structure path
mof_dir = '/home/kaihang/2dhist/Tobacco'
#mof_dir = '/global/project/projectdirs/m538/kshi/Tobacco1.0'

#--------------------------------------------------------------------------------------
# Read in MOF lists
#--------------------------------------------------------------------------------------
# command-line argument
if len(sys.argv) >2:
	print('Only one argument is allowed!')
	exit()

# Read file
# sys.argv[0] is the program ie. script name.
input_file = open(sys.argv[1]).readlines()
mof_id = []

for line in input_file:
    # convert string to list
    fields = line.strip().split()
    mof_id.append(str(fields[0]))


#---------------------------------------------------------------------------------------
# The heart of the code
#---------------------------------------------------------------------------------------
# define a core function that performs the calculation 
def minCells_strict(mof_id):

    # system operation
    os.chdir(mof_dir)
    #os.chdir(mof_id)
    filelist = os.listdir('.')

    # set up regular expression for matching purpose
    #r = re.compile(".*cif")
    r = re.compile("{}\\.cif".format(mof_id))
    cif_file_name = list(filter(r.match,filelist))[0]
    
    # Read the coordinates from the cif file using pymatgen
    struct = Structure.from_file(cif_file_name, primitive=False)
    # struct belongs to class 'Structure'
    aa = struct.lattice.a
    bb = struct.lattice.b
    cc = struct.lattice.c
    alpha = struct.lattice.alpha
    beta  = struct.lattice.beta
    gamma = struct.lattice.gamma

    # unit cell matrix
    ax = aa
    ay = 0.0
    az = 0.0
    bx = bb * np.cos(gamma * np.pi / 180.0)
    by = bb * np.sin(gamma * np.pi / 180.0)
    bz = 0.0
    cx = cc * np.cos(beta * np.pi / 180.0)
    cy = (cc * np.cos(alpha * np.pi /180.0) * bb - bx * cx) / by
    cz = (cc ** 2 - cx ** 2 - cy ** 2) ** 0.5
    unit_cell =  np.asarray([[ax, ay, az],[bx, by, bz], [cx, cy, cz]])

    #Unit cell vectors
    A = unit_cell[0]
    B = unit_cell[1]
    C = unit_cell[2]

    #minimum distances between unit cell faces (Wa = V_tot/area_BC)
    Wa = np.divide(np.linalg.norm(np.dot(np.cross(B,C),A)), np.linalg.norm(np.cross(B,C)))
    Wb = np.divide(np.linalg.norm(np.dot(np.cross(C,A),B)), np.linalg.norm(np.cross(C,A)))
    Wc = np.divide(np.linalg.norm(np.dot(np.cross(A,B),C)), np.linalg.norm(np.cross(A,B)))

    uc_x = int(np.ceil(2.0*rcut/Wa))
    uc_y = int(np.ceil(2.0*rcut/Wb))
    uc_z = int(np.ceil(2.0*rcut/Wc))

    # write to file
    fout.write("{0:10}\t{1:d}\t{2:d}\t{3:d}\n".format(mof_id,uc_x,uc_y,uc_z))
    

# Start looping over all training sets
fout=open('unitcells_strict.txt','a')
for index in mof_id:
	minCells_strict(index)
	fout.flush()
    
# close file
fout.close()
