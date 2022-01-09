# ------------------------------------------------------------------
# This script calculates energy/pore size grids for a large amount of structures 
# 	in parallel manner & split data into training and testing set
#
# Output: 'summary.txt' containing MOF IDs for training and testing data, and the 
#			wall-time (in seconds) required to generate grids for that certain structure.
#
# Kaihang Shi  Created on 8-22-2021
# 
# ------------------------------------------------------------------

# load packages
import os
import datetime
import time 
import sys
import subprocess
import random
import re
import pandas as pd
import math
import numpy as np  
# multiprocessing is pre-installed in Python distribution
import multiprocessing
from joblib import Parallel, delayed
from shutil import copy2
from pymatgen.core import Structure


# ------------------------------------------
# Control parameters 
# ------------------------------------------
# Specify mode
# 1 - read MOF IDs from gcmc file; 2 - random sample from entire pool
mode = 1
# shuffle the MOF IDs/data
shuffle = True

# gcmc path
gcmc_dir = '/home/kaihang/pore_2dhist/data_split/big_data/gcmc_XeKr_273K_1Bar_indiXe_cm3cm3.txt'

# Specify parameters
n_train = 1000 
n_test = 1000

# cutoff radius [A] for calculations (to calculate minimum size of unit cell)
rcut = 12.0

# MOF structure path
mof_dir = '/home/kaihang/2dhist/Tobacco'

# Skip a MOF if raw grid file is already there
checkgrid = True
rawgridname = 'ener_psize_Xe_1A.txt'

# path to C++ code for energy/pore-size grid calculations
LJ_BIN = '/home/kaihang/pore_2dhist/gridsCalc/bin/grid_Xe_poresize_1A_fast'

# number of CPU cores available
# NOTE: num_cores should be adjusted according to the available memory in cluster
num_cores =  28 # multiprocessing.cpu_count()


# -----------------------------------------
# Preparation 
# -----------------------------------------
# write timestamp
fout=open('summary.txt','a')
starttime = time.clock()
fout.write('Timestamp: {:%Y-%b-%d %H:%M:%S}\n'.format(datetime.datetime.now()))

# set up regular expression for matching purpose
r = re.compile(".*cif")

path_orig = os.getcwd()
# change to directory where cifs are located
os.chdir(mof_dir)
# get parent directory of cif files
path_files=os.getcwd()
mof_list=os.listdir('.')

# -------------------------------------------------
# Randomly pick MOFs for training and testing
# -------------------------------------------------
# read data from pre-existing gcmc file
if mode == 1:
	# read in file (as pandas DataFrame)
	df = pd.read_csv(gcmc_dir,sep='\t')
	# convert to list (string format)
	ls_id = [str(x) for x in pd.Series.tolist(df['id'])]
	# check if enough data are available 
	if len(ls_id)<(n_train + n_test):
		print('Not enough data for requested number of training and testing data!')
		exit()
	# split data
	train_set = [x for x in ls_id[0:n_train] ]
	test_set = [x for x in ls_id[len(ls_id)-n_test:len(ls_id)]]
	# if mix & shuffle training and testing data
	if shuffle:
		all_set = train_set + test_set
		train_set = random.sample(all_set,n_train)
		test_set = [x for x in all_set if x not in train_set]

# no gcmc file available 
elif mode == 2:
	train_set = random.sample(mof_list,n_train)
	left_set = [x for x in mof_list if x not in train_set]
	test_set = random.sample(left_set,n_test)

# double check if training set has common elements with test set
if any(x in train_set for x in test_set):
	print('Common elements in training and testing set!')
	exit()


#------------------------------------------------------
# The heart of the code
#-------------------------------------------------------
def run_gridcpp(strc_index):

	os.chdir(path_files)
	os.chdir(strc_index)
	filelist = os.listdir('.')
	cif_file_name = list(filter(r.match,filelist))[0]
	
	# check if skip this structure if grids has already been calculated
	if checkgrid:
		if os.path.exists(rawgridname):
			return strc_index, cif_file_name, 0.0, 0, 0, 0


	# Calculate minimum size of simulation box 
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

	
	# Execute CPP
	timer_start = time.time()
	
	process = subprocess.run([LJ_BIN,cif_file_name,str(uc_x),str(uc_y),str(uc_z)])

	# write timing for individual CIF
	timer_end = time.time()
	elapsed_seconds = timer_end - timer_start
	
	# return results
	return strc_index, cif_file_name, elapsed_seconds, uc_x, uc_y, uc_z


# ------------ Training data set -------------
fout.write('Training Set' + '\n')
# Parallelize the job
# https://blog.dominodatalab.com/simple-parallelization/
results = Parallel(n_jobs = num_cores)(delayed(run_gridcpp)(cif) for cif in train_set)
# collect results
for imof in results:

	fout.write(imof[0] + '\t' + imof[1] + '\t' + str(round(imof[2], 2)) + '\n')
	#print(imof[3],imof[4],imof[5])
	fout.flush()


# ---------- Testing data set ---------------
fout.write('Testing Set' + '\n')
results = Parallel(n_jobs = num_cores)(delayed(run_gridcpp)(cif) for cif in test_set)
# collect results
for imof in results:

	fout.write(imof[0] + '\t' + imof[1] + '\t' + str(round(imof[2], 2)) + '\n')
	#print(imof[1],imof[3],imof[4],imof[5])
	fout.flush()


# close 'summary.txt'
fout.close()

