# ========================================================
# This script does a bulk calculation of the persistent diagram/image
# All length units are in [Angstrom]
# 
# How to use: type the following command in command-line
# python persistent_homology.py diagram (or image)
# 
# Author: Kaihang Shi
# Email : kaihangshi0@gmail.com
# Last update: 7/12/2021 - make it a parallel version
# ========================================================

import pymatgen.core as pmcore
import diode as dd
import dionysus as dy
import numpy as np  
import matplotlib.pyplot as plt
import os 
import sys                
import re 
import math
import time 
import random
import pandas as pd
import flammkuchen as flk
# multiprocessing is pre-installed in Python distribution
import multiprocessing

from joblib import Parallel, delayed
from persim import PersistenceImager
from persim.images_weights import linear_ramp
from tqdm import tqdm

# ========================================================
# Input parameters
# ========================================================
# MOF structure path
mof_dir = '/home/kaihang/2dhist/Tobacco'

# gcmc path
gcmc_dir = '/home/kaihang/2dhist/energy-histograms/python_R/big_data/gcmc_Butane_298K_1.2Bar_cm3cm3.txt'

# pickle file name 
fh5_persdgm = 'persdgm_1.h5'
fcsv_persimg_1 = 'persimg_1D_1.csv'
fcsv_persimg_2 = 'persimg_2D_1.csv'
ftxt_log = 'output_1.txt'

# a set of MOFs to calculate
idx_start = 0
idx_end = 2

# number of CPU cores available
# multiprocessing.cpu_count()
# NOTE: num_cores should be adjusted according to the available memory in cluster
num_cores =  28 

# -- PERSISTENT DIAGRAM -- 
# Make a super cell with abc_norm^3 to normalize the data set 
# 250 A is at least twice as large as the abc of the largest MOFs in Tobacco 1.0
abc_norm = 125.0 #250.0

# exactness of alpha shape calculations (set to False by default, for quality run set it to True)
exactness = True

# -- PERSISTENT IMAGE --
# the alpha cutoff to remove noise homology 
# noisy value represents the roughness on the wall
# current value 2.25 is derived from PLD = 3 A
noise_alpcut = 2.25

# pixel size of persistence image [Angstrom]
pixel_size = 1.0

# Gaussian kernel sigma
sigma = 0.15

# if fit birth and persistence range according to the entire database
fit_range = True
# if stop right after determining the range to save time
fit_earlystop = True

# birth range
birth_range=(0.75, 39.75)
# persistence range
pers_range=(0.0, 16.0)



# ========================================================
# Main body
# ========================================================
# command-line argument
if len(sys.argv) >2:
	print('Only one argument is allowed!')
	exit()
# sys.argv[0] is the program ie. script name.
mode = sys.argv[1]

# current working directory
home = os.getcwd()

# calculate full persistent diagram
if mode == 'diagram':

	# read in file (as pandas DataFrame)
	df = pd.read_csv(gcmc_dir,sep='\t')
	# convert to list (string format)
	ls_id = [str(x) for x in pd.Series.tolist(df['id'])]
	data_set = [x for x in ls_id[idx_start:idx_end] ]

	# set up regular expression for matching purpose
	r = re.compile(".*cif")
	
	# log file
	fout=open(ftxt_log,'a')


	# define a function for parallel purpose
	def calc_persdgm(mof_id):

		timer_start = time.time()
		os.chdir(mof_dir)
		os.chdir(mof_id)
		filelist = os.listdir('.')
		cif_file = list(filter(r.match,filelist))[0]

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

		# a numpy array containing all cartesian coordiantes 
		# (points for persistent homology)
		points = struct.cart_coords

		# Calculate all simplices based on given points
		simplices = dd.fill_alpha_shapes(points,exact=exactness)

		# Perform alpha-shape filtration (ie nested alpha-complex)
		filtr = dy.Filtration(simplices)

		# Compute persistent homology
		ph = dy.homology_persistence(filtr)

		# initialize persistent diagram
		persdgm_all = dy.init_diagrams(ph,filtr)

		# convert dionysus object to numpy array for easy manipulating
		dict_tmp = {}

		# loop over all dimensions
		for dim in range(0,3):
			persdgm = np.empty(shape=(0,2))

			for pt in persdgm_all[dim]:
				persdgm = np.append(persdgm,np.array([[pt.birth,pt.death]]), axis=0)

			dict_tmp[dim] = persdgm

		# write timing for individual CIF
		timer_end = time.time()
		elapsed_seconds = timer_end - timer_start

		# return results
		return mof_id, dict_tmp, str(round(elapsed_seconds, 2)), str(struct.lattice.a), str(struct.lattice.b), str(struct.lattice.c) 


	##### Parallelize the job
	# https://blog.dominodatalab.com/simple-parallelization/
	output = Parallel(n_jobs = num_cores)(delayed(calc_persdgm)(mof_id) for mof_id in data_set)


	# initialize a empty dict
	results = {}

	# collect output 
	for res in output:

		# append to dict
		results[res[0]] = res[1]

		# write to log file
		fout.write(res[0] + '\t' + res[2] + '\t' +
			res[3] + '\t' + res[4] + '\t' +
			res[5] + '\n')
		fout.flush()

	os.chdir(home)

	# Update 3/26/21, pickle consumes huge amount of RAM for large object, eg 2GB RAM for 12 M file
	# save to pickle file 
	# fpik=open(fpik_persdgm,'wb')
	# pickle.dump(results,fpik)
	# fpik.close()

	# save to HDF5 file
	flk.save(fh5_persdgm,results)
	fout.close()

		

# Calculate persistent image
elif mode == 'image':

	# read in HDF5 file which contains full persistent diagram
	results = flk.load(fh5_persdgm)

	# -- define a processing function
	def PersImgs(results,dim,outfile):

		# initialize an empty list
		persdgm_dataset = []
		mofid_dataset = []

		# loop over all mof key 
		for (mof_id,dict_tmp) in tqdm(results.items()):

			# extract full persistent diagram for a single MOF
			#dict_tmp = results[mof_id]

			# initialize a new numpy array
			persdgm = np.empty(shape=(0,2))

			# loop over all original birth-death point in nD-homology
			for pt in dict_tmp[dim]:
				# only filter out important homology points (modified on 3/31/2021)
				#if pt[1] - pt[0] >= noise_alpcut:
				if pt[1] >= noise_alpcut:
					# convert alpha value to more intuitive radius of ball (this is time-consuming)
					r_birth = math.sqrt(pt[0])
					r_death = math.sqrt(pt[1])
					persdgm = np.append(persdgm,np.array([[r_birth,r_death]]), axis=0)

			# safety check
			if persdgm.shape[0] == 0:
				print('this mof has zero-array persistent diagram:')
				print(mof_id)
				print('need to modify the code!!! [3/31/2021]')

			# append to list
			persdgm_dataset.append(persdgm)
			mofid_dataset.append(mof_id)


		# initialize persistent image
		persimg = PersistenceImager(pixel_size = pixel_size, birth_range=birth_range, pers_range=pers_range)

		# determine birth and persistence range to capture all persistence pair; 
		# The option `skew=True` specifies that the diagram is currently in birth-death-
		# coordinates and must first be transformed to birth-persistence coordinates
		if fit_range:
			persimg.fit(persdgm_dataset,skew=True)

		# change weight function 
		persimg.weight = linear_ramp
		# standard linear ramp
		#persimg.weight_params = {'low':0.0, 'high':1.0, 'start':0.0, 'end':10.0}
		# no weighting (uniform function)
		persimg.weight_params = {'low':1.0, 'high':1.0, 'start':0.0, 'end':1.0}
		# this implementation remains a uniform one but can ensure persistence >=0 always
		#persimg.weight_params = {'low':0.0, 'high':1.0, 'start':0.0, 'end':1e-200}

		# set Gaussian sigma
		persimg.kernel_params = {'sigma': sigma}

		# print some info
		print('persistent image info:')
		print(persimg)

		if fit_earlystop:
			exit()

		# calculate persistent images
		pimgs = persimg.transform(persdgm_dataset,skew=True)

		# flatten the array
		pimgs_flat = np.empty(shape=(0,pimgs[0].size+1))
		indx = 0

		for img in pimgs:

			tmp_array = np.insert(img.flatten(),0,mofid_dataset[indx])
			pimgs_flat = np.append(pimgs_flat,np.array([tmp_array]),axis =0)
			indx = indx + 1

		# save final results to csv file
		np.savetxt(outfile,pimgs_flat,delimiter=",")

	# -- finished defining a function


	# Perform persistent image calculation for both 1D & 2D
	#PersImgs(results,1,fcsv_persimg_1)
	PersImgs(results,2,fcsv_persimg_2)

else:
	print('ERROR: UNKNOWN ARGUMENT!')
	exit()









