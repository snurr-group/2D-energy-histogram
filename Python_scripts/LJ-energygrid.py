# Python script to calculate the energy/energy gradient grid on a bunch of mofs, report that 
# number and also output the grid energy/gradient values for every mof in the chosen set.
# Calls LJ calculator in bin/energyhist


import os
import datetime
import time 
import sys
import subprocess
import random
import re
import pandas as pd

# ------------------------------------------
# Control parameters 
# ------------------------------------------
# Specify mode
# 1 - read from gcmc file; 2 - random sample from entire pool
smode = 1
shuffle = True

# gcmc path
gcmc_dir = '/home/kaihang/2dhist/energy-histograms/python_R/big_data/gcmc_Kr_273K_10Bar_cm3cm3.txt'

# Specify parameters
n_train = 1000       
n_test = 1000

# MOF structure path
mof_dir = '/home/kaihang/2dhist/Tobacco'

# if skip a MOF if raw grid file is already there
checkgrid = True
rawgridname = 'ener_grad_Kr_norm_0.5A.txt'

# path to C++ code for energy/virial grid calculations
LJ_BIN = '/home/kaihang/2dhist/energy-histograms/bin/grid_Kr_norm_0.5A'


# -----------------------------------------
# Preparation 
# -----------------------------------------
# write timestamp
fout=open('summary.txt','a')
starttime = time.clock()
fout.write('Timestamp: {:%Y-%b-%d %H:%M:%S}\n'.format(datetime.datetime.now()))

# set up regular expression for matching purpose
r = re.compile(".*cif")

# ---------------------------------------------
# Prepare dir structure
# ---------------------------------------------
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
if smode == 1:
	# read in file (as pandas DataFrame)
	df = pd.read_csv(gcmc_dir,sep='\t')
	# convert to list (string format)
	ls_id = [str(x) for x in pd.Series.tolist(df['id'])]
	# check if enough data are available 
	if len(ls_id)<(n_train + n_test):
		print('Not enough data for requested number of training and testing data!')
		exit()
	# only n_test controls the partition of data
	train_set = [x for x in ls_id[0:n_train] ]
	test_set = [x for x in ls_id[len(ls_id)-n_test:len(ls_id)]]
	# if mix & shuffle training and testing data
	if shuffle:
		all_set = train_set + test_set
		train_set = random.sample(all_set,n_train)
		test_set = [x for x in all_set if x not in train_set]

# no gcmc file available 
elif smode == 2:
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
# define a function for the general use
def loop_rungridcpp(id_set):

	for strc_index in id_set:

		os.chdir(strc_index)
		filelist = os.listdir('.')
		cif_file_name = list(filter(r.match,filelist))[0]
		
		# check if skip this structure if 2d hist has already been calculated
		if checkgrid:
			if os.path.exists(rawgridname):
				fout.write(strc_index + '\t' + cif_file_name + '\t' + str(0.0) + '\n')
				fout.flush()
				os.chdir(path_files)
				continue

		
		timer_start = time.time()
	
		subprocess.check_output([LJ_BIN, cif_file_name])

		# write timing for individual CIF
		timer_end = time.time()
		elapsed_seconds = timer_end - timer_start

		fout.write(strc_index + '\t' + cif_file_name + '\t' + str(round(elapsed_seconds, 2)) + '\n')
		fout.flush()
		os.chdir(path_files)

# processing
fout.write('Training Set' + '\n')
loop_rungridcpp(train_set)

fout.write('Testing Set' + '\n')
loop_rungridcpp(test_set)


os.chdir(path_orig)
#print overall timing
endtime = time.clock()
elaptime = endtime-starttime
fout.write('Timestamp: {:%Y-%b-%d %H:%M:%S}\n'.format(datetime.datetime.now()))
fout.write('Elapsed time\t%f\n' % (elaptime))
fout.close()
