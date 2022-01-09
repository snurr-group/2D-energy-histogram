# ------------------------------------------------------------------
# This script manages to calculate textural properties using Zeo++
# This is a parallel version
#
# Kaihang Shi  Created on 6-22-2021
# Last modified on 6-24-2021 for coremof
# ------------------------------------------------------------------

# load packages
import re
import os
import pandas as pd
import subprocess
import time
# multiprocessing is pre-installed in Python distribution
import multiprocessing
from joblib import Parallel, delayed
from shutil import copy2


# Path to cif files
cif_dir = '/gpfs_backup/gubbins_data/kshi3/Textural_properties/coremof2014/'

# range of files for calculation
num_st = 0
num_end = 5

# csv file name
csv_name = 'coremof2014.csv'

# number of CPU cores available
# NOTE: num_cores should be adjusted according to the available memory in cluster
num_cores =  10 # multiprocessing.cpu_count()

# current path
home = os.getcwd()
os.chdir(cif_dir)
filelist = os.listdir('.')[num_st:num_end]
os.chdir(home)

# compile regular expression
#regex = re.compile(r'\d+')
# For CoREMOF, we take the full name 
regex = re.compile("(.*)\.cif")

# checking function
def is_non_zero_file(fpath):  
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

# define a function
def zeopp(cif):
	# copy file to the current directory
	copy2(cif_dir+cif,home)

	# zeopp output file name
	outres = cif+'.res'
	outsa = cif + '.sa'

	# execute Zeo++ command (first in the list is command, rest are the arguments and flags)
	process = subprocess.run(['network','-ha','-r','UFF.rad','-res',outres,'-sa','1.86','1.86','1000',outsa,cif])

	# save id name
	#mofid = int(regex.findall(cif)[0])
	mofid = regex.findall(cif)[0]

	# check if file exist
	if is_non_zero_file(outres):
		# check pore size (only one line in the file)
		fres = open(outres).readlines()[0]
		fres = fres.strip().split()
		lcd = fres[1]
		pld = fres[2]
		lcd_free = fres[3]

		subprocess.run(['rm',outres])
	else:
		lcd = float("nan")
		pld = float("nan")
		lcd_free = float("nan")



	if is_non_zero_file(outsa):
		# check surface area (first line is the data)
		fsa = open(outsa).readlines()[0]
		fsa = fsa.strip().split()

		sa_acc_m2cm3 = float(fsa[9])
		sa_acc_m2g = float(fsa[11])
		sa_tot_m2cm3 = float(fsa[9]) + float(fsa[15])
		sa_tot_m2g = float(fsa[11]) + float(fsa[17])

		subprocess.run(['rm',outsa])
	else:
		sa_acc_m2cm3 = float("nan")
		sa_acc_m2g = float("nan")
		sa_tot_m2cm3 = float("nan")
		sa_tot_m2g = float("nan")


	# delete local files
	subprocess.run(['rm',cif])

	# return results
	return mofid, lcd, pld, lcd_free, sa_acc_m2cm3, sa_tot_m2cm3, sa_acc_m2g, sa_tot_m2g



starttime = time.time()
# Parallelize the job
# https://blog.dominodatalab.com/simple-parallelization/
results = Parallel(n_jobs = num_cores)(delayed(zeopp)(cif) for cif in filelist)


# initialize list
idname = []
sa_acc_m2g = []
sa_acc_m2cm3 = []
sa_tot_m2g = []
sa_tot_m2cm3 = []
pld = []
lcd = []
lcd_free = []

# collect results
for imof in results:

	idname.append(imof[0])
	lcd.append(imof[1])
	pld.append(imof[2])
	lcd_free.append(imof[3])
	sa_acc_m2cm3.append(imof[4])
	sa_tot_m2cm3.append(imof[5])
	sa_acc_m2g.append(imof[6])
	sa_tot_m2g.append(imof[7])


# save results to list
# convert all data to pandas data frame type
dic = { 'id':idname, 
		'sa_acc_m2g':sa_acc_m2g,
		'sa_tot_m2g':sa_tot_m2g,
		'sa_acc_m2cm3':sa_acc_m2cm3,
		'sa_tot_m2cm3':sa_tot_m2cm3,
		'pld':pld,
		'lcd':lcd,
		'lcd_free':lcd_free
		}

df = pd.DataFrame(dic)

# save to csv
df.to_csv(csv_name)

# print computational time
print('Time taken = {} seconds'.format(time.time() - starttime))
