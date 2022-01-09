# ================================================
# Main program for diagnosing the MOF and energy histogram 
# 
# Author: Kaihang Shi
# =================================================

# How to use
# In command prompt: 
#   Rscript --vanilla xxxxxxxxx


rm(list=ls())

# ------------- preload  -------------
source("package_verification.R")
source("read_files.R")
source("analyze_functions.R")

# ------------ Basic parameters ----------------
# directory parameters 
sum_dir = "../pro10_ch3/summary.txt"          # full directory of 'summary.txt'
out_dir = "../big_data"                        # output directory
grid_dir = "/home/kaihang/2dhist/Tobacco"  # # parent directory where all grid&json files are located
rawgridname = 'ener_grad_CH3_norm_0.5A.txt'      # file name for original grid data

# histogram parameters
normalize_ev = FALSE               # if normalize the energy and virial to the same scale across database
#e_nbin = 30                    # number of bins for energy histogram
#v_nbin = 30                    # number of bins for virial histogram
ener_range = c(-80,2)           # histogram range for energy in kJ/mol 
grad_range = c(0,200)         # histogram range for energy gradient in kJ/mol (virial) or kJ/(mol*A) (norm)

# GCMC parameters (have not been used for the moment)
# prs = c(4.5)                # pressure for gcmc data 
# tmp = c(298.0)                # temperature for gcmc data in Kelvin
# ads = "Methane"             # adsorbate type
# a_units = "mol/kg"            # units for adsorption
# p_units = "bar"                # units for pressure
# constants 
k2kjmol = 0.00831446            # convert K to kJ/mol


# -------------- Read information about training/testing set ---------------
# Read summary.txt which contains information about training and testing set
# read_sum is defined in 'read_files.R'
print("START READING SUMMARY.TXT...")
ls_set = read_sum(sum_dir) 

# extract data frame
df_trainid = ls_set$train_set
df_testid = ls_set$test_set

# combine training and testing set
id_all = c(df_trainid$id,df_testid$id)


# ---------------- Analyze lower boundary of grid -----------------
print("START CALCULATING LOWER BOUND...")
# get_lo_bound defined in analyze_functions.R
ls_min = get_lo_bound(grid_dir,id_all,rawgridname)
# extract minimum value in kJ/mol
e_min = ls_min$e_min
v_min = ls_min$v_min
print(paste0(" energy minimum: ",e_min,
             " virial/norm minimum: ",v_min, " in kJ/mol or kJ/mol/A"))



# --------------- Calculate energy/energy gradient distribution  ------------------------
print("START CALCULATING DISTRIBUTION...")
# get_ev_dist defined in analyzed_functions.R
ls_dist = get_ev_dist(grid_dir,
                      id_all,
                      rawgridname,
                      ener_range,
                      grad_range,
                      1.0,
                      1.0,
                      normalize_ev)

print('energy distribution: ')
ls_dist$energy
print('energy gradient distribution: ')
ls_dist$grad










