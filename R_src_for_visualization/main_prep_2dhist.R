# ===============================================================
# Main program for preparing energy/energy gradient 2D histogram
#   from raw 3D energy/energy-gradient grids
# 
# Author: Kaihang Shi
# Date: 5/22/2023
# ===============================================================

# How to use
# [Linux] In command prompt: 
#   Rscript --vanilla XX.R
# [RStudio] In console type:
# source('main_prep_2dhist.R')

rm(list=ls())

# ------------- preload  -------------
source("package_verification.R")
source("read_files.R")
source("analyze_functions.R")

# ------------ Parameters ----------------
# file name for original energy/energy gradient grid data
rawgridname = 'example_input/ener_grad_Kr_norm_0.5A.txt'   
# file name for output 2d histogram in R-native RDS format
tdhistname = "tdhist_Kr_example.rds"     

# histogram parameters
# bin width for energy histogram [kJ/mol]
ener_binwid = 2                    
# bin width for energy gradient histogram [kJ/mol/A]
grad_binwid = 66                   
# histogram range for energy in [kJ/mol] 
ener_range = c(-30,0)         
# histogram range for energy gradient in [kJ/mol/A]
grad_range = c(0,200)        



# --------------- Calculate 2D histogram ------------------------
# get_2d_hist is defined in analyze_functions.R
ls_hist = get_2d_hist(rawgridname,
                      ener_range,
                      grad_range, 
                      ener_binwid,
                      grad_binwid,
                      FALSE,
                      tdhist_type=2)
  
# write to RDS. RDS is a R's custom binary format 
write_rds(ls_hist,tdhistname)
  
  









