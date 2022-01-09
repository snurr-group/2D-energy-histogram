# ===============================================================
# Main program for preparing energy/energy gradient 2D histogram
#   and collecting GCMC data from json files (MOFdb)
# 
# Author: Kaihang Shi
# ===============================================================

# How to use
# In command prompt: 
#   Rscript --vanilla xxxxxxxxx


rm(list=ls())

# ------------- preload  -------------
source("package_verification.R")
source("read_files.R")
source("analyze_functions.R")

# ------------ Basic parameters ----------------
# full directory of 'summary.txt'
sum_dir = "../eth20/summary.txt"
# output directory
data_dir = "../big_data/"     
# parent directory where all grid&json files are located
grid_dir = "/home/kaihang/2dhist/Tobacco"  
# file name for original 2d grid data
rawgridname = 'ener_grad_CH3_norm_0.5A.txt'   
# if skip the MOF if tdhist.rds has already been created
checktdhist = TRUE        
# type of 2D histogram; 
# 1 - virial (having extra bins in both ends)
# 2 - norm of energy gradient (only have one extra bins at the upper end)
tdhist_type = 2
# file name for output 2d histogram 
tdhistname = "tdhist_CH3_norm_12x12_0.5A.rds"     

# histogram parameters
# bin parameters used (in real units) IF normalize_ev = FLASE
# bin width for energy histogram [kJ/mol]
ener_binwid = 2                    
# bin width for energy gradient histogram [kJ/mol] for virial, [kJ/mol/A] for norm
grad_binwid = 10                    
# histogram range for energy in [kJ/mol] 
ener_range = c(-26,0)         
# histogram range for virial in [kJ/mol] or norm in [kJ/mol/A]
grad_range = c(0,80)        


# # if normalize the energy and energy gradient to the same scale across database
# normalize_ev = FALSE             
# # bin parameters IF normalize_ev = TRUE
# # bin width for reduced energy 
# re_binwid = 0.05       
# # bin width for reduced virial
# rv_binwid = 0.1    
# # histogram range for reduced energy [energy/abs(min(energy))] 
# re_range = c(-1,0)         
# # histogram range for reduced virial 
# rv_range = c(-1,1)          


# if collect gcmc from JSON files 
readgcmc = FALSE    
# GCMC parameters for extracting data from JSON 
# pressure for gcmc data 
prs = 4.5    
# temperature for gcmc data in Kelvin
tmp = 298.0                
# adsorbate type
ads = "Methane"         
# units for adsorption
a_units = "cm3/cm3"            
# units for pressure
p_units = "bar"                

# constants. Convert K to kJ/mol
k2kjmol = 0.00831446           


# -------------- Read information about training/testing set ---------------
# Read summary.txt which contains information about training and testing set
# read_sum is defined in 'read_files.R'
ls_set = read_sum(sum_dir) 

# extract data frame
df_trainid = ls_set$train_set
df_testid = ls_set$test_set

# combine training and testing set
id_all = c(df_trainid$id,df_testid$id)

# ---------------- Collect GCMC data from JSON files (MOFdb) ---------------
if(readgcmc){
  
  # initialize an empty dataframe
  df_gcmc = data.frame(id=integer(0),loading=numeric(0))
  
  # Collect GCMC data and write to files for later use
  for (id in id_all) {
    
    cif_path = file.path(grid_dir,id)
    
    # json file path
    json_path = file.path(cif_path,list.files(cif_path,pattern = ".*json$"))
    
    # read_json_gcmc is defined in 'read_files.R'
    gcmcdata = read_json_gcmc(json_path,ads,a_units,prs,p_units,tmp)
    
    df_tmp = data.frame(id=id,loading=gcmcdata)
    
    df_gcmc = rbind(df_gcmc,df_tmp)
    
  }
  
  # set up output file name and path
  gcmc_filename = paste0('gcmc','_',ads,'_',prs,p_units,'_',tmp,'.txt')
  gcmc_path = file.path(data_dir,gcmc_filename)
  # write to file
  write.table(df_gcmc,gcmc_path,sep='\t')
  
}




# --------------- Calculate 2D histogram ------------------------
# set up progress bar
pb = txtProgressBar(min = 1, max = length(id_all), style = 3)
pt = 0

# # Normalize energy and virial using the absolute value of minimum energy/virial
# if (normalize_ev) {
#   # histogram parameters replaced by the reduced values
#   e_binwid = re_binwid                   
#   v_binwid = rv_binwid                   
#   e_range = re_range          
#   v_range = rv_range
# }

# loop over all grid files
for (id in id_all) {
  
  # progress bar
  pt = pt + 1 
  setTxtProgressBar(pb, pt)
  
  # set up 2D histogram file path
  tdhist_file = file.path(grid_dir,id,tdhistname)
  
  # check if tdhist.rds exists already
  if(checktdhist){
    if(file.exists(tdhist_file)){
      next
    }
  }
  
  
  grid_file = file.path(grid_dir,id,rawgridname)
  
  # get_2d_hist is defined in analyze_functions.R
  ls_hist = get_2d_hist(grid_file,
                        ener_range,
                        grad_range, 
                        ener_binwid,
                        grad_binwid,
                        FALSE,
                        tdhist_type)
  
  # write to RDS. RDS is a R's custom binary format 
  write_rds(ls_hist,tdhist_file)
  
  
  
}
# close progress bar
close(pb)








