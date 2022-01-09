# ================================================
# Functions for reading files
# ================================================

source("package_verification.R")

# ======================= Read summary.txt file ========================
read_sum = function(file){
  
  # ====================================================================================
  # Output: Return a list of data frames that contain id and name of MOFs files for both
  #         training and testing data set
  # Input:
  # file      - full directory to summary.txt
  # ====================================================================================
  
  con = file(file,"r")
  flag = 0
  while (TRUE) {
    
    ln = readLines(con, n = 1)
    
    if (length(ln) == 0) {
      break
    } else if (ln == "Training Set") {
      # initialize 
      train_id = c()
      train_name = c()
      flag = 1
      next
    } else if (ln == "Testing Set") {
      # initialize
      test_id = c()
      test_name = c()
      flag = 2
      next
    } 
    
    # save training set data 
    if (flag ==1) {
      ln = read.table(text = ln)
      #if (!is.integer(ln$V1)){next} # this was commented out because id is not always number
      train_id = append(train_id,ln$V1)
      train_name = append(train_name,str_remove(ln$V2,"cif"))
      # save testing set data
    } else if (flag == 2) {
      ln = read.table(text = ln)
      #if (!is.integer(ln$V1)){next}
      test_id = append(test_id,ln$V1)
      test_name = append(test_name,str_remove(ln$V2,"cif"))
      
    }
    
  }
  
  close(con)
  
  # convert to data frame
  train = data.frame(cbind(id = train_id, name = train_name))
  test  = data.frame(cbind(id = test_id, name = test_name))

  # return a list
  list(train_set = train,
       test_set  = test)
  
}




# ========= Read GCMC adsorption data from JSON file (from MOFdb) 10/29/20 ==============
read_json_gcmc = function(file, ad_name, ads_units, press, press_units, temp = 298){
  
  # ====================================================================================
  # Output: return a gcmc adsorption amount 
  # Input:
  # file        - full directory to the JSON file
  # ad_name     - adsorbate type for GCMC adsorption data 
  #               [Methane, Xenon, Krypton, Nitrogen, Hydrogen]
  # ads_units   - units for adsorption data 
  #               [mol/kg, cm3/cm3, kj/mol, ...]
  # press       - pressure
  # press_units - units for pressure 
  #               [bar, Pa]
  # temp        - temperature in Kelvin, default - 298 K
  # ====================================================================================

  # read data from file
  ls_data = fromJSON(file)
  df_gcmc = data.frame(ls_data$isotherms)
  
  # Get entry index for chosen adsorption units
  uid_a = which(df_gcmc$adsorptionUnits %in% ads_units)
  if (length(uid_a) == 0){stop("INVALID ADSORPTION UNITS!")}
  
  # Check if pressure units are consistent 
  uid_p = which(df_gcmc$pressureUnits %in% press_units)
  if (length(uid_p) == 0){stop("INVALID PRESSURE UNITS!")}
  
  # filter out the final valid isotherm
  uid_cm = intersect(uid_a,uid_p)
  id = c()
  for (i in uid_cm) {
    if(df_gcmc$adsorbates[[i]]$name == ad_name){
      id = append(id,i)
    }
  }
  # double check
  if (length(id)<1){
    print(file)
    stop("NO ENTRY WAS FOUND AT SPECIFIED CONDITIONS!")
  }
  if (length(id)>1){
    print(file)
    print(paste0("WARNING: ",length(id)," ENTRIES WERE FOUND AT SPECIFIED CONDITIONS!!"))
    print("       THIS MIGHT BE DUE TO DUPLICATE ISOTHERMS IN MOFDB!! FIX IT!")
    stop()
    # For this special case, we pick the first item for now. NEED FIX!
    #id = id[1]
  }
  
  # extract the final isotherm data
  df_data = df_gcmc$isotherm_data[[id]]
  tot_ad = df_data[df_data$pressure == press,]$total_adsorption
  if (length(tot_ad) == 0) {stop("NO PRESSURE DATA WAS FOUND!")}
  tot_ad
}



# ============== Read textural properties from the JSON file (from MOFdb) ================
read_json_texprop = function(file){
  
  # ====================================================================================
  # Output: return a list of textural properties available in JSON files 
  # Input:
  # file        - full path to the JSON file
  # ====================================================================================
  
  # read data from file
  ls_data = fromJSON(file)
  
  # largest cavity diameter in Angstrom
  lcd = ls_data$lcd
  
  # pore limiting diamter in Angstrom
  pld = ls_data$pld
  
  # void fraction
  void_frac = ls_data$void_fraction
  
  # surface area
  area_g = ls_data$surface_area_m2g
  area_v = ls_data$surface_area_m2cm3
  
  # return a list
  list(lcd = lcd,
       pld = pld,
       void_frac = void_frac,
       area_g = area_g,
       area_v = area_v)

}



# =============== Read energy/virial grid information ==================
read_grid = function(file){
  
  # ====================================================================================
  # Output: return a data frame containing energy and virial both in kj/mol
  # Input: 
  # file - full directory to grid file
  # ====================================================================================
  
  # constants 
  k2kjmol = 0.00831446            # convert K to kJ/mol
  
  df_raw = read.table(file)
  
  # columns V4 and V5 store energy and virial
  # columns V1-V3 store grid Cartesian coordinates
  df_grid = dplyr::select(df_raw,V4,V5)
  names(df_grid) = c("energy","virial")
  # convert units 
  df_grid$energy = df_grid$energy * k2kjmol
  df_grid$virial = df_grid$virial * k2kjmol
  
  df_grid
  
}





# ======== Read all 2D histogram files (into vector) and pre-collected GCMC data ========
read_hist_gcmc = function(hist_path,tdhistname,gcmc_file,idset,nonzero_load = FALSE,byrow=FALSE) {
  
  # ======================================================================
  # Output: return a list consisting of flatted 2D histogram (data frame)
  #         and the corresponding GCMC adsorption data (vector) for all MOFs in the set;
  #         number of row and number of column of the 2D histogram matrix are also
  #         stored for later conversion from vector back to matrix
  # Input:
  # hist_path   - full directory to the parent folder of the 2D histogram
  # tdhistname  - file name for processed 2d histogram rds file
  # gcmc_file   - file directory to the pre-collected GCMC data for the
  #               corresponding training or testing MOF set
  # idset       - IDs of MOFs that are going to be sampled
  # nonzero_gcmc - flag to exclude zero gcmc loading. Default to include all gcmc loadings.
  #                 This option was designed to exclude bad data entries with fake zero loading.
  # byrow        - convert matrix to a 1D vector by row 
  # ======================================================================
  
  # initialize empty data frame
  df_x = data.frame()
  
  # read pre-collected GCMC data
  df_gcmc = read.table(gcmc_file)
  
  # convert id in gcmc file to string format (7-18-2021)
  df_gcmc$id = as.character(df_gcmc$id)
  
  # extract assigned IDs for MOFs
  #id_set = df_gcmc$id
  # extract corresponding MOF names (have not been used)
  #name_set = df_gcmc$name
  
  
  # loop over all histogram files
  for (id in idset) {
    
    # directory to the pre-calculated histogram file
    hist_file = file.path(hist_path,id,tdhistname)
    
    # check if file exist, if not, report error
    if(!file.exists(hist_file)){
      stop(paste0("tdhist file does not exist for id ",id))
    }
    
    # read in rds object (a R-specific binary file), it returns a list
    ls_hist = read_rds(hist_file)
    
    # extract normalized 2d histogram (matrix)
    mx_hist = ls_hist$hist
    
    # convert matrix to a 1D vector 
    # by row
    if (byrow) {
      vec_hist = as.vector(t(mx_hist))
      
    } else {
      # column by column, default of R 
      vec_hist = as.vector(mx_hist)
      
    }
    
    # convert vector to a temporary data frame
    df_tmp = data.frame(rbind(vec_hist))
    # mutate() adds new variables and preserves existing ones
    df_tmp = mutate(df_tmp,id=id)
    
    # combine data frame
    df_x = rbind(df_x,df_tmp)

    
  }
  
  # save dimension of the matrix for later conversion from vector to matrix
  n_row = nrow(mx_hist)
  n_col = ncol(mx_hist)
  
  # save coordinates for later plotting
  ener_coord = ls_hist$x
  norm_coord = ls_hist$y
  
  # joint data frame
  df_joint = df_x %>%left_join(df_gcmc, by="id") 
  
  # exclude entries with zero gcmc loading
  if (nonzero_load) {
    df_joint = df_joint[!(df_joint$loading == 0),]
  }
  
  # target value (loading) is a vector
  #gcmc_loading = df_joint$loading
  
  # return a list 
  ls_xy = list(df_xy = df_joint,
               hist_nrow = n_row,
               hist_ncol = n_col,
               xcoord = ener_coord,
               ycoord = norm_coord,
               byrow = byrow)
  

}




# ====== Read all 2D histogram files (in matrix) and pre-collected GCMC data =======
read_hist_gcmc_mx = function(hist_path,tdhistname,gcmc_file,idset) {
  
  # ======================================================================
  # Output: return a list consisting of an array of 2D histogram
  #         and the corresponding GCMC adsorption data (matrix) for all MOFs in the set;
  #         number of row and number of column of the 2D histogram matrix are also
  #         stored for later conversion from vector back to matrix
  # Input:
  # hist_path   - full directory to the parent folder of the 2D histogram
  # tdhistname  - file name for processed 2d histogram rds file
  # gcmc_file   - file directory to the pre-collected GCMC data for the
  #               corresponding training or testing MOF set
  # ======================================================================
  
  # read pre-collected GCMC data
  df_gcmc = read.table(gcmc_file)
  
  # convert id in gcmc file to string format (7-18-2021)
  df_gcmc$id = as.character(df_gcmc$id)

  # read a hist file first and extract basic info
  hist_file = file.path(hist_path,idset[1],tdhistname)
  ls_hist = read_rds(hist_file)
  mx_hist = ls_hist$hist
  # save dimension of the matrix for later conversion from vector to matrix
  n_row = nrow(mx_hist)
  n_col = ncol(mx_hist)
  # save coordinates for later plotting
  ener_coord = ls_hist$x
  norm_coord = ls_hist$y
  
  # initialize array for x data
  ary_x = array(0, dim=c(length(idset),n_row,n_col))
  df_x_id = data.frame()
  pt = 1
  
  # loop over all histogram files
  for (id in idset) {
    
    # directory to the pre-calculated histogram file
    hist_file = file.path(hist_path,id,tdhistname)
    
    # check if file exist, if not, report error
    if(!file.exists(hist_file)){
      stop(paste0("tdhist file does not exist for id ",id))
    }

    # read in rds object (a R-specific binary file), it returns a list
    ls_hist = read_rds(hist_file)
    
    # extract normalized 2d histogram (matrix)
    mx_hist = ls_hist$hist
    
    # assign matrix to the array
    ary_x[pt,,] = mx_hist
    
    # combine data frame for x_id
    df_tmp = data.frame(id= id)
    df_x_id = rbind(df_x_id,df_tmp)
    
    # update counter
    pt = pt + 1
    
    
  }
  
  # joint data frame
  df_joint = df_x_id %>%left_join(df_gcmc, by="id") 
  
  
  # target value (loading) is a matrix
  mx_y = matrix(df_joint$loading)
  
  # return a list 
  ls_xy = list(x = ary_x,
               y = mx_y,
               df_id = df_x_id,
               hist_nrow = n_row,
               hist_ncol = n_col,
               xcoord = ener_coord,
               ycoord = norm_coord)
  
  
}





# ============== Read in GCMC and training/testing data set from a pre-existing file ==========
prep_gcmc = function(file,ftype,xlsx_tab = 'Sheet1'){
  
  # ======================================================================
  # Output: Return a txt file following my gcmc 'big_data' format. The output 
  #         file can be used directly with the following ML workflow.
  # Input:
  # file      - pre-existing GCMC file from Zhao Li
  # ftype     - type of file; 'txt' or 'xlsx'
  # xlsx_tab  - sheet name in a multi-sheet xlsx file
  # ndata     - total number of data points for recollection
  # ======================================================================
  
  if (ftype == 'txt') {
    df_data = read_table2(file)
    
    # initiate an empty data frame 
    #df_new = data.frame('id'=integer(0),'loading'=numeric(0))
    df_new = data.frame()
    
    # screen bad data entries in file
    df_new = df_data[!is.na(df_data$ID),] %>% dplyr::select(.,ID,Molec_cm3overcm3)
    
    # rename
    names(df_new) = c('id','loading')
    
    # write to file
    write.table(df_new,'gcmc_Kr_10bar_273_cm3cm3.txt',sep='\t')
    
  } else if (ftype == 'xlsx'){
    
    df_data = read_excel(file,sheet=xlsx_tab)
    
    df_tmp = data.frame()
    
    df_tmp = dplyr::select(df_data,MOF.ID,"GCMC_Uptake [cm3/cm3]")
    
    names(df_tmp) = c('id','loading')
    
    # remove duplicated MOFs
    vec_dupid = df_tmp$id[duplicated(df_tmp$id)]
    df_tmp = df_tmp[!duplicated(df_tmp$id),]
    print(paste0('duplicated ID:',vec_dupid))
    
    
    # randomly shuffle rows in dataframe
    shuffled_rows = sample(nrow(df_tmp))
    df_shuffled = df_tmp[shuffled_rows,]
    
    # write to file
    write.table(df_shuffled,paste0('gcmc_',xlsx_tab,'_cm3cm3.txt'),sep='\t')
    
  }

}


# ========= Read in pre-existing textural properties from CSV file ===================
read_csv_texprop = function(os = 'win',extra_features) {
  
  # ====================================================================================
  # Output: Return a data frame containing selected textural properties
  # Input:
  # os              - operianal system which specifies different directory for csv file
  # extra_features  - a vector containing extra features to feed into ML model, defined in main program 
  # ====================================================================================
  
  if(os == 'win'){
    csv_file = 'C:/Users/khshi/Dropbox/Projects/2D_Energy_Histogram/code/All_data/textural_properties/textprop_tobacco_consistentnew.csv'
    
  } else if (os == 'linux') {
    csv_file = '/home/kaihang/2dhist/energy-histograms/python_R/big_data/textprop_tobacco.csv'
  }

  # read in textural properties  
  df_texprop = read.csv(csv_file)
  df_texprop_sel = dplyr::select(df_texprop,c('id',extra_features))
  
  # convert id to character
  df_texprop_sel$id = as.character(df_texprop_sel$id)
  
  df_texprop_sel
}





# ========= Prepare a data frame that contains features, loading and textural properties ========
read_all = function(hist_path = 'E:/Research_data/2Dhistogram/tobdown',
                    sum_dir = "E:/Research_data/2Dhistogram/hex10/summary.txt",
                    tdhistname = "tdhist_CH3_norm_14x5_0.5A.rds",
                    gcmc_file = "C:/Users/khshi/Dropbox/Projects/2D_Energy_Histogram/code/All_data/gcmc_Hexane_495K_10Bar_cm3cm3.txt",
                    idset = NULL,
                    nonzero_load = FALSE,
                    byrow=FALSE,
                    extra_features=c('vf','sa_tot_m2cm3','pld','lcd')){
  
  # ====================================================================================
  # Output: Return a list containing data for ML workflow (features, labels etc)
  # See previous functions for the definition of the input parameters
  # ====================================================================================
  
  # in case no idset is provided
  if (is.null(idset)) {
    # Read summary.txt which contains information about training and testing set
    # read_sum is defined in 'read_files.R'
    ls_set = read_sum(sum_dir) 
    # extract data frame
    vec_trainid = ls_set$train_set$id
    vec_testid = ls_set$test_set$id
    idset = c(vec_trainid,vec_testid)
  }
  
  # function defined in read_files.R
  ls_xy = read_hist_gcmc(hist_path = hist_path,
                         tdhistname = tdhistname,
                         gcmc_file = gcmc_file,
                         idset = idset,
                         nonzero_load = nonzero_load,
                         byrow = byrow)
  
  # extract specified features and loading
  df_joint = ls_xy$df_xy
  df_joint = dplyr::filter(df_joint, df_joint$id %in% idset) 
  
  # get textural properties
  # function defined in 'read_files.R'
  df_texprop_sel = read_csv_texprop(extra_features = extra_features)
  
  # add textural properties to the data frame
  df_joint = df_joint %>% left_join(df_texprop_sel, by="id")
  
  # return a list
  ls_out = list(df_xy = df_joint,
                hist_nrow = ls_xy$hist_nrow,
                hist_ncol = ls_xy$hist_ncol,
                xcoord = ls_xy$xcoord,
                ycoord = ls_xy$ycoord,
                byrow = byrow)

}




# ======== Read persistent image data from existent CSV file ========
read_pers_image = function(csv_file) {
  
  # ======================================================================
  # Output: return a data frame containing id name of Tobacco MOFs and 
  #         persistent image pixels. One pixel per column. 
  # Input:
  # csv_file   - full directory to the CSV file storeing persistent image features
  # ======================================================================
  
  # read in csv data
  df_persimg = read.csv(csv_file,header = F)
  
  # change first column name to id
  names(df_persimg)[1]='id'

  # return dataframe
  df_persimg
  
}
















