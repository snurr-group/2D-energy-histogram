# =======================================
# Functions for analyzing data
# =======================================

source("package_verification.R")
source("read_files.R")


# ============= Determine the lower bound of energy and virial =============
get_lo_bound = function(path,id_set,filename){
  
  # --------------------------------------------
  # Output: Return a list of lower bound for energy and virial [kJ/mol] of the entire input MOF set 
  # path      - full directory to the parent folder of the grid
  # id_set    - vector of IDs for the input MOF set 
  # filename  - file name for the cpp generated 2D energy/virial
  # --------------------------------------------
  
  # number of grid files
  num_grid <- length(id_set)
  
  # set up a progress bar
  #pb <- txtProgressBar(min = 1, max = num_grid, style = 3)
  
  # initialize minimum value
  e_min = 0
  v_min = 0 
  #pt = 0
  
  # loop over all grids
  for (id in id_set){
    
    grid_file = file.path(path,id,filename)
    # read_grid is defined in read_files.R, units [kJ/mol] for virial or [kJ/mol/A] for norm
    df_grid = read_grid(grid_file)
    
    if (e_min > min(df_grid$energy)){
      e_min = min(df_grid$energy)
    }
    
    if (v_min > min(df_grid$virial)){
      v_min = min(df_grid$virial)
    }
    #pt = pt + 1 
    #setTxtProgressBar(pb, pt)
  }
  #close(pb)
  
  ls_min = list(e_min = c(e_min), v_min = c(v_min))
  ls_min
  
}



# =============== 2D histogram constructor =======================
get_2d_hist = function(grid_file, e_min_max, v_min_max, e_wid, v_wid, normalize_ev = FALSE, tdhist_type){
  
  # --------------------------------------------
  # Output: Return a matrix of 2D energy-virial histogram (normalized, i.e., volume frac.)
  # Input:
  # grid_file   - full directory to the grid file
  # e_min_max   - vector of pre-set min and max energy value [kJ/mol] 
  # v_min_max   - vector of pre-set min and max virial value [kJ/mol] 
  # e_wid       - bin width for energy [kJ/mol]
  # v_wid       - bin width for virial [kJ/mol] 
  # normalize_ev - normalize energy and virial using the energy and virial minimum of each MOF, default - FALSE
  # tdhist_type - output type for 2d histogram; 
  #               1 - virial (having extra bins in both ends)
  #               2 - norm of energy gradient (only have one extra bins at the upper end)
  # --------------------------------------------
  
  # histogram breaks for energy
  e_bins = seq(from = e_min_max[1],
               to   = e_min_max[2],
               by   = e_wid)
  # histogram breaks for energy gradient
  v_bins = seq(from = v_min_max[1],
               to   = v_min_max[2],
               by   = v_wid)
  
  # read_grid is defined in read_files.R, units [kJ/mol]
  df_grid = read_grid(grid_file)
  
  if (normalize_ev) {
    # get the energy and virial minimum for this particular MOF
    e_min = min(df_grid$energy)
    v_min = min(df_grid$virial)
    
    df_grid$energy = df_grid$energy/abs(e_min)
    df_grid$virial = df_grid$virial/abs(v_min)
    
    # create a matrix 
    td_hist = matrix(0,nrow=(length(e_bins)), ncol = (length(v_bins)))
    
    # findInterval returns location of element in histogram breaks
    e_coord = findInterval(df_grid$energy,vec = e_bins)
    v_coord = findInterval(df_grid$virial,vec = v_bins)
    
    if(length(e_coord) != length(v_coord)) {stop("FATAL ERROR!")}
    
    # construct 2d map
    td_coord = cbind(e_coord,v_coord)
    
    # loop over all elements in the matrix
    for (i in 1:length(e_coord)) {
      
      td_hist[td_coord[i,1],td_coord[i,2]] = td_hist[td_coord[i,1],td_coord[i,2]] + 1
      
    }
    
    # total number of points in the 2D map
    ngrid = sum(td_hist)
    # double check 
    if(ngrid != length(df_grid$energy)) {stop("FATAL ERROR!")}
    
    # normalize 2d histogram
    td_hist = td_hist/ngrid
    # double check 
    if(!near(sum(td_hist),1)) {stop("SUM NOT TO 1!")}
    
    # calculate grid points corresponding to the matrix element
    # x - row (energy)
    x = seq(from = e_min_max[1] + e_wid/2.0,
            to   = e_min_max[2] + e_wid/2.0,
            by   = e_wid)
    # y - column (virial)
    y = seq(from = v_min_max[1] + v_wid/2.0,
            to   = v_min_max[2] + v_wid/2.0,
            by   = v_wid)
    
  } else {
    # ------------------------------
    if (tdhist_type == 1) {
      
      # adding extra 'lumping' bins on both ends of histogram
      # create a matrix
      td_hist = matrix(0,nrow=(length(e_bins)+1), ncol = (length(v_bins)+1))
      
      # findInterval returns location of element in histogram breaks
      e_coord = findInterval(df_grid$energy,vec = e_bins)
      v_coord = findInterval(df_grid$virial,vec = v_bins)
      
      if(length(e_coord) != length(v_coord)) {stop("FATAL ERROR!")}
      # construct 2d map
      td_coord = cbind(e_coord+1,v_coord+1)
      
      # loop over all elements in the matrix
      for (i in 1:length(e_coord)) {
        
        td_hist[td_coord[i,1],td_coord[i,2]] = td_hist[td_coord[i,1],td_coord[i,2]] + 1
        
      }
      
      # total number of points in the 2D map
      ngrid = sum(td_hist)
      # double check 
      if(ngrid != length(df_grid$energy)) {stop("FATAL ERROR!")}
      
      # normalize 2d histogram
      td_hist = td_hist/ngrid
      # double check 
      if(!near(sum(td_hist),1)) {stop("SUM NOT TO 1!")}
      
      # calculate grid points corresponding to the matrix element
      # x - row (energy)
      x = seq(from = e_min_max[1] - e_wid/2.0,
              to   = e_min_max[2] + e_wid/2.0,
              by   = e_wid)
      # y - column (virial)
      y = seq(from = v_min_max[1] - v_wid/2.0,
              to   = v_min_max[2] + v_wid/2.0,
              by   = v_wid)
    # ----------------------------  
    } else if (tdhist_type == 2) {
      
      # adding only one extra 'lumping' bins on gradient histogram 
      # create a matrix
      td_hist = matrix(0,nrow=(length(e_bins)+1), ncol = length(v_bins))
      
      # findInterval returns location of element in histogram breaks
      e_coord = findInterval(df_grid$energy,vec = e_bins)
      v_coord = findInterval(df_grid$virial,vec = v_bins)
      
      if(length(e_coord) != length(v_coord)) {stop("FATAL ERROR!")}
      # construct 2d map
      td_coord = cbind(e_coord+1,v_coord)
      
      # loop over all elements in the matrix
      for (i in 1:length(e_coord)) {
        
        td_hist[td_coord[i,1],td_coord[i,2]] = td_hist[td_coord[i,1],td_coord[i,2]] + 1
        
      }
      
      # total number of points in the 2D map
      ngrid = sum(td_hist)
      # double check 
      if(ngrid != length(df_grid$energy)) {stop("FATAL ERROR!")}
      
      # normalize 2d histogram
      td_hist = td_hist/ngrid
      # double check 
      if(!near(sum(td_hist),1)) {stop("SUM NOT TO 1!")}
      
      # calculate grid points corresponding to the matrix element
      # x - row (energy)
      x = seq(from = e_min_max[1] - e_wid/2.0,
              to   = e_min_max[2] + e_wid/2.0,
              by   = e_wid)
      # y - column (norm)
      y = seq(from = v_min_max[1] + v_wid/2.0,
              to   = v_min_max[2] + v_wid/2.0,
              by   = v_wid)
      
      
      
    }
    
  }

  # return list 
  ls_out = list(hist = td_hist, x = x, y = y, ngrid = ngrid)
  ls_out
  
}
  




# =================== Summarize energy/energy gradient distribution in a set of MOFs  ===================== 
get_ev_dist = function(path, id_set, filename, e_min_max, v_min_max, e_wid, v_wid, normalize_ev = FALSE) {
  
  # --------------------------------------------
  # This function is helpful to determine the boundaries for energy/virial histograms 
  # (see Table S5 in Ben's MSDE (2019) paper)
  # 
  # Output: Return a list containing two data frames, each of which stores counts of MOFs 
  #         having grid points falling into specified energy/virial bin
  # Input:
  # path        - full directory to the parent folder of the grid
  # id_set      - vector of IDs for the input MOF set
  # filename    - file name for the cpp generated original 2D energy/virial
  # e_min_max   - vector of pre-set min and max energy value 
  # v_min_max   - vector of pre-set min and max virial value 
  # e_wid       - bin width for energy
  # v_wid       - bin width for virial
  # normalize_ev - if normalize energy and virial using the energy and virial minimum of each MOF, default - FALSE
  # --------------------------------------------
  
  
  # if normalize the energy and virial using the minimum for each MOF
  if (normalize_ev) {
    
    # the reduced energy will always range from (-1,0)
    # assuming a small reduced e_wid
    e_wid = 0.025
    e_bins = seq(from = -1.0,
                 to   = 0,
                 by   = e_wid) %>% append(Inf)
    
    # assuming a small reduced v_wid
    v_wid = 0.1
    v_bins = seq(from = -1.0,
                 to   = 5.0,
                 by   = v_wid) %>% append(Inf)
    
  } else {
    
    # histogram breaks for energy
    e_bins = seq(from = e_min_max[1],
                 to   = e_min_max[2],
                 by   = e_wid) %>% append(Inf)
    # histogram breaks for virial
    v_bins = seq(from = v_min_max[1],
                 to   = v_min_max[2],
                 by   = v_wid) %>% append(Inf)
    
  }
  
  
  # number of grid files
  num_grid = length(id_set)
  
  # initialize
  e_cnts = data.frame(lower = e_bins[1:length(e_bins)-1], counts = integer(length(e_bins)-1))
  v_cnts = data.frame(lower = v_bins[1:length(v_bins)-1], counts = integer(length(v_bins)-1))
  
  #pt = 0
  # set up a progress bar
  #pb <- txtProgressBar(min = 1, max = num_grid, style = 3)
  
  # loop over all grid files in the id_set
  for (id in id_set) {
    
    grid_file = file.path(path,id,filename)
    # read_grid is defined in read_files.R, units [kJ/mol]
    df_grid = read_grid(grid_file)
    
    if(normalize_ev){
      
      # get the energy and virial minimum for this particular MOF
      e_min = min(df_grid$energy)
      v_min = min(df_grid$virial)
      
      df_grid$energy = df_grid$energy/abs(e_min)
      df_grid$virial = df_grid$virial/abs(v_min)
      
    }
    
    # calculate energy histogram
    e_hist = hist(df_grid$energy,breaks=e_bins,plot=F)
    
    # Modified on Jan 17, 2021
    # We report counts for gradient values that also satisfy the condition of energy <=0 kJ/mol
    df_grid = df_grid[df_grid$energy <= 0.0,]
    # calculate the gradient histogram
    v_hist = hist(df_grid$virial,breaks=v_bins,plot=F)
    
    nonzero_idx = which(e_hist$counts != 0)
    e_cnts$counts[nonzero_idx] = e_cnts$counts[nonzero_idx] + 1 
    
    nonzero_idx = which(v_hist$counts != 0)
    v_cnts$counts[nonzero_idx] = v_cnts$counts[nonzero_idx] + 1 
    
    #pt = pt + 1 
    #setTxtProgressBar(pb, pt)
    

  }
  #close(pb)
  
  list(energy = e_cnts, grad = v_cnts)
}  
  
  
  







