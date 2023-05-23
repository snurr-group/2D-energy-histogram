# =======================================
# Functions for making plots 
# =======================================

source("package_verification.R")
source("read_files.R")
# install ggpatern
# remotes::install_github("coolbutuseless/ggpattern")
library("ggpattern")
library("inlmisc")

# Some pre-defined color palettes
colr_spectrum1 = c('#507bbf','#75b44b','#f5ee48','#e19c25','#cc2127')
colr_spectrum2 = wes_palette("Zissou1", 50, type = "continuous")
colr_magma_incr = viridis(256,option='magma', direction = -1)
colr_gray2purple = c()
# https://waterdata.usgs.gov/blog/tolcolors/
colr_rainbow_full = inlmisc::GetColors(512, scheme = "smooth rainbow",bias=1.2)
colr_rainbow_b2r = inlmisc::GetColors(512, scheme = "smooth rainbow", start = 0.3, end = 0.9)
colr_rainbow_full_discrete = inlmisc::GetColors(40, scheme = "smooth rainbow")
# some standard colors good for discrete color plots 
colr_std = c('#000000', # rich black
             '#BBB8B8', # light gray
             "#0077be", # dark blue
             "#131d64", # darker blue
             "#75AADB", # R blue (light)
             "#CC0000", # NCSU red (dark)
             "#7EB875", # smooth rainbow green (light)
             "#007c42", # JCTC green (dark)
             "#E59036", # smooth rainbow orange (light)
             "#D0B440", # smooth rainbow yellow (light)
             "#824D99", # smooth rainbow purple (light)
             '#4E2A84', # NW purple (dark)
             # gradual change
             "#cae7c8",# light green
             "#4fbbd5",
             "#2472b8",
             "#131d64" # dark blue
             )



# ============ Function to insert minor ticks (use with ggplot2) ================
# http://web.csulb.edu/~tgredig/research/report.ggplot.html
mark_major = function(major_labs, n_minor) {
  labs =  c( sapply( major_labs, function(x) c(x, rep("", n_minor) ) ) )
  labs[1:(length(labs)-n_minor)]
}


# =============== General theme function for ggplot ============
my_theme_general = function(){
  
  # modify other elements
  theme(panel.background = element_rect(fill='transparent'),
        # create a panel border (similar to the style in OriginLab)
        panel.border = element_rect(fill = NA, linetype = "solid", size = 1.5),
        # remove grid line
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # specify axis text/title size
        axis.text.y   = element_text(size=28,colour = 'black'),
        axis.text.x   = element_text(size=28,colour = 'black'),
        axis.title.y  = element_text(size=28,colour = 'black',face='bold'),
        axis.title.x  = element_text(size=28,colour = 'black',face ='bold'),
        # ticks specification
        axis.ticks.length = unit(6,'pt'),
        axis.ticks = element_line(color = 'black',linetype = 'solid'),
        # legend
        legend.key = element_rect(colour = "transparent",fill = NA),
        legend.background = element_blank(),
        # set up plot margin 
        plot.margin = ggplot2::margin(10,30,10,10))
  
}


# =============== Theme function for ggplot heatmap ============
my_theme_heatmap = function(){
  
  # modify other elements
  theme(panel.background = element_rect(fill='transparent'),
        # create a panel border (similar to the style in OriginLab)
        panel.border = element_rect(fill = NA, linetype = "solid", size = 1.5),
        # remove grid line
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        # specify axis text/title size, size = 28 (default)
        axis.text.y   = element_text(size=35,colour = 'black'),
        axis.text.x   = element_text(size=35,colour = 'black'),
        axis.title.y  = element_text(size=32,colour = 'black',face='bold'),
        axis.title.x  = element_text(size=32,colour = 'black',face = 'bold'),
        # ticks specification
        axis.ticks.length = unit(6,'pt'),
        axis.ticks = element_line(color = 'black',linetype = 'solid'),
        # legend
        legend.key = element_rect(colour = "transparent",fill = NA),
        legend.background = element_blank(),
        # size = 25 (default)
        legend.text = element_text(size=30),
        legend.title= element_text(size=30),
        # set up plot margin 
        #plot.margin = ggplot2::margin(10,30,10,10)
        ) 
}


# ============= Save image function =================
fig_save = function(p,save,fname,w=9,h=8){
  
  # --------------------------------------------
  # Output: a saved figure
  # Input: 
  # p - ggplot object
  # save - flag to save the file
  # fname - figure name
  # --------------------------------------------
  
  if (save) {
    # take out file name without the suffix
    name = paste0(fname,'.tif')
    # save figure
    ggsave(filename=name,
           plot = p,
           device = "tiff",
           width = w,
           height = h,
           units = 'in',
           dpi = 300,
           compression = 'lzw')
  }
}



# =============== A general plain plot for paper ================
plot_learning_curve = function(modrds,save=FALSE,fname='untitled'){
  
  # --------------------------------------------
  # Output: Return a general plain plot
  # Input: 
  # modrds - RDS file for saved model
  # save - flag to save the file
  # fname - figure name
  # --------------------------------------------
  
  # predefined colors
  colr = c('#4E2A84','#3182BD')
  # predefined shape
  pt_shape = c(16,15)
  
  # read in file
  mod = read_rds(modrds)
  df_lcv = mod$lcv
  df_new = df_lcv %>% gather(key='type',value = 'RMSE',-x)
  
  # make the plot
  p = ggplot(df_new,aes(x=x, y =RMSE)) + 
    geom_line(mapping = aes(color = type), size = 2) + 
    geom_point(mapping = aes(color = type), size = 5) + 
    scale_color_manual(breaks = c('train','test'),values = colr) + 
    xlab('Number of training data') +
    ylab(expression(bold(paste('RMSE (',cm[STP]^{3},'/',cm^{3},')'))))+ 
    # my_theme function applys standard theme for paper, function defined in 'make_plot.R'
    my_theme_general() + 
    # adjust legend position manually
    theme(legend.position = c(0.9, 0.9),
          legend.title = element_blank(),
          legend.text = element_text(size=28))

  # save plot, function defined in 'make_plot.R'
  fig_save(p,save,fname)
  
  # output the plot
  p
  
}


# =============== Plot a heatmap for 2D histogram ================
plot_2dhist = function(tdhist_rds,colr_trans= TRUE,barwd= 1.5,barht=25,save=FALSE,fname='2dheatmap',width=9,height=6.75){
  
  # --------------------------------------------
  # Output: Return 2d heat map figure for 2D histogram
  # Input:
  # tdhist_rds          - RDS file containing histogram object created by get_2d_hist function
  # colr_trans          - transform the value with logarithm to have a better looking
  # barwd                - bar width for plotting
  # barht                - bar height for plotting
  # --------------------------------------------
  
  # read in rds object
  ls_hist = read_rds(tdhist_rds)

  # extract total number of grid points
  ngrid = ls_hist$ngrid
  
  # extract normalized 2d histogram
  hist_norm = ls_hist$hist
  # get coordinates
  ener_coord = ls_hist$x
  norm_coord = ls_hist$y
  
  # scale the histogram value for good-looking plot
  #hist_std = hist_norm*ngrid
  #hist_log =  log(hist_std)
  # set -Inf to -1.1 for plotting purpose
  #hist_log[is.infinite(hist_log)] = -1.1
  
  # transform the data for ggplot
  # expand.grid - https://blog.csdn.net/qq_27586341/article/details/91040908
  vec_hist = as.vector(hist_norm)
  df_heatmap = expand.grid(X=ener_coord,Y=norm_coord)
  df_heatmap = mutate(df_heatmap,Z=vec_hist)
  
  # make a heat map plot
  heatplot = ggplot(df_heatmap)+
    geom_tile(mapping = aes(x=X, y=Y, fill = Z)) 
  
  # perform value transform
  if (colr_trans) {
    
    heatplot = heatplot + scale_fill_gradientn(colours= append(c('#FFFFFF'),colr_magma_incr), 
                                               name = 'Vol. Frac.', 
                                               trans = scales::pseudo_log_trans(sigma = 1e-4), 
                                               #oob = scales::squish_infinite,
                                               #na.value = 'transparent',
                                               breaks = c(0,1e-3,1e-2,1e-1,1), 
                                               labels = c(0,1e-3,1e-2,1e-1,1), 
                                               limits = c(0,1)) 
      
  } else {
    
    heatplot = heatplot + scale_fill_gradientn(colours= append(c('#FFFFFF'),colr_magma_incr), 
                                               name = 'Vol. Frac.', 
                                               #trans = scales::pseudo_log_trans(sigma = 1e-4), 
                                               #oob = scales::squish_infinite,
                                               #na.value = 'transparent',
                                               #breaks = c(0,1e-3,1e-2,1e-1,1), 
                                               #labels = c(0,1e-3,1e-2,1e-1,1), 
                                               limits = c(0,1)) 
    
  }
  
  # add other components 
  heatplot = heatplot + scale_x_continuous(expand=c(0,0))+
    scale_y_continuous(expand=c(0,0))+
    xlab('Energy (kJ/mol)') +
    ylab(expression(bold(paste('Energy gradient (kJ/mol/',ring(A),')')))) + 
    # theme for heatmap, function defined in 'make_plot.R'
    my_theme_heatmap()+ 
    # legend bar control
    guides(fill = guide_colourbar(barwidth = barwd, barheight = barht))
  
  # save plot, function defined in 'make_plot.R'
  fig_save(heatplot,save,fname,w=width,h=height)
  
  # make the plot
  heatplot
}



# ============ Plot a heat map for the importance of each elements in the 2D histogram features ==============
plot_importance = function(trained_rds,ml_name,save=FALSE,fname='2dimportance',width=9,height=6.75){
  
  # --------------------------------------------
  # Output: Return 2d heat map figure for model coefficient
  # Input:
  # trained_rds     - RDS file of trained model returned by 'main_ml_2dhist.R'
  # ml_name         - machine learning algorithm name: 'lasso', 'rf'
  # --------------------------------------------
  
  # read in RDS file 
  ls_mod = read_rds(trained_rds)
  
  # extract dimension and coordinates
  xlabel = ls_mod$xcoord
  ylabel = ls_mod$ycoord
  hist_nrow = ls_mod$hist_nrow
  hist_ncol = ls_mod$hist_ncol
  model = ls_mod$train$mod
  
  if (ml_name == 'lasso') {
    
    # ------ LASSO ---------
    # extract original column name
    names_orig = colnames(ls_mod$hist_only)
  
    # extract coefficient and bin names 
    # Extract beta coefficients from a trained (ridge/LASSO) regression model
    tmp_coeffs = coef(model)
    # non-zero coefficient name (including intercept if any)
    names_curr = tmp_coeffs@Dimnames[[1]][tmp_coeffs@i+1]
    # extract coefficient for non-zero terms
    coeff_value = tmp_coeffs@x
    # safety check (redundent)
    if (length(names_curr) != length(coeff_value)) {stop("FATAL: ERROR!")}
    
    # create a data frame storing coefficients
    df_coeff = data.frame(name = names_curr, coeff = coeff_value)
    # exclude intercept
    df_coeff_noint = df_coeff[-1,]
    # make it a row
    df_coeff_noint = spread(df_coeff_noint,name,coeff)
    # Find names of missing columns
    Missing = setdiff(names_orig, names(df_coeff_noint))  
    # Add them, filled with '0's
    df_coeff_noint[Missing] = 0    
    
    # put columns in desired order
    df_coeff_noint = df_coeff_noint[names_orig]   
    
    # transform the data for ggplot
    # DEFAULT is by column conversion from/to matrix/vector
    vec_coef = as.vector(as.matrix(df_coeff_noint))
    df_heatmap = expand.grid(X=xlabel,Y=ylabel)
    df_heatmap = mutate(df_heatmap,Z=vec_coef)
    
    # ggplot setting
    scale_name = 'Coef.'
    low = min(vec_coef)
    maxm = max(vec_coef)
    
    
  } else if (ml_name == 'rf') {
    
    # ------- Random Forest -------
    
    
    
  }
  
  # color transformation (log scale applicable to both positive and negative values)
  asinh_trans = scales::trans_new(name = 'asinh', transform = function(x) asinh(x), 
                                  inverse = function(x) sinh(x))

  
  # make plots
  heatplot = ggplot(df_heatmap)+
    geom_tile(mapping = aes(x=X, y=Y, fill = Z)) +
    scale_fill_gradientn(colours=c("#5566B6","white","#C1211F"),
                         name = scale_name,
                         trans = asinh_trans,
                         #values = c(low,0,maxm),
                         #oob = scales::squish_infinite,
                         #na.value = 'transparent',
                         breaks = c(-5,0,5),
                         labels = c(-5,0,5),
                         limits = c(-5,5)
                         ) +
    # remove border 
    scale_x_continuous(expand=c(0,0))+
    scale_y_continuous(expand=c(0,0))+
    xlab('Energy (kJ/mol)') +
    ylab(expression(bold(paste('Energy gradient (kJ/mol/',ring(A),')')))) + 
    # theme for heatmap, function defined in 'make_plot.R'
    my_theme_heatmap()+ 
    # legend bar control
    guides(fill = guide_colourbar(barwidth = 1.5, barheight = 25))
  
  # save plot, function defined in 'make_plot.R'
  fig_save(heatplot,save,fname,w=width,h=height)
  
  # make the plot
  heatplot
  
  

          
}




# =================== Parity plot of final results for LASSO/RF ==================
plot_parity = function(modrds,datatype='test',n_data=0,ml_name='ML name',
                       label=FALSE,mapcolr=NULL,low=-1,hi=300,intv=50,save = FALSE,width=9,height=8){
  
  # --------------------------------------------
  # Output: Return a plot
  # Input:
  # modrds           - RDS file for saved ML model 
  # datatype         - type of model; 'train' or 'test'
  # n_data           - number of training/testing data
  # ml_name         - name of ML model, used for annotation in plots
  # label           - label points with TOBMOF IDs (for diagnosing only)
  # mapcolr         -map points with colors. Chocies are:
  #                   'vf' (void fraction), 'lcd','pld','sag'(gravimatric surface area),
  #                   'sav' (volumetric surface area)
  # hi              - upper bound for x and y-scale
  # intv            - breaks for axis scale
  # save            - if save the figure
  # width           - figure width. For plot with color bar, set it to 9.8
  # height          - figure height. For plot with color bar, set it to 7.77
  # --------------------------------------------
  
  # read in textural properties
  csv_file = 'C:/Users/khshi/Dropbox/Projects/2D_Energy_Histogram/code/All_data/textural_properties/textprop_tobacco_consistentnew_modify.csv'
  #csv_file = 'C:/Users/khshi/Dropbox/Projects/2D_Energy_Histogram/code/All_data/textural_properties/textprop_apm.csv'
  #csv_file = '/home/kaihang/2dhist/energy-histograms/python_R/big_data/textprop_tobacco.csv'
  # read in textural properties  
  df_texprop = read.csv(csv_file)
  
  # convert id to character
  df_texprop$id = as.character(df_texprop$id)
  
  # read in trained model from RDS file
  ls_mod = read_rds(modrds)
  
  
  if (datatype == 'train') {
    mod = ls_mod$train
    fig_lab = paste0(ml_name,'\n',n_data,' Training data')
    pt_color = "#4E2A84"
    lab_color = "#4E2A84"
    vec_id = as.character(ls_mod$trainid)
      
  } else if (datatype == 'test') {
    mod = ls_mod$test
    fig_lab = paste0(ml_name,'\n',n_data,' Testing data')
    pt_color =  "#0077be"
    lab_color = "#0077be"
    vec_id = as.character(ls_mod$testid)
    
  }
  
  # collect metrics
  r2 = round(mod$eval_metric$R2,2)
  mape = round(mod$eval_metric$mape,1)
  mae = round(mod$eval_metric$mae,1)
  rmse = round(mod$eval_metric$rmse,1)
  
  metric_lab = list(paste0('R^2==',r2),
                    paste0('MAPE==',mape,"*\'%\'"), # % must be escaped
                    paste0('MAE==',mae,"~cm[STP]^{3}/cm^{3}"),
                    paste0('RMSE==',rmse,"~cm[STP]^{3}/cm^{3}"))
  
  # create a data frame for ggplot
  df_data = data.frame(id = vec_id,
                       X = mod$actu_y,
                       Y = mod$pred_y)
  df_join = df_data %>% left_join(df_texprop, by="id")

  # ------------- make the plot ----------
  p = ggplot(df_join,aes_string(x='X',y='Y',label = 'id',color = mapcolr))
  
  # Map COLOR to points 
  if (!is.null(mapcolr)) {
    
    if (is.numeric(df_join[,mapcolr])) {
      
      # continuous color mapping (for coloring vf, surface area etc)
      p = p + geom_point(size = 4, alpha=0.5) +
      scale_color_gradientn(colours= colr_rainbow_full,
                            name = mapcolr,
                            # https://ggplot2.tidyverse.org/reference/guide_colourbar.html
                            guide = guide_colorbar(ticks.colour = "white",
                                                  ticks.linewidth = 2,
                                                  barwidth = 1.5,
                                                  barheight = 28)
                            # control color bar scale
                            #breaks = c(0,10,20,1e-1,1),
                            #labels = c(0,1e-3,1e-2,1e-1,1),
                            #limits = c(0,1)
                            ) 
    } else {
      
      # discrete color mapping 
      p = p + geom_point(size = 4, alpha=0.5) +
        # discrete color mapping (built for coloring amorphous porous materials)
        scale_color_manual(values= c('#878787',"#CC0000","#007c42",'#4E2A84'),  
                           name = mapcolr,
                           # https://ggplot2.tidyverse.org/reference/guide_colourbar.html
                           # guide = guide_colorbar(ticks.colour = "white",
                           #                        ticks.linewidth = 2,
                           #                        barwidth = 1.5, 
                           #                        barheight = 28)
                           ) 
    }
    
    p = p +
      # Adjust color bar 
      theme(#legend.position = c(0.93, 0.55),
            legend.key.size = unit(1, "cm"),
            legend.text = element_text(size=25),
            legend.title= element_text(size=25))
    
    
  } else {
    # no color mapping
    p = p + geom_point(color = pt_color,size = 4,alpha = 0.5) 
  } 
  
  # only label points that have large deviation
  if (label) {
    p = p + geom_text(aes(label=ifelse(abs(X-Y)/X>=0.2,id,'')),hjust=0,vjust=0,color='black')
  }
    
  # axis & ticks, only label major ticks 
  p = p +
    geom_abline(slope = 1, intercept = 0,linetype='dashed',size = 0.5) +
    scale_x_continuous(limits = c(low,hi),
                       breaks = seq(low,hi,intv/2),
                       labels = mark_major(seq(low,hi,intv),1)) + 
    scale_y_continuous(limits = c(low,hi),
                       breaks = seq(low,hi,intv/2),
                       labels = mark_major(seq(low,hi,intv),1)) +
    # label (https://astrostatistics.psu.edu/su07/R/html/grDevices/html/plotmath.html)
    xlab(expression(bold(paste('GCMC capacity (',cm[STP]^{3},'/',cm^{3},')')))) +
    ylab(expression(bold(paste('ML predicted capacity (',cm[STP]^{3},'/',cm^{3},')'))))+
    # annotate plot title
    annotate(geom='text',
             x=low+1, y = hi, hjust = 'inward', vjust = 'inward',
             label = fig_lab, color = lab_color, size = 9)+
    # annotate metrics
    annotate(geom='text',
             x=hi, y =c(0.3*(hi-low)+low,0.2*(hi-low)+low,0.1*(hi-low)+low,low+1),  
             hjust='inward', vjust = 'inward',
             label = metric_lab, parse= T, color = lab_color, size = 8)+
    # modify other elements
    my_theme_general()
  
  
  # save plot
  fig_save(p,save,paste0(sub('\\.rds$','',modrds),'_',datatype),w=width,h=height)
  
  # output the plot
  p
  

}






# =================== feature vs. feature (loading) plot for data mining ==================
plot_datamining = function(df_full,vec_features,xlab=NULL,ylab=NULL,scale_lo=c(0,0),scale_hi,pt_size=2,label=FALSE,mapcolr='loading',save=FALSE,width=9,height=8){
  
  # --------------------------------------------
  # Output: Return a plots
  # Input:
  # df_full  -  data frame containing all information (including features, loading and textural prop.)
  #             this can be generated from 'read_all' function in 'read_files.R'
  # vec_features  - vector of features(or properties) to be plotted
  # xlab      - label for x-axis
  # ylab      - label for y-axis
  # scale_lo  - lower limit of the scales for feature values
  # scale_hi  - upper limit of the scales for feature values
  # pt_size   - point size
  # width     - figure width
  # height    - figure height
  # --------------------------------------------
  
  
  #df_filtered = dplyr::select(df_full,append(c('id','loading'),vec_features))
  # Note: delete/add some of columns in the following command. E.g., delete 'Type' if it is only for MOFs
  df_filtered = dplyr::select(df_full,append(c('id','loading','vf','lcd','pld','sa_tot_m2cm3','Type'),vec_features))
  
  # ------- Make 2D plot ------
  if (length(vec_features) <=2) {
    
    if (length(vec_features) == 1) {
      
      # With only one input
      fig_name = paste0(vec_features[1],'_vs_loading')
      
      p = ggplot(df_filtered,aes_string(x=vec_features[1],y='loading',label = 'id',color = mapcolr))+
        scale_x_continuous(limits = c(scale_lo[1],scale_hi[1])) +
        xlab(vec_features[1]) +
        ylab(expression(bold(paste('GCMC capacity (',cm[STP]^{3},'/',cm^{3},')'))))
    
    } else if (length(vec_features) == 2) {
      
      # with two inputs
      fig_name = paste0(vec_features[1],'_vs_',vec_features[2])
      
      p = ggplot(df_filtered,aes_string(x=vec_features[1],y=vec_features[2],label = 'id',color = mapcolr))+ 
        scale_x_continuous(limits = c(scale_lo[1],scale_hi[1])) + 
        scale_y_continuous(limits = c(scale_lo[2],scale_hi[2])) + 
        xlab(xlab) +
        ylab(ylab)
      
    } 
    
    # Map color to points 
    if (!is.null(mapcolr)) {
      
      # Continuous color mapping
      if (is.numeric(df_filtered[,mapcolr])) {
        p = p + geom_point(size = pt_size) +
          # continuous color
          scale_color_gradientn(colours= colr_rainbow_full,#colr_rainbow_b2r, #colr_spectrum2,
                                name = mapcolr,
                                guide = guide_colorbar(ticks.colour = "white",
                                                        ticks.linewidth = 2,
                                #                        barwidth = 1.5, 
                                #                        barheight = 28
                                )
                                ) +
        
          theme(#legend.position = c(0.9, 0.8),
                legend.key.size = unit(1, "cm"),
                legend.text = element_text(size=25),
                legend.title= element_text(size=25))
      
      # Discrete color mapping
      } else {

        p = p + geom_point(size = pt_size) +
          # discrete color mapping (built for coloring amorphous porous materials)
          scale_color_manual(values= c('#878787',"#CC0000","#007c42",'#4E2A84',"#75AADB"),  
                               name = mapcolr,
                               # https://ggplot2.tidyverse.org/reference/guide_colourbar.html
                               # guide = guide_colorbar(ticks.colour = "white",
                               #                        ticks.linewidth = 2,
                               #                        barwidth = 1.5, 
                               #                        barheight = 28)
          ) +
            
          theme(#legend.position = c(0.9, 0.8),
                legend.key.size = unit(1, "cm"),
                legend.text = element_text(size=25),
                legend.title= element_text(size=25))
      }
    
    # No color mapping    
    } else {
      p = p + geom_point(color = "#000000",size = pt_size,alpha = 0.5) 
    } 
    
    # only label points that have large deviation
    if (label) {
      p = p + geom_text(hjust=0,vjust=0,color='black')
    }
    
    # Apply general theme, function defined in 'make_plot.R'
    p = p + my_theme_general()
    
    # save plot
    fig_save(p,save,fname = fig_name,w = width, h=height)
    
    # output the plot
    p
    
    
  # -------  Make 3D plot  -----------
  } else if (length(vec_features) == 3) {
    fig_name = paste0(vec_features[1],'_vs_',vec_features[2],'_vs_',vec_features[3])
    
    # make a 3D plot
    p = plot_ly(x=df_filtered[,vec_features[1]],
                y=df_filtered[,vec_features[2]],
                z=df_filtered[,vec_features[3]],
                type = 'scatter3d', mode='markers',
                marker = list(color =df_filtered[,mapcolr], 
                              colorscale = list(c(0,1/9,2/9,3/9,4/9,5/9,6/9,7/9,8/9,1),
                                                c("#3B9AB2","#56A6BA",
                                                  "#71B3C2","#9EBE91","#D1C74C",
                                                  "#E8C520","#E4B80E","#E29E00",
                                                  "#EA5C00","#F21A00")),
                              showscale = TRUE,
                              size = 5))
                
    
    p = p %>% add_markers() %>% layout(scene = list(xaxis = list(title = vec_features[1]),
                                  yaxis = list(title = vec_features[2]),
                                  zaxis = list(title = vec_features[3])))
    
    
    p
    
    
  }
}





# =================== feature vs. feature (loading) plot for data mining ==================
plot_umap = function(umap_file,label=NULL,mapcolr='loading',zmscale = c(0,1,0,1),
                     pt_size = 2,alp=1,barwd=1.5,barht=26,save=FALSE,fig_name='umap'){
  
  # --------------------------------------------
  # Output: Return a plots
  # Input:
  # umap_file   -  RDS file containing umap object (R version)
  # label       -  label particular id (set to 'all' to label all points, otherwise label points in input vec)
  # zmscale     -  range of axis for x and y (respectively) for plotting
  # --------------------------------------------
  
  ls_umap = read_rds(umap_file)
  
  # axis scales
  x_lim = range(ls_umap$df_trans[,1])
  y_lim = range(ls_umap$df_trans[,2])
  # determine axis scales for making plots
  x_lo = diff(x_lim)*zmscale[1] + x_lim[1]
  x_hi = diff(x_lim)*zmscale[2] + x_lim[1]
  y_lo = diff(y_lim)*zmscale[3] + y_lim[1]
  y_hi = diff(y_lim)*zmscale[4] + y_lim[1]
  
  df_umap = ls_umap$df_trans %>% cbind(.,ls_umap$df_xy[,c(mapcolr,'id')])
  
  p = ggplot(df_umap,aes_string(x='X1',y='X2',label = 'id',color = mapcolr)) + 
      scale_x_continuous(limits = c(x_lo,x_hi)) + 
      scale_y_continuous(limits = c(y_lo,y_hi)) +
      xlab('UMAP coordinate 1') +
      ylab('UMAP coordinate 2')

  # add points 
  p = p + geom_point(size = pt_size,alpha=alp) 
  
  # color mapping of points
  if (is.numeric(df_umap[,mapcolr])) {
    
    # continuous mapping
    p = p + scale_color_gradientn(colours= colr_rainbow_b2r, #colr_spectrum2,
                                  name = mapcolr
                                  )+
      # legend bar control (this control can be put here or included in scale_color_gradientn func.)
      guides(colour = guide_colourbar(barwidth = barwd, 
                                      barheight = barht,
                                      ticks.colour = "white",
                                      ticks.linewidth = 1))

  } else {
    # discrete mapping
    p = p + scale_color_manual(values= c('#878787',"#CC0000","#007c42","#75AADB",'#4E2A84'), #colr_rainbow_full_discrete,
                               name = mapcolr)

  }
       

  # only label points that have large deviation
  if (!is.null(label)) {
    
    if (label == 'all') {
      # label all points
      p = p + geom_text_repel(max.overlaps = Inf)
      #geom_text(hjust=0,vjust=0,color='black')
      
    } else {
      p = p + geom_text_repel(data = subset(df_umap,id %in% label),
                              size=8,
                              # adjust label position
                              nudge_x = 1,
                              nudge_y = 1
                              )
    }
    
      
  }
  
  # Apply general theme, function defined in 'make_plot.R'
  p = p + my_theme_general() + 
    theme(#axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          #axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank()) +
    theme(#legend.position = c(0.9, 0.55),
          legend.key.size = unit(1, "cm"),
          legend.text = element_text(size=25),
          legend.title= element_text(size=25))
  
  # save plot
  fig_save(p,save,fname = fig_name,w = 8.5, h=6.75)
  
  # output the plot
  p
    
    

}



# =================== make a general histogram plot ==================
plot_bar_charts = function(xlsx_file,task='inter',y='R2',y_scale = c(0,1),bar_width=0.6,
                           flip = FALSE, save=FALSE,fig_name='BarCharts',width=15,height=6){
  
  # --------------------------------------------
  # Output: Return a plots
  # Input:
  # xlsx_file   -  Excel input file
  # task        -  'inter' to compare 1D and 2D energy histogram; 'intra' to compare different ML methods for 2D EH
  # y           -  Metrics: 'R2', 'MAE', 'RMSE'
  # y_scale     -  scale range for y
  # bar_width   -  bar width for plotting
  # flip        -  if to plot a horizontal bar plot
  # save        -  save figure
  # fig_name    -  figure name
  # width, height  - width and height of saved figure
  # --------------------------------------------
  
  df_data = data.frame()
  
  # some plot info
  if (y == 'R2') {
    ylabel = expression(bold(paste(R^{2})))
  } else if (y == 'MAE') {
    ylabel = expression(bold(paste('MAE (',cm[STP]^{3},'/',cm^{3},')')))
  } else if (y == 'RMSE') {
    ylabel = expression(bold(paste('RMSE (',cm[STP]^{3},'/',cm^{3},')')))
  } else if (y == 'MAPE') {
    ylabel = expression(bold(paste('MAPE (%)')))
  }
  
  # compare different types of features (1D histogram versus 2D histogram)
  if (task == 'inter') {
    
    df_EH = read_excel(xlsx_file,sheet='ML_compare_w_Li_etal')
    
    df_data = df_EH[df_EH$Metric == y,] %>% dplyr::select(., c('System',"ML-method",'Features','Value'))
    
    # lock in factor level order so that the plotting will following my data frame order
    #df_data$System = factor(df_data$System, levels = rev(unique(df_data$System)))
    df_data$System = factor(df_data$System, levels = unique(df_data$System))

    # ----------- Make plots --------------
    p = ggplot(df_data,aes_string(x='System',y='Value',pattern = "`ML-method`", fill = 'Features')) 
    p = p + geom_bar_pattern(stat='identity',
                             width=bar_width,
                             color = 'black',
                             pattern_fill = 'black',
                             pattern_angle = 45,
                             pattern_density = 0.05,
                             pattern_spacing = 0.025,
                             pattern_key_scale_factor = 0.6,
                             position=position_dodge(width=0.7)) +
      # ylabel 
      ylab(ylabel)+ 
      # fill color
      scale_fill_manual(values = c("#cae7c8","#4fbbd5")) +
      scale_pattern_manual(values = c('LASSO'='none', 'RF'='stripe')) + 
      guides(pattern = guide_legend(override.aes = list(fill = "white")),
             fill = guide_legend(override.aes = list(pattern = "none"))) +
      # remove margin between plot and axis
      scale_y_continuous(expand=c(0,0))
    
    
    
  } else if (task == 'intra') {
    
    # compare different ML models using the same type of features
    df_EH = read_excel(xlsx_file,sheet='ML_2DEH_This_work')
    
    df_data = df_EH[df_EH$Metric == y,] %>% dplyr::select(., c('System',"ML-method",'Value'))
    
    # lock in factor level order so that the plotting will following my data frame order
    #df_data$System = factor(df_data$System, levels = rev(unique(df_data$System)))
    df_data$System = factor(df_data$System, levels = unique(df_data$System))
    df_data$`ML-method` = factor(df_data$`ML-method`, levels = unique(df_data$`ML-method`))
    
    # ----------- Make plots --------------
    p = ggplot(df_data,aes_string(x='System',y='Value',fill = "`ML-method`")) 
    p = p + geom_bar(stat='identity',
                     width=bar_width,
                     #color = 'black',
                     position=position_dodge(width=0.7)) +
      # ylabel 
      ylab(ylabel)+ 
      # fill color
      scale_fill_manual(values = c("#cae7c8","#4fbbd5","#2472b8","#131d64")) +
      guides(pattern = guide_legend(override.aes = list(fill = "white")),
             fill = guide_legend(override.aes = list(pattern = "none"))) +
      # remove margin between plot and axis
      scale_y_continuous(expand=c(0,0))
  }
  
  

  # Apply general theme, function defined in 'make_plot.R'
  p = p + my_theme_general() +
    theme(#legend.position = c(0.9, 0.55),
    legend.key.size = unit(1, "cm"),
    legend.text = element_text(size=25),
    legend.title= element_text(size=25),
    # specify axis text/title size
    axis.text.y   = element_text(size=25,colour = 'black'),
    axis.text.x   = element_text(size=25,colour = 'black'),
    axis.title.y  = element_text(size=25,colour = 'black',face='bold'),
    axis.title.x  = element_text(size=25,colour = 'black',face ='bold'))

  # Horizontal bar plot
  if (flip) {
    p = p + coord_flip(ylim= c(y_scale[1],y_scale[2])) +
      theme(axis.title.y=element_blank(),
            # add vertical grid line
            panel.grid.major.x = element_line(colour = "black",linetype="dashed",size=0.1),
            #panel.grid.minor.x = element_line(colour = "black",linetype="dashed",size=0.1)
            )
  } else {
    p = p + coord_cartesian(ylim=c(y_scale[1],y_scale[2])) +
      theme(legend.position="top",
            axis.title.x=element_blank(),
            axis.text.x = element_text(angle = 45, hjust=1, size =25),
            panel.grid.major.y = element_line(colour = "black",linetype="dashed",size=0.1),
            panel.grid.minor.y = element_line(colour = "black",linetype="dashed",size=0.1)
            )
  }

  
  # save plot
  fig_save(p,save,fname = fig_name,w = width, h=height)
  
  p

}







# ======== Prepare a PDB file containing grid points of physical significance ===========
find_grid = function(modrds,grid_file,PDB_template,outfile,
                     ener_binwid,grad_binwid,val_range,occ=0,my_resname=NULL,
                     collapse_norm=FALSE,colr_type='ml'){
  
  # ====================================================================================
  # 4/14/2021
  # Output: Output a PDB file containing the grid points (for visualization) falling into the histogram range 
  #         which carries physical significance based on ML model. Temperature factor field in
  #         PDB file contains normalized value of ML feature importance (eg, LASSO coefficient)
  # Input: 
  # modrds        - RDS file of trained ML model
  # grid_file     - original energy/energy gradient file with xyz coordinates
  # PDB_template  - PDB tempelate file (corresponding PDB converted from CIF structure)
  # outfile       - full path&name of the output PDB file
  # ener_binwid   - energy bin width [kJ/mol]
  # grad_binwid   - energy gradient bin width [kJ/mol/A]
  # val_range     - specifying the range of the coefficient/energy/norm and corresponding grid points will be written
  #                 For colr_type = 'ener/norm' mode, four values will be needed, first two for energy, last two for norm 
  # occ           - occupancy label in PDB for visualization purpose
  # my_resname    - specify the residue name manually for visualization purpose
  # collapse_norm - collapse second dimension (norm), this is similar to using 1D energy histogram as features (deprecated,6/11/2021)
  # colr_type     - type of property for color mapping, available options:
  #                 'ml' : coefficient, importance from machine learning
  #                 'energy': grid energy [kJ/mol]
  #                 'norm': norm (absolute gradient) [kJ/mol/A]
  #                 'ener/norm': grid energy and norm 
  # ====================================================================================
  
  # --- Read in energy/energy gradient grid points ---
  # constants 
  k2kjmol = 0.00831446            # convert K to kJ/mol
  
  # read in file
  df_grid = read.table(grid_file)
  
  # columns V4 and V5 store energy and gradient
  # columns V1-V3 store grid Cartesian coordinates
  names(df_grid) = c('x','y','z','energy','norm')
  
  # convert units 
  df_grid$energy = df_grid$energy * k2kjmol
  df_grid$norm = df_grid$norm * k2kjmol
  
  # --- Read in a PDB template
  pdb = read.pdb(PDB_template,CONECT = FALSE)
  
  # --- Main body ----
  # initialize empty vector
  rx = c()
  ry = c()
  rz = c()
  tempfac = c()
  
  # proceed based on the color mapping type
  if (colr_type == 'ml') {
    
    # --- Read in trained ML model ---
    # read in rds object
    ls_mod = read_rds(modrds)
    model = ls_mod$train$mod
    
    # extract coordinates 
    ener_coord = ls_mod$xcoord
    norm_coord = ls_mod$ycoord
    maxecrd = max(ener_coord)
    minecrd = min(ener_coord)
    maxncrd = max(norm_coord)
    minncrd = min(norm_coord)
    
    # create a look-up table 
    df_lookup = expand.grid(X=ener_coord,Y=norm_coord)
    
    # extract ML training coefficient and create a data frame
    beta = data.frame(feature=dimnames(model$beta)[[1]],
                      coef=as.vector(model$beta))
    maxcoef = max(beta$coef)
    mincoef = min(beta$coef)
    
    # select feature based on the value range
    # be noted that between(x,left,right) includes both bounds 
    feature_id = between(beta$coef,val_range[1],val_range[2]) %>%
      which(.)
    
    # loop over all selected histogram pixels/coefficient
    for (coefid in feature_id) {
      
      ener_loc = df_lookup$X[coefid]
      norm_loc = df_lookup$Y[coefid]
      
      # energy boundary
      if (ener_loc == minecrd) {
        ener_lo = -999
        ener_hi = ener_loc + ener_binwid/2
      } else if (ener_loc == maxecrd){
        ener_lo = ener_loc - ener_binwid/2
        ener_hi = Inf
      } else {
        ener_lo = ener_loc - ener_binwid/2
        ener_hi = ener_loc + ener_binwid/2
      }
      
      # norm (gradient) boundary
      if (norm_loc == minncrd) {
        norm_lo = 0
        norm_hi = norm_loc + grad_binwid/2
      } else if (norm_loc == maxncrd){
        norm_lo = norm_loc - grad_binwid/2
        norm_hi = Inf
      } else {
        norm_lo = norm_loc - grad_binwid/2
        norm_hi = norm_loc + grad_binwid/2
      }
      
      # search for grid points in the specified range
      # minus 0.00001 to be consistent with 2D energy histogram construction (so upper bound don't include)
      grid_id_energy = between(df_grid$energy,ener_lo,ener_hi-0.00001) %>% 
        which(.)
      grid_id_norm = between(df_grid$norm,norm_lo,norm_hi-0.00001) %>%
        which(.)
      
      # ignore norm dimension
      if (collapse_norm) {
        grid_id_norm = grid_id_energy
      }
      
      # intersect of two sets 
      grid_id = intersect(grid_id_energy,grid_id_norm)
      
      # calculate normalized coefficient (in range 0,1000)
      if (beta$coef[coefid] >=0) {
        #tempfac_normalized = beta$coef[coefid]/maxcoef * 1000
        tempfac_normalized = beta$coef[coefid]/7500 * 1000
      } else {
        tempfac_normalized = beta$coef[coefid]/mincoef * 1000
      }
      
      # append grid info to vectors
      rx = append(rx, df_grid$x[grid_id])
      ry = append(ry, df_grid$y[grid_id])
      rz = append(rz, df_grid$z[grid_id])
      tempfac = append(tempfac,rep(tempfac_normalized,times=length(grid_id)))
      
    }
    
  } else if (colr_type == 'energy') {
    
    # search for grid points in the specified range
    grid_id_energy = between(df_grid$energy,val_range[1],val_range[2]) %>% 
      which(.)
    
    # truncate positive energies at 2 kJ/mol
    energy_trunc = df_grid$energy[grid_id_energy]
    #energy_trunc[energy_trunc >=2] = 2
    # translate energy by -2 kJ/mol for better visualization (distinguish 0 and positive energy)
    #energy_trunc = energy_trunc -2
    
    energy_trunc[energy_trunc >=0] = - 99
    
    # normalize to 1000, energy_trunc <=0 in all cases
    #tempfac_normalized = energy_trunc/min(energy_trunc) * 1000
    tempfac_normalized = -energy_trunc
    
    # append grid info to vectors
    rx = df_grid$x[grid_id_energy]
    ry = df_grid$y[grid_id_energy]
    rz = df_grid$z[grid_id_energy]
    tempfac = tempfac_normalized
    
    
  } else if (colr_type == 'norm'){
    
    
    # search for grid points in the specified range
    grid_id_norm = between(df_grid$norm,val_range[1],val_range[2]) %>% 
      which(.)
    
    # truncate large norm at 140 kJ/mol/A
    norm_trunc = df_grid$norm[grid_id_norm]
    norm_trunc[norm_trunc >=140] = 140
    
    # normalize to 1000
    tempfac_normalized = norm_trunc/140 * 1000
    
    # append grid info to vectors
    rx = df_grid$x[grid_id_norm]
    ry = df_grid$y[grid_id_norm]
    rz = df_grid$z[grid_id_norm]
    tempfac = tempfac_normalized
  
  } else if (colr_type == 'ener/norm') {
    
    # search for grid points in the specified range
    grid_id_energy = between(df_grid$energy,val_range[1],val_range[2]) %>% 
      which(.)
    
    grid_id_norm = between(df_grid$norm,val_range[3],val_range[4]) %>%
      which(.)
    
    # ignore norm dimension
    if (collapse_norm) {
      stop("collapse_norm cannot be set together with colr_type = 'ener/norm' option")
    }
    
    # intersect of two sets 
    grid_id = intersect(grid_id_energy,grid_id_norm)
    
    # append grid info to vectors
    rx = df_grid$x[grid_id]
    ry = df_grid$y[grid_id]
    rz = df_grid$z[grid_id]
    tempfac = 0 #df_grid$energy[grid_id]
    
    
  }
  
  # specify residue name
  if (!is.null(my_resname)) {
    resname = my_resname
  } else {
    resname = rep('UND', times = length(rx))
  }
  
  
  # Create 'atom' class 
  pdb$atoms = atoms(recname = rep('ATOM',times = length(rx)),
                    eleid = 1:length(rx),
                    elename = rep('H',times = length(rx)),
                    alt="",
                    resname = resname,
                    chainid = "",
                    resid = rep(1,times = length(rx)),
                    insert = "",
                    x1 = rx,
                    x2 = ry,
                    x3 = rz,
                    occ = rep(occ,times = length(rx)),
                    temp = tempfac,
                    segid = resname  )
  
  # for large PDB file (eleid > 99,999)
  pdb$atoms$eleid[pdb$atoms$eleid>99999] = 
    as.character(as.hexmode(pdb$atoms$eleid[pdb$atoms$eleid>99999]))
  
  # write a title
  pdb$title = "GRID POINTS FOR VISUALIZATION"
  
  # write to PDB file
  write.pdb(pdb, file = outfile)
  
}




# ================== make a general x-y line/point plot =======================
plot_basicxy = function(df_full,x_scale,y_scale,lsize=2,psize=3,mapcolr=NULL,save=FALSE,fig_name,width=9,height=6.75){
  
  # --------------------------------------------
  # Output: Return a plots.
  # Note: CODES SHOULD BE MODIFIED IN DIFFERENT CASES
  # 
  # Input:
  # df_full  -  data frame containing all information 
  # x_scale  -  x-range scale
  # y_scale  -  y-range scale
  # lsize    -  line size
  # psize    -  point size
  # mapcolr  -  column for color mapping
  # save     -  flag to save image to file
  # fig_name -  output image file name
  # --------------------------------------------
  
  # --------- Single line -------------
  # p = ggplot(df_full,aes(x=num,y=mean,color = mapcolr))+
  #   scale_x_continuous(limits = c(x_scale[1],x_scale[2],expand=c(0,0)), trans='log10') +
  #   
  #   scale_y_continuous(limits = c(y_scale[1],y_scale[2]),expand=c(0,0)) +
  #   # x and y label
  #   xlab('Number of MOFs') +
  #   ylab('Increase in CPU time %')
  # #ylab(expression(bold(paste('GCMC capacity (',cm[STP]^{3},'/',cm^{3},')'))))
  # 
  # 
  # # map color to points 
  # if (!is.null(mapcolr)) {
  #   p = p + geom_point(size = psize) +
  #     scale_color_gradientn(colours= colr_spectrum2,
  #                           name = mapcolr) + 
  #     
  #     theme(#legend.position = c(0.9, 0.55),
  #       legend.key.size = unit(1, "cm"),
  #       legend.text = element_text(size=20),
  #       legend.title= element_text(size=20))
  # } else {
  #   # no color mapping
  #   p = p + 
  #     geom_line(size = lsize,
  #               linetype = 'solid',
  #               color = "#75AADB") +
  #     geom_point(size = psize,
  #                color = "#75AADB")+
  #     geom_pointrange(aes(ymin=mean-sd, ymax=mean+sd),size=psize,color = "#75AADB")
  # } 
  
  # --------- Multiple line ------------- w= 14, h =6
  p = ggplot(df_full,aes(x=pressure,y=loading,color = mof))+
    scale_x_continuous(limits = c(x_scale[1],x_scale[2],expand=c(0,0))) +
    
    scale_y_continuous(limits = c(y_scale[1],y_scale[2]),expand=c(0,0)) +
    # x and y label
    xlab('Pressure (Pa)') +
    #ylab('Increase in CPU time %')
    ylab(expression(bold(paste('GCMC capacity (',cm[STP]^{3},'/',cm^{3},')'))))
  
  
  # map color to points 
  if (!is.null(mapcolr)) {
    p = p + geom_point(size = psize) +
      scale_color_gradientn(colours= colr_spectrum2,
                            name = mapcolr) + 
      
      theme(#legend.position = c(0.9, 0.55),
        legend.key.size = unit(1, "cm"),
        legend.text = element_text(size=20),
        legend.title= element_text(size=20))
  } else {
    # no color mapping
    p = p + 
      geom_line(aes(group=mof),size = lsize,
                linetype = 'solid' ) +
      geom_point(size = psize)+
      scale_color_manual(values=c("#DE3D25",'#5074C1'))
      #+ geom_pointrange(aes(ymin=loading-sd, ymax=loading+sd),size=psize)
  } 
  
  
  # Apply general theme, function defined in 'make_plot.R'
  p = p + my_theme_general()
  
  # save plot
  fig_save(p,save,fname = fig_name,w = width, h=height)
  
  # output the plot
  p
  
}









