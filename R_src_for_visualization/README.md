## Instruction for visualization of 2D energy histogram and energy/energy gradient grids
### Generation of 2D energy histogram
First, convert raw 3D energy/energy gradient grids that are generated using C++ binary file in `Energy_EnergyGradients_generator` folder to 2D energy histogram using `main_prep_2dhist.R` file. In this file, modify 2D histogram parameters and input/output file name under "Parameters" section. Then, in RStudio, execute the file by typing the following command in the Console panel 
```
> source('main_prep_2dhist.R')
```
The output file is the 2D energy histogram file in RDS format and its name is specified by `tdhistname` in the code. An example output file `tdhist_Kr_example.rds` is included under this directory.

### Visualization of 2D energy histogram
To visualize the 2D energy histogram, in RStudio Console panel, type
```
> source('make_plot.R')
> plot_2dhist('tdhist_Kr_example.rds')
``` 
Replace `tdhist_Kr_example.rds` with whatever output file name you specified in `main_prep_2dhist.R`.

### Visualization of 3D energy/energy gradient grids
To write grids for visualization, we can use `find_grid` function in `make_plot.R` file. For more information, please read the source code. Here we demonstrate how to write raw **energy** grid file into a PDB format and visualize it in [VMD software](https://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=VMD). <br>
First, in RStudio, type 
```
> source('make_plot.R')
> find_grid(modrds = NULL,                                            # set to NULL because we only want to extract energy info
          grid_file = 'example_input/ener_grad_Kr_norm_0.5A.txt',     # raw 3D energy and energy gradient grids file
          PDB_template = 'example_input/tobmof-3559.pdb',             # a PDB template file converted from CIF file of the MOF
          outfile = 'enggrids4visualization.pdb',                     # output file name
          ener_binwid = 2,                                            # energy bin width parameter, same in main_prep_2dhist.R
          grad_binwid = 66,                                           # gradient bin width parameter, same in main_prep_2dhist.R
          val_range = c(-100,0),                                      # range of energy value to be written in PDB in [kJ/mol]
          occ=0,
          my_resname = NULL,
          collapse_norm = FALSE,
          colr_type = 'energy'                                        # Only want to visualize energy information
          )     
```
Then the output PDB file `enggrids4visualization.pdb` can be visualized using VMD. In VMD, choose 'Graphics'->'Representation'->'Coloring Method'->'Beta' to visualize energy values. We converted energy values into (0,100) scale for visualization.