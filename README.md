# 2D Energy Histogram
R implementation of machine learning using 2D energy histogram features for prediction of adsorption.<br/>

![Workflow to construct 2D energy histogram features](https://github.com/snurr-group/2D-energy-histogram/blob/main/feature_engineering_scheme.jpg)


## Files
This repo contains R and python code to reproduce the results in the paper. 

## Usage
### Compilation
Once installed libgmxfort, FORTRAN code for pressure tensor calculation can be compiled using
```bash
gfortran $file -I/usr/local/include `pkg-config --libs libgmxfort`

```
where ```/usr/local/include``` is path to the installed module file. Replace ```$file``` with the corresponding file name for FORTRAN code.

### Calculation parameters
All input parameters for the pressure tensor calculations are set at the beginning of the FORTRAN source code. GROMACS trajectory file (.xtc) and center-of-mass coordinate file for spherical nucleus (CM_True.dat) are needed. "CM_True.dat" file is used to translate the system origin to the center of the nucleus. An example of "CM_true.dat" is provided in the parent directory.


## Reference
[1]  P. Montero de Hijes, **K. Shi**, E. G. Noya, E. E. Santiso, K. E. Gubbins, E. Sanz and C. Vega, \"The Young–Laplace equation for a solid–liquid interface\". *Journal of Chemical Physics*, 153 (2020) 191102. [[link]](https://aip.scitation.org/doi/10.1063/5.0032602)[[PDF]](http://kaihangshi.github.io/assets/docs/paper/Hijes_jcp_2020.pdf)
