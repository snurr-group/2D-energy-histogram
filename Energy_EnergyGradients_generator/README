# C++ code to generate energy & energy gradient grids (adapted from Ben Bucior's enery grids code)

## Installation and usage
After downloading this folder, the easiest way to set it up is by running `make init`. (after unzipping the "openbabel.zip".) This step may take some time as it compiles and sets up Open Babel in local subdirectories. After it is set up, `make grid` is sufficient for compiling energy grid C++ source code in `src`.

Force field file needs to be set up in `src/forcefield`

You can call the executable directly as `bin/energyhist my_mof.cif` which operates on a `CIF_FILE` (here "my_mof.cif") from the current working directory and also reports timing information. 

Output file contains five columns: X, Y, Z, Energy [K], Energy gradient [K/A]

## Requirements
* cmake
* make
* GCC or another compiler recognized by cmake

## External code
This repository uses code from [Open Babel](https://github.com/openbabel/openbabel), licensed as GPL-2.

If you're curious, the openbabel/ directory was imported from the Open Babel 2.4.1 release using git subtree.  With help from [this useful tutorial](https://medium.com/@porteneuve/mastering-git-subtrees-943d29a798ec), the one-time import process in Git Bash was:

```
git remote add ob-upstream ../Contrib/openbabel/
git subtree add --prefix=openbabel --squash ob-upstream openbabel-2-4-1
```
