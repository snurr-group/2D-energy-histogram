#ifndef SUPERCELL_H
#define SUPERCELL_H

#include <openbabel/mol.h>
#include <openbabel/generic.h>

namespace OpenBabel {

	struct int3 {
		int x;
		int y;
		int z;
	};

	class SuperCell
	{
	public:
		// use default constructors and destructors.  Stateless class
		//! Expands molp into its P1 supercell, without performing bond perception
		void makeSuper(OBMol* molp, int a, int b, int c);
		//! Applies symmetry operations to molp, but does not perform bond perception
		void makeP1(OBMol *molp);
		//! Minimum number of unit cells required to be at least twice the LJ cutoff
		int3 minCells(OBMol* molp, double rcut);

	private:
		//! Updates lattice parameters for the larger unit cell
		void cellParametersChanged(OBMol *molp, double a, double b, double c);
		//! Replicates the entire unit cell the number of times specified
		void duplicateUnitCell(OBMol* molp, int a, int b, int c);
		//! Convenience function to get the OBUnitCell* from an OBMol*
		OBUnitCell* getUnitCell(OBMol *molp);
	};

} // end namespace OpenBabel

#endif
