/* Derived from supercellextension.cpp in Avogadro
 * See also https://github.com/cryos/avogadro/blob/master/libavogadro/src/extensions/supercellextension.cpp
 */

#include "supercell.h"

#include <openbabel/mol.h>
#include <openbabel/generic.h>
#include <cmath>

namespace OpenBabel {

	OBUnitCell* SuperCell::getUnitCell(OBMol *molp) {
		OBUnitCell* uc = NULL;
		if (!molp) {
			obErrorLog.ThrowError(__FUNCTION__, "Invalid OBMol pointer", obError);
		} else {
			uc = (OBUnitCell * )molp->GetData(OBGenericDataType::UnitCell);
		}
		if (!uc) {
			obErrorLog.ThrowError(__FUNCTION__, "OBUnitCell is missing.", obWarning);
		}
		return uc;
	}

	void SuperCell::makeSuper(OBMol* molp, int a, int b, int c) {
		makeP1(molp);
		duplicateUnitCell(molp, a, b, c);
	}

	void SuperCell::makeP1(OBMol *molp) {
		OBUnitCell *uc = getUnitCell(molp);
		uc->FillUnitCell(molp);
	}

	int3 SuperCell::minCells(OBMol* molp, double rcut) {
		OBUnitCell* uc = getUnitCell(molp);
		int3 abc;
		abc.x = static_cast<int>(ceil(2.0 * rcut / uc->GetA()));
		abc.y = static_cast<int>(ceil(2.0 * rcut / uc->GetB()));
		abc.z = static_cast<int>(ceil(2.0 * rcut / uc->GetC()));
		return abc;
	}

	void SuperCell::cellParametersChanged(OBMol* molp, double a, double b, double c) {
		OBUnitCell* uc = getUnitCell(molp);
		if (!molp || !uc)
			return;

		std::vector<vector3> cellVectors = uc->GetCellVectors();

		vector3 A = vector3(cellVectors[0].x() * a,
												cellVectors[0].y() * a,
												cellVectors[0].z() * a);

		vector3 B = vector3(cellVectors[1].x() * b,
												cellVectors[1].y() * b,
												cellVectors[1].z() * b);

		vector3 C = vector3(cellVectors[2].x() * c,
												cellVectors[2].y() * c,
												cellVectors[2].z() * c);

		uc->SetData(A, B, C);
		//m_molecule->setOBUnitCell(uc);  // should automatically propogate to OBMol since the pointer is shared

	}

	void SuperCell::duplicateUnitCell(OBMol* molp, int a, int b, int c) {
		OBUnitCell* uc = getUnitCell(molp);
		std::vector<vector3> cellVectors = uc->GetCellVectors();

		std::vector<OBAtom*> orig_atoms;  // Atoms in original unit cell
		FOR_ATOMS_OF_MOL(ap, *molp) {
			orig_atoms.push_back(&*ap);
		}

		molp->BeginModify();
		for (int x = 0; x < a; ++x) {
			for (int y = 0; y < b; ++y) {
				for (int z = 0; z < c; ++z) {
					if (x==0 && y==0 && z==0) {
						continue;  // Do not copy the unit cell onto itself
					}
					// Calculate transform in Cartesian coordinates (cellVec x dCell)
					vector3 disp(
						cellVectors[0].x() * x + cellVectors[1].x() * y + cellVectors[2].x() * z,
						cellVectors[0].y() * x + cellVectors[1].y() * y + cellVectors[2].y() * z,
						cellVectors[0].z() * x + cellVectors[1].z() * y + cellVectors[2].z() * z
						);

					for (std::vector<OBAtom*>::iterator it=orig_atoms.begin(); it!=orig_atoms.end(); ++it) {
						OBAtom* origAtom = *it;
						OBAtom* newAtom = molp->NewAtom();
						*newAtom = *origAtom;
						newAtom->SetVector((origAtom->GetVector()) + disp);
					}

				}
			}
		}
		molp->EndModify();

		// Update the length of the unit cell
		cellParametersChanged(molp, a, b, c);
	}
} // end namespace OpenBabel
