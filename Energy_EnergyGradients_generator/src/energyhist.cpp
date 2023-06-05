/* Energy & Energy Gradients calculator */
/* ADD AUTHORS, LICENSING, ETC. */
/* Added force calculation on 10-24-2020 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <queue>
#include <vector>
#include <map>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <openbabel/obconversion.h>
#include <openbabel/mol.h>
#include <openbabel/obiter.h>
#include <openbabel/babelconfig.h>
#include <cmath>

#include "config_ob_data.h"
#include "supercell.h"


using namespace OpenBabel;  // See http://openbabel.org/dev-api/namespaceOpenBabel.shtml

class ForceFieldReader {
protected:
	std::string source;
	std::vector<std::string> atom_types;
	std::map<std::string, double> sigmas;
	std::map<std::string, double> epsilons;

public:
	void clear() {
		source.clear();
		atom_types.clear();
		sigmas.clear();
		epsilons.clear();
	}

	int readFile(std::string mixing_file) {
		// Parse data from RASPA force_field_mixing_rules.def file
		clear();  // overwrite existing data
		std::ifstream infile;
		infile.open(mixing_file.c_str());
		if (!infile.is_open()) {
			obErrorLog.ThrowError(__FUNCTION__, "Unknown mixing rules file " + mixing_file, obError);
			return 0;
		}

		int lineno = 0;
		int decl_definitions = 0;
		int end_lineno = 8;
		std::string line;
		while (getline(infile, line)) {
			++lineno;

			if (lineno <= 5 || lineno == 7) {
				continue;  // skip the first 5 header lines, and line 7
			} else if (lineno == 6) {
				sscanf(line.c_str(), "%d", &decl_definitions);
				end_lineno += decl_definitions;
			} else if (lineno < end_lineno) {
				// For now, let's neglect the case where a line starts with a comment.
				// If we add that later, `peek` will be useful, in addition to incrementing end_lineno
				char site_name[256];
				double sig;
				double eps;
				// Could also consider writing a custom string splitting function
				sscanf(line.c_str(), "%s lennard-jones %lf %lf", site_name, &eps, &sig);
				atom_types.push_back(std::string(site_name));
				sigmas[std::string(site_name)] = sig;
				epsilons[std::string(site_name)] = eps;
			} else if (lineno == end_lineno) {
				// check that it matches the end of file comment
				if (line.substr(0, 21) != "# general mixing rule") {
					// Test that we've reached the end of the data.
					// Something weird was happening with the end of line, so just compare the beginning part
					obErrorLog.ThrowError(__FUNCTION__, "Bad format: too many lines in FF definitions", obWarning);
				}
			}  // ignore everything beyond the mixing rules comment
		}

		if (atom_types.size() != decl_definitions) {
			std::stringstream errMsg;
			errMsg << "Wrong number of imported FF definitions.  Found " << atom_types.size();
			errMsg << ", expected " << decl_definitions;
			errMsg << std::endl;
			obErrorLog.ThrowError(__FUNCTION__, errMsg.str(), obWarning);
		}

		infile.close();
		return atom_types.size();
	}

	std::map<std::string, double> getParam(std::string lj_param) {
		if (lj_param == "sigma") {
			return sigmas;
		} else if (lj_param == "epsilon") {
			return epsilons;
		} else {
			obErrorLog.ThrowError(__FUNCTION__, "Unknown data type " + lj_param, obWarning);
			return std::map<std::string, double>();
		}
	}
};

// Function prototypes
// currently none

int main(int argc, char* argv[])
{
	obErrorLog.SetOutputLevel(obWarning);  // See also http://openbabel.org/wiki/Errors
	char* filename = argv[1];
	// TODO: have "main" call multiple functions, so we can encapsulate this within emscripten or a related environment
	// also clean up our import list and code throughout, so we can publish how quick it is

	// Set up the babel data directory to use a local copy customized for MOFs
	// (instead of system-wide Open Babel data) for this particular program
	std::stringstream dataMsg;
	dataMsg << "Using local Open Babel data saved in " << LOCAL_OB_DATADIR << std::endl;
	obErrorLog.ThrowError(__FUNCTION__, dataMsg.str(), obAuditMsg);
	setenv("BABEL_DATADIR", LOCAL_OB_DATADIR, 1);


	const double grid_spacing = 5.0; // in Angstrom
	double probe_epsilon = 221.0;     // in Kelvin
	double probe_sigma = 4.1;      // in Angstrom
	double rcut = 12.0;              // in Angstrom
	const double rcut2 = rcut * rcut;
	const double del = 0.005;  // value used for approximating Delta function
    char const *output_name = "ener_grad_Xe_norm_1A.txt";  // file name for output energy/gradient grid
  	char const *atom_label_type = "label" ;  // name read from CIF: "symbol" or "label". Choose "symbol" if all, say, carbon atoms have the same force field parameters;
						// if "C1" and "C2", for example, do not have the same force field parameters, choose "label" here, and specify "C1", "C2" etc. force field parameters in FF_FILE.

	std::string FF_FILE = LOCAL_OB_DATADIR;
	FF_FILE += "/../forcefield/APM/force_field_mixing_rules.def";

	ForceFieldReader read_ff;
	read_ff.readFile(FF_FILE);
	std::map<std::string, double> sigmas = read_ff.getParam("sigma");
	std::map<std::string, double> epsilons = read_ff.getParam("epsilon");;

	// Read CIF without bonding
	OBMol orig_mol;
	OBConversion obconversion;
	obconversion.SetInFormat("mmcif");
	obconversion.AddOption("b", OBConversion::INOPTIONS);  // disable bonding entirely
	if (!obconversion.ReadFile(&orig_mol, filename)) {
		printf("Error reading file: %s", filename);
		exit(1);
	}
	std::cout << "Analyzing file: " << filename << std::endl;

	OBUnitCell* pCell = (OBUnitCell * )orig_mol.GetData(OBGenericDataType::UnitCell);
	// Perform symmetry operations on the MOF (make it P1) and expand the supercell
	SuperCell sc;
	int3 min_cells = sc.minCells(&orig_mol, rcut);
	sc.makeSuper(&orig_mol, min_cells.x, min_cells.y, min_cells.z);

	std::cout << "Number of unit cells: "
		<< min_cells.x << " x "
		<< min_cells.y << " x "
		<< min_cells.z << std::endl;

	// Currently, we're calculating a slanted grid (abc, not cart. xyz) within the original unit cell.
	// If we want the full supercell, remove the min_cells.x, etc., in nx block.
	// grid_dx, etc., is based on the supercell, because our atom fractional coordinates are in those terms.
	double grid_dx = grid_spacing / pCell->GetA();
	double grid_dy = grid_spacing / pCell->GetB();
	double grid_dz = grid_spacing / pCell->GetC();
	// 1.0 / min_cells.x is the fractional coordinate size of the original unit cell
	int nx = static_cast<int>(1.0/grid_dx/min_cells.x) + 1;  // number of grid points that fit in the UC fractional coords, plus the origin
	int ny = static_cast<int>(1.0/grid_dy/min_cells.y) + 1;
	int nz = static_cast<int>(1.0/grid_dz/min_cells.z) + 1;
	int ngrid = nx * ny * nz;

	std::vector<vector3> grid_points(ngrid);
	std::vector<double> energies(ngrid, 0.0);   // initialize to 0.0
	// force component
	std::vector<double> frcx(ngrid, 0.0);  // force component in x-direction
	std::vector<double> frcy(ngrid, 0.0);
	std::vector<double> frcz(ngrid, 0.0);
	std::vector<double> normf(ngrid, 0.0);   // norm of \nabla U_i

	FOR_ATOMS_OF_MOL(a, orig_mol) {
		// Atom's fractional coordinate
		vector3 atom_loc = pCell->CartesianToFractional(a->GetVector());

		std::string atom_type;

		if (atom_label_type == "symbol")
		{
			atom_type = etab.GetSymbol(a->GetAtomicNum());
		} else if (atom_label_type == "label")
		{
			atom_type = (a -> GetData("_atom_site_label")) -> GetValue(); 
		} else
		{
			std::cout << "ERROR: Invalid atom_label_type" << std::endl;
		}

		// print atom name for safety checking (can be commented out)
		//std::cout << atom_type  << std::endl;
	
		if (sigmas.find(atom_type) == sigmas.end() || epsilons.find(atom_type) == epsilons.end()) {
			obErrorLog.ThrowError(__FUNCTION__, "Force field parameters not defined for atom type " + atom_type, obWarning);
		}
		double sig = (probe_sigma + sigmas[atom_type]) / 2.0;
		double eps = sqrt(probe_epsilon * epsilons[atom_type]);
		// LJ potential at cutoff for impulsive contribution
		double uc = 4.0*eps * (( pow(sig, 12)/pow(rcut2, 6) ) - ( pow(sig, 6)/pow(rcut2, 3) ));

		int curr_pt = 0;
		double z = 0.0;
		for (int k = 0; k < nz; ++k) {
			double y = 0.0;
			for (int j = 0; j < ny; ++j) {
				double x = 0.0;
				for (int i = 0; i < nx; ++i) {

					double dx = atom_loc.x() - x;
					double dy = atom_loc.y() - y;
					double dz = atom_loc.z() - z;

					// Apply periodic boundary conditions using minimum image convention
					dx -= round(dx);
					dy -= round(dy);
					dz -= round(dz);

					double rsq = pCell->FractionalToCartesian(vector3(dx, dy, dz)).length_2();
					

					// Convert to Cartesian
					vector3 dr = pCell->FractionalToCartesian(vector3(dx,dy,dz));
					
					// Threshold value was changed from 1e-4 to 1e-2. 
					if (rsq < 1e-2) {
						rsq = 1e-2;  // If grid point lies directly on atom, shift it slightly.  E will still be highly repulsive
					}

					double rdist = sqrt(rsq);

					if ((rdist <= (rcut + del)) && (rdist > rcut)) {
						// impulsive force only
						//double ff = - 24*eps/rdist *(pow(sig,6)/pow(rdist,6) - 2*pow(sig,12)/pow(rdist,12));
						frcx[curr_pt] += dr.x()/rdist * uc/(2.0*del);
						frcy[curr_pt] += dr.y()/rdist * uc/(2.0*del);
						frcz[curr_pt] += dr.z()/rdist * uc/(2.0*del);

					} else if ( (rdist <= rcut) && (rdist >= (rcut - del))  ) {
						// impulsive + standard foce
						double frc = - 24.0*eps/rdist *(pow(sig,6)/pow(rdist,6) - 2.0*pow(sig,12)/pow(rdist,12));
						frcx[curr_pt] += dr.x()/rdist * (frc + uc/(2.0*del));
						frcy[curr_pt] += dr.y()/rdist * (frc + uc/(2.0*del));
						frcz[curr_pt] += dr.z()/rdist * (frc + uc/(2.0*del));
						// energy
						energies[curr_pt] += (4*eps) * ((pow(sig, 12) / pow(rsq, 6)) - (pow(sig, 6) / pow(rsq, 3)));

					} else if ( rdist < (rcut-del)) {
						// standard force 
						double frc = - 24*eps/rdist *(pow(sig,6)/pow(rdist,6) - 2*pow(sig,12)/pow(rdist,12));
						frcx[curr_pt] += dr.x()/rdist * frc;
						frcy[curr_pt] += dr.y()/rdist * frc;
						frcz[curr_pt] += dr.z()/rdist * frc;
						// energy
						energies[curr_pt] += (4*eps) * ((pow(sig, 12) / pow(rsq, 6)) - (pow(sig, 6) / pow(rsq, 3)));
					}
					

					grid_points[curr_pt] = vector3(x, y, z);  // redundant after the first atom assignment
					curr_pt++;
					x += grid_dx;
				}
				y += grid_dy;
			}
			z += grid_dz;
		}
	}



	// Write out energies and forces
	FILE* eFile = fopen(output_name, "w");
	//FILE* esFile = fopen("sm_Energy_Values.txt", "w");
	for (int i = 0; i < ngrid; ++i) {

		vector3 cart_pt = pCell->FractionalToCartesian(grid_points[i]);

		// calculate norm of potential gradient on each grid
		normf[i] = sqrt( pow(frcx[i],2) + pow(frcy[i],2) + pow(frcz[i],2) );

		fprintf(eFile, "%f\t%f\t%f\t%f\t%f\n", cart_pt.x(), cart_pt.y(), cart_pt.z(), energies[i], normf[i]);
		//fprintf(esFile, "%.6g\n", energies[i]);
	}
	fclose(eFile);
	//fclose(esFile);
	//
	// Write out coordinates
	/*FILE* coordFile = fopen("atomcoords.txt", "w");
	FOR_ATOMS_OF_MOL(ac, orig_mol) {
		vector3 avec = ac->GetVector();
		fprintf(coordFile, "%f %f %f\n", avec.x(), avec.y(), avec.z());
	}
	fclose(coordFile);
	*/
	return(0);
}
