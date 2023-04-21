/* ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "pair_tensormd.h"

#include <memory>

#include "atom.h"
#include "cnpy.h"
#include "comm.h"
#include "eigen/Eigen/Dense"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "tensormd.h"

using namespace LAMMPS_NS;

#define MAXLINE 1024
#define KB_TO_EV(x) ((x) / 11604.522060401008)

/* ---------------------------------------------------------------------- */

PairTensorMD::PairTensorMD(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;

  allocated = 0;

  dpot = nullptr;
  fpot = nullptr;
  scale = nullptr;

  cutmax = 0.0;
  etemp = 0.0;
}

/* ----------------------------------------------------------------------
free all arrays
check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairTensorMD::~PairTensorMD()
{
  delete dpot;
  delete fpot;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
  }
}

/* ---------------------------------------------------------------------- */

void PairTensorMD::setup_compute(int &nnl_max, int *typenums)
{
  int ntypes = atom->ntypes;
  int numneigh_max[ntypes];
  double a, b, ratio;
  int elti, i, ii;

  // Set zeros
  for (elti = 0; elti < this->atom->ntypes; elti++) {
    typenums[elti] = 0;
    numneigh_max[elti] = 0;
  }

  // Get `numneigh_max` and `ntypes`
  for (ii = 0; ii < list->inum; ii++) {
    i = list->ilist[ii];
    elti = map[atom->type[i]];
    typenums[elti]++;
    numneigh_max[elti] = MAX(numneigh_max[elti], list->numneigh[i]);
  }

  // Compute `nnl_max`
  a = 0;
  b = 0;
  for (elti = 0; elti < ntypes; elti++) {
    a = MAX(a, static_cast<double>(typenums[elti]));
    b = MAX(b, static_cast<double>(numneigh_max[elti]));
  }
  // TODO: `ratio` is fixed to 1.0. Can we easily predict its value?
  // ratio = std::pow(cutforce / (cutforce + neighbor->skin), 3);
  // ratio = MIN(1.0, 0.5 + ratio * static_cast<double>(a / list->inum));
  ratio = 1.0;
  nnl_max = static_cast<int>(ratio * b);
}

/* ---------------------------------------------------------------------- */

void PairTensorMD::compute(int eflag, int vflag)
{
  int numneigh_max;
  int typenums[atom->ntypes];
  double etotal;

  ev_init(eflag, vflag);

  // typenums[elti] is the number of atoms of type `elti`
  setup_compute(numneigh_max, typenums);

  // Allocate memory for tensors
  if (dpot)
    dpot->setup_local(atom->nlocal, numneigh_max, typenums);
  else
    fpot->setup_local(atom->nlocal, numneigh_max, typenums);

  // Calculate density
  if (dpot)
    dpot->calc_tensor_density(list->ilist, list->inum, atom->type, map, atom->x,
                              numneigh_max, list->numneigh, list->firstneigh);
  else
    fpot->calc_tensor_density(list->ilist, list->inum, atom->type, map, atom->x,
                              numneigh_max, list->numneigh, list->firstneigh);

  // Run
  if (dpot)
    etotal = dpot->run(typenums, etemp, eatom, atom->f, vatom);
  else
    etotal = fpot->run(typenums, etemp, eatom, atom->f, vatom);

  if (eflag) { eng_vdwl = etotal; }

  // The total virial can be computed
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairTensorMD::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(scale, n + 1, n + 1, "pair:scale");

  map = new int[n + 1];
}

/* ----------------------------------------------------------------------
global settings
------------------------------------------------------------------------- */

void PairTensorMD::settings(int narg, char ** /*arg*/)
{
  if (narg != 0) error->all(FLERR, "Illegal pair_style command");
}

/* ----------------------------------------------------------------------
set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairTensorMD::coeff(int narg, char **arg)
{
  int m, n, i, interp;
  double dx;
  bool read_elt = true;

  if (!allocated) allocate();

  if (narg < 4) error->all(FLERR, "Incorrect args for pair coefficients");

  // insure I,J args are * *
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  // check for presence of first meam file

  std::string lib_file = utils::get_potential_file_path(arg[2]);
  if (lib_file.empty())
    error->all(FLERR, fmt::format("Cannot open potential file {}", lib_file));

  // read atom symbols and options
  m = 1;
  i = 3;
  dx = -1.0;
  interp = 0;
  while (i < narg) {
    auto var = std::string(arg[i]);
    if (var == "interp") {
      dx = std::strtod(arg[i + 1], nullptr);
      i += 2;
      if (i < narg) {
        char *ptr;
        auto trial = std::strtol(arg[i], &ptr, 10);
        if (*ptr == '\0') {
          interp = static_cast<int>(trial);
          i++;
        }
      }
      read_elt = false;
    } else if (var == "etemp") {
      etemp = KB_TO_EV(std::strtod(arg[i + 1], nullptr));
      i += 2;
      read_elt = false;
    } else if (read_elt) {
      eltmap[m] = var;
      m++;
      i++;
    }
  }
  read_potential_file(lib_file);
  if (dpot)
    dpot->setup_global(&cutmax);
  else
    fpot->setup_global(&cutmax);

  // setup fnn interpolation
  if (dx > 0) {
    if (dpot) {
      dpot->setup_interpolation(dx, interp);
    } else {
      fpot->setup_interpolation(static_cast<float>(dx), interp);
    }
    std::string algo;
    switch (interp) {
      case 0:
        algo = "default";
        break;
      case 1:
        algo = "batch";
        break;
      case 2:
        algo = "batch (ghost)";
        break;
      case 3:
        algo = "batch (direct)";
        break;
      default:
        error->all(FLERR, "Interpolation algo should be 0, 1, 2 or 3.\n");
        break;
    }
    if (logfile && comm->me == 0) {
      fprintf(logfile, "Use interpolation: dr = %.4f, algo = %s\n", dx,
              algo.c_str());
    }
    if (screen && comm->me == 0) {
      fprintf(screen, "Use interpolation: dr = %.4f, algo = %s\n", dx,
              algo.c_str());
    }
  }

  // clear setflag since coeff() called once with I,J = * *
  n = atom->ntypes;
  for (i = 1; i <= n; i++)
    for (int j = i; j <= n; j++) setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements
  // set mass for i,i in atom class

  int count = 0;
  for (i = 1; i <= n; i++) {
    for (int j = i; j <= n; j++) {
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        if (i == j) atom->set_mass(FLERR, i, mass[map[i]]);
        count++;
      }
      scale[i][j] = 1.0;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");

  for (i = 2; i < narg; i++) { this->kwargs.emplace_back(arg[i]); }
}

/* ----------------------------------------------------------------------
init specific to this pair style
------------------------------------------------------------------------- */

void PairTensorMD::init_style()
{
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style tensoralloy/native requires newton pair on");

  // need a full neighbor list
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairTensorMD::init_one(int i, int j)
{
  if (setflag[i][j] == 0) scale[i][j] = 1.0;
  scale[j][i] = scale[i][j];
  return cutmax;
}

/* ----------------------------------------------------------------------
Setup `map`: map[eltype] = i
 eltmap: specified in `pair_coeff`
 model_eltmap: specified in the potential file.
------------------------------------------------------------------------- */

void PairTensorMD::setup_eltmap(int nelt, const int *numbers)
{
  int i, j;
  char elements[nelt][3];
  std::map<int, std::string> model_eltmap;
  for (i = 0; i < nelt * 2; i += 2) {
    elements[i][0] = static_cast<char>(numbers[i + 0]);
    if (numbers[i + 1] > 0) {
      elements[i][1] = static_cast<char>(numbers[i + 1]);
      elements[i][2] = '\0';
    } else {
      elements[i][1] = '\0';
    }
    model_eltmap[i / 2 + 1] = std::string(elements[i]);
  }
  map[0] = 0;
  for (i = 1; i < nelt + 1; i++) {
    for (j = 1; j < nelt + 1; j++) {
      if (eltmap[i] == model_eltmap[j]) { map[i] = j - 1; }
    }
  }
}

/* ----------------------------------------------------------------------
Read the npz potential file
------------------------------------------------------------------------- */

void PairTensorMD::read_potential_file(const std::string &globalfile)
{
  int elti, nelt;
  std::vector<int> numbers;
  std::vector<double> masses;
  auto npz = cnpy::npz_load(globalfile);
  if (npz.find("precision") != npz.end()) {
    if (npz["precision"].data<int>()[0] == 32) {
      fpot = new TensorMD<float>(memory, error, screen, logfile, comm->me);
      if (logfile && comm->me == 0)
        fprintf(logfile, "float precision: 32 bits\n");
      if (screen && comm->me == 0)
        fprintf(screen, "float precision: 32 bits\n");
    }
  }
  if (!fpot)
    dpot = new TensorMD<double>(memory, error, screen, logfile, comm->me);
  if (fpot)
    fpot->read_npz(npz, nelt, numbers, masses);
  else
    dpot->read_npz(npz, nelt, numbers, masses);
  this->setup_eltmap(nelt, numbers.data());
  for (elti = 0; elti < nelt; elti++) mass.push_back(masses[elti]);
}

/* ----------------------------------------------------------------------
memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double PairTensorMD::memory_usage()
{
  if (dpot)
    return dpot->memory_usage();
  else
    return fpot->memory_usage();
}

/* ---------------------------------------------------------------------- */

void *PairTensorMD::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str, "scale") == 0) return (void *) scale;
  return nullptr;
}
