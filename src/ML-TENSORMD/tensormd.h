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

#ifndef LAMMPS_TENSORMD_H
#define LAMMPS_TENSORMD_H

#include <cmath>
#include <map>
#include <string>

#include "cnpy.h"
#include "eigen/unsupported/Eigen/CXX11/Tensor"
#include "math_const.h"
#include "math_special.h"
#include "ration.hpp"

#define maxelt 5
#define maxmot 3

namespace LAMMPS_NS {

class Memory;
class Error;
template <typename Scalar> class NeuralNetworkPotential;
template <typename Scalar> class FeaturePotential;
template <typename Scalar> class Cutoff;
template <typename Scalar> class Descriptor;
class TensorMDTimer;

template <typename Scalar> class TensorMD {
 public:
  TensorMD(Memory *mem, Error *error, FILE *scr, FILE *log, int mpiid);
  ~TensorMD();

  void calc_tensor_density(int *ilist, int inum, const int *type,
                           const int *fmap, double **x, int numneigh_max,
                           int *numneigh, int **firstneigh);

  double memory_usage() const;

  void read_npz(cnpy::npz_t &npz, int &nelt, std::vector<int> &numbers,
                std::vector<double> &masses);
  void setup_interpolation(Scalar delta, int algo);
  void setup_global(double *cutmax);
  void setup_local(int nlocal, int numneigh_max, const int *ntypes);
  Scalar run(int *ntypes, double etemp, double *eatom, double **f,
             double **vatom);

 private:
  Memory *memory;
  Error *error;
  NeuralNetworkPotential<Scalar> *nnp;
  FeaturePotential<Scalar> *fnn;
  Descriptor<Scalar> *descriptor;

  FILE *screen;
  FILE *logfile;
  int mpiid;

  // cutforce = force cutoff
  // cutforcesq = force cutoff squared
  Scalar cutforce, cutforcesq;
  Cutoff<Scalar> *cutoff;

  int neltypes;
  int dims;
  int total_dims;
  int nv_dims;

  struct _eltind_t {
    int offset;
    int row;
  };
  struct _eltind_t eltind[maxelt];
  int **eltij_pos;

  // timer
  TensorMDTimer *timer;
  bool disable_timer;

 protected:
  int vmD_nv[maxmot + 1]{};
  int nmax;
  int cmax;
  int alocal;
  int num_filters;
  int max_moment;
  Scalar rmax;
  bool is_T_symmetric;

  // Map `ilocal` to/from row index of `rho`.
  int *i2row;
  int *row2i;

  Eigen::Tensor<int, 4> ijlist;      // 2cba

  // The tensor-based algorithm
  Eigen::Tensor<Scalar, 2> T;        // md
  Eigen::Tensor<Scalar, 4> M;        // dcba
  Eigen::Tensor<Scalar, 4> V;        // dcba
#ifndef USE_FUSED_F2
  Eigen::Tensor<Scalar, 5> dMdrx;    // dxcba
#endif
  Eigen::Tensor<Scalar, 3> R;        // cba
  Eigen::Tensor<int, 3> mask;        // cba
  Eigen::Tensor<Scalar, 4> H;        // kcba
  Eigen::Tensor<Scalar, 4> U;        // kcba
  Eigen::Tensor<Scalar, 4> dHdr;     // kcba
  Eigen::Tensor<Scalar, 4> G;        // mkba
  Eigen::Tensor<Scalar, 4> dEdG;     // mkba
  Eigen::Tensor<Scalar, 4> dEdS;     // dkba
  Eigen::Tensor<Scalar, 4> P;        // dkba
  Eigen::Tensor<Scalar, 4> drdrx;    // xcba
  Eigen::Tensor<Scalar, 4> F1;       // xcba
  Eigen::Tensor<Scalar, 4> F2;       // xcba
  Eigen::Tensor<Scalar, 3> sij;      // cba
  Eigen::Tensor<Scalar, 3> dsij;     // cba

 public:
  void setup_tensors(const int *ilist, int inum, const int *type,
                     const int *fmap, double **x, int numneigh_max,
                     const int *numneigh, int **firstneigh);

  void timer_switch(bool use_timer) {
    disable_timer = !use_timer;
  }

  const Scalar *get_G() { return G.data(); }
  const int *get_i2row() { return i2row; }
  int get_total_dim() { return total_dims; }
};

}    // namespace LAMMPS_NS
#endif

// LAMMPS_TENSORMD_H
