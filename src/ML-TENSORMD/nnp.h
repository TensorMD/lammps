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

#ifndef LAMMPS_TENSORMD_NNP_H
#define LAMMPS_TENSORMD_NNP_H

#include "memory.h"
#include <cmath>
#include <map>
#include <vector>

#if defined(EIGEN_USE_MKL_ALL)
#include "mkl.h"
#elif defined(__APPLE__) && defined(AccelerateFramework)
#include <Accelerate/accelerate.h>
#else
#include "cblas.h"
#endif

#include "cnpy.h"
#include "eigen/unsupported/Eigen/CXX11/Tensor"

using Eigen::Tensor;
using std::map;
using std::vector;

namespace LAMMPS_NS {

class Error;
template <typename Scalar> class Activation;

typedef enum { ReLU, Softplus, Tanh, Squareplus } Activation_fn;

template <typename Scalar>
class NeuralNetworkPotential {
 public:
  NeuralNetworkPotential(Memory *mem, Error *error);
  ~NeuralNetworkPotential();

  void setup_global(cnpy::npz_t &npz);
  void setup_global(int nelt, int num_in, int nlayers, const int *layer_sizes,
                    int actfn, Scalar ***weights, Scalar ***biases,
                    bool use_resnet_dt, bool apply_output_bias);
  Scalar compute(int elti, Scalar *x_in, int n, Scalar *y, Scalar *dydx);
  Scalar forward(int elti, Scalar *x_in, int n, Scalar *y);
  void backward(int elti, Scalar *grad_in, Scalar *grad_out);
  virtual double get_memory_usage();
  double get_flops_per_atom();

 protected:
  Memory *memory;
  Error *error;

  int nelt;
  map<int, int> nmax;
  map<int, int> nlocal;

  // Network structure
  int n_in, n_out;
  Scalar **pool;
  int n_pool;
  map<int, Scalar **> dz;

  int nlayers;
  map<int, int> full_sizes;
  int max_size;

  map<int, map<int, Scalar *>> weights;
  map<int, map<int, Scalar *>> biases;

  bool use_resnet_dt;
  bool apply_output_bias;

  // FLOPs
  double forward_flops_per_one, backward_flops_per_one;

  // Activation
  Activation_fn actfn;
  Activation<Scalar> *act;

  // Internal functions
  bool sum_output;
  Scalar forward_resnet(int elti, Scalar *x_in, int n, Scalar *y);
  Scalar forward_dense(int elti, Scalar *x_in, int n, Scalar *y);
  void backward_resnet(int elti, Scalar *grad_in, Scalar *grad_out);
  void backward_dense(int elti, Scalar *grad_in, Scalar *grad_out);
};

}    // namespace LAMMPS_NS

#endif    //LAMMPS_TENSORALLOY_NNP_H
