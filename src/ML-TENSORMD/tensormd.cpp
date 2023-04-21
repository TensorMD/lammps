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

#include "tensormd.h"

#include "cutoff.h"
#include "descriptor.h"
#include "eigen/unsupported/Eigen/CXX11/Tensor"
#include "error.h"
#include "fnn.h"
#include "memory.h"
#include "nnp.h"
#include "tensor_debug.h"
#include "tensor_ops.h"
#include "tensormd_timer.h"

#if defined(USE_LIBXSMM)
#include "libxsmm.h"
#endif

using namespace LAMMPS_NS;
using Eigen::IndexPair;
using Eigen::Tensor;

/* ---------------------------------------------------------------------- */

template <typename Scalar>
TensorMD<Scalar>::TensorMD(Memory *mem, Error *err, FILE *scr,
                                             FILE *log, int mpid)
{
  memory = mem;
  error = err;
  screen = scr;
  logfile = log;
  mpiid = mpid;

  alocal = 0;
  nmax = 0;
  cmax = 0;
  i2row = nullptr;
  row2i = nullptr;

  neltypes = 0;
  rmax = 0.0;
  num_filters = 0;
  max_moment = 0;

  dims = 0;
  total_dims = 0;
  nv_dims = 0;

  is_T_symmetric = false;

  cutforce = 0;
  cutforcesq = 0;

  eltij_pos = nullptr;

  nnp = nullptr;
  fnn = nullptr;
  descriptor = nullptr;
  cutoff = nullptr;

  disable_timer = false;

  // Initialize the FLOPs timer
  timer = new TensorMDTimer(this->screen, this->logfile);
}

template <typename Scalar> TensorMD<Scalar>::~TensorMD()
{
  if (this->timer && !disable_timer) this->timer->print();

  memory->destroy(this->eltij_pos);

  delete this->nnp;
  delete this->fnn;
  delete this->timer;
  delete this->cutoff;
  delete this->descriptor;

#ifdef USE_LIBXSMM
  libxsmm_finalize();
#endif
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::read_npz(cnpy::npz_t &npz, int &nelt,
                                         std::vector<int> &numbers,
                                         std::vector<double> &masses)
{
  int i, j, k;
  int use_fnn;

  // Global parameters
  rmax = *npz["rmax"].data<Scalar>();
  neltypes = *npz["nelt"].data<int>();
  max_moment = *npz["max_moment"].data<int>();
  auto masses_array = npz["masses"];
  for (i = 0; i < masses_array.num_vals; i++) {
    masses.push_back(static_cast<double>(masses_array.data<Scalar>()[i]));
  }
  auto numbers_array = npz["numbers"];
  for (i = 0; i < numbers_array.num_vals; i++) {
    numbers.push_back(numbers_array.data<int>()[i]);
  }
  nelt = neltypes;

  // Setup the atomistic neural network potential
  nnp = new NeuralNetworkPotential<Scalar>(memory, error);
  nnp->setup_global(npz);
  timer->setup_nn(nnp->get_flops_per_atom());

  // Setup the descriptor
  use_fnn = *npz["use_fnn"].data<int>();
  if (use_fnn) {
    this->fnn = new FeaturePotential<Scalar>(this->memory, this->error);
    this->fnn->setup_global(npz, num_filters);
    double forward, backward;
    this->fnn->get_flops_per_atom(forward, backward);
    this->timer->setup_fnn(forward, backward);
  } else {
    this->descriptor = new Descriptor<Scalar>(this->memory, this->error, npz);
    this->num_filters = this->descriptor->get_num_filters();
  }

  // is T_md symmetric?
  if (npz.find("is_T_symmetric") != npz.end() &&
      npz["is_T_symmetric"].data<int>()[0] == 1) {
    is_T_symmetric = true;
  }

  if (*npz["fctype"].data<int>() == 0) {
    cutoff = new Cutoff<Scalar>(rmax);    // Cosine
  } else {
    cutoff = new Cutoff<Scalar>(rmax, 5.0);    // Polynomial
  }

  // Misc
  memory->create(eltij_pos, nelt, nelt, "pair:eltij_pos");
  for (i = 0; i < nelt; i++) {
    k = 1;
    for (j = 0; j < nelt; j++) {
      if (i == j) {
        eltij_pos[i][j] = 0;
      } else {
        eltij_pos[i][j] = k;
        k++;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::setup_global(double *cutmax)
{
  int m;

  // Force cutoff
  this->cutforce = this->rmax;
  this->cutforcesq = this->cutforce * this->cutforce;

  // Pass cutoff back to calling program
  *cutmax = static_cast<double>(this->cutforce);

  this->vmD_nv[0] = 1;
  this->vmD_nv[1] = 3;
  this->vmD_nv[2] = 6;
  this->vmD_nv[3] = 10;

  this->dims = this->num_filters * (this->max_moment + 1);
  this->total_dims = this->dims * this->neltypes;
  this->nv_dims = 0;
  for (m = 0; m < this->max_moment + 1; m++) {
    this->nv_dims += this->vmD_nv[m];
  }

  this->timer->setup(neltypes, nv_dims, max_moment + 1, num_filters,
                     cutoff->is_cosine());

  // Initialize the multiplicity tensor T_md
  Eigen::array<Eigen::Index, 2> shape{max_moment + 1, nv_dims};
  T.resize(shape);
  T.setZero();
  T(0, 0) = 1;
  if (max_moment > 0) {
    T(1, 1) = 1;    // x
    T(1, 2) = 1;    // y
    T(1, 3) = 1;    // z
    if (max_moment > 1) {
      T(2, 4) = 1;    // xx
      T(2, 5) = 2;    // xy
      T(2, 6) = 2;    // xz
      T(2, 7) = 1;    // yy
      T(2, 8) = 2;    // yz
      T(2, 9) = 1;    // zz
      if (is_T_symmetric) { T(2, 0) = -1.0 / 3.0; }
      if (max_moment > 2) {
        T(3, 10) = 1;    // xxx
        T(3, 11) = 3;    // xxy
        T(3, 12) = 3;    // xxz
        T(3, 13) = 3;    // xyy
        T(3, 14) = 6;    // xyz
        T(3, 15) = 3;    // xzz
        T(3, 16) = 1;    // yyy
        T(3, 17) = 3;    // yyz
        T(3, 18) = 3;    // yzz
        T(3, 19) = 1;    // zzz
        if (is_T_symmetric) {
          T(3, 1) = -3.0 / 5.0;
          T(3, 2) = -3.0 / 5.0;
          T(3, 3) = -3.0 / 5.0;
        }
      }
    }
  }

#ifdef USE_LIBXSMM
  libxsmm_init();
  libxsmm_mmfunction<Scalar, Scalar>(LIBXSMM_GEMM_FLAGS('N', 'N'), 3, 1,
                                     nv_dims, 3, nv_dims, 3, 1.0, 0.0);
#endif
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::setup_local(int nlocal, int numneigh_max,
                                            const int *typenums)
{
  int i, elti;
  int a, b, c, d, k, m, x;

  // update the timer
  timer->tic();
  timer->update(nlocal, numneigh_max);

  // grow local arrays if necessary
  if (nlocal > nmax || numneigh_max > cmax) {
    if (neltypes > 1) {
      memory->destroy(i2row);
      memory->destroy(row2i);
    }
    nmax = nlocal;
    cmax = numneigh_max;
    if (neltypes > 1) {
      memory->create(row2i, nmax, "tensoralloy:row2i");
      memory->create(i2row, nmax, "tensoralloy:i2row");
    }

    a = nmax;
    b = neltypes;
    c = numneigh_max;
    d = nv_dims;
    k = num_filters;
    m = max_moment + 1;
    x = 3;

    // The tensor-based algorithm
    Eigen::array<Eigen::Index, 4> M_shape{d, c, b, a};
    M.resize(M_shape);
    V.resize(M_shape);
#ifndef USE_FUSED_F2
    Eigen::array<Eigen::Index, 5> dMdrx_shape{d, x, c, b, a};
    dMdrx.resize(dMdrx_shape);
#endif
    Eigen::array<Eigen::Index, 3> R_shape{c, b, a};
    R.resize(R_shape);
    mask.resize(R_shape);
    if ((fnn && !fnn->is_interpolatable()) ||
        (descriptor && !descriptor->is_interpolatable())) {
      sij.resize(R_shape);
      dsij.resize(R_shape);
    }
    Eigen::array<Eigen::Index, 4> P_shape{d, k, b, a};
    P.resize(P_shape);
    dEdS.resize(P_shape);
    Eigen::array<Eigen::Index, 4> G_shape{m, k, b, a};
    G.resize(G_shape);
    dEdG.resize(G_shape);
    Eigen::array<Eigen::Index, 4> H_shape{k, c, b, a};
    H.resize(H_shape);
    U.resize(H_shape);
    Eigen::array<Eigen::Index, 4> dHdr_shape{k, c, b, a};
    dHdr.resize(dHdr_shape);
    Eigen::array<Eigen::Index, 4> ijlist_shape{2, c, b, a};
    ijlist.resize(ijlist_shape);
    Eigen::array<Eigen::Index, 4> drdrx_shape{x, c, b, a};
    drdrx.resize(drdrx_shape);
    F2.resize(drdrx_shape);
    F1.resize(drdrx_shape);
  }

  // Setup imap when `neltypes > 1` because `imap[i] = i` is always
  // true if `neltypes == 1`.

  eltind[0].offset = 0;
  eltind[0].row = 0;
  if (neltypes > 1) {
    for (i = 0; i < nmax; i++) { i2row[i] = row2i[i] = 0; }
    m = typenums[0];
    for (elti = 1; elti < neltypes; elti++) {
      eltind[elti].offset = m;
      eltind[elti].row = m;
      m += typenums[elti];
    }
  }
  alocal = nlocal;

  timer->record(TIMER::ALLOC);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::setup_tensors(const int *ilist, const int inum,
                                              const int *type, const int *fmap,
                                              double **pos, int numneigh_max,
                                              const int *numneigh,
                                              int **firstneigh)
{
  int i, j, a, b, c, d;
  int ii, jn, elti, eltj;
  int neigh[neltypes];
  double xtmp, ytmp, ztmp;
  double rijinv, rij, rij2, dij[3];
  double x, y, z, xx, xy, xz, yy, yz, zz, xxx, xxy, xxz, xyy, xyz, xzz;
  double yyy, yyz, yzz, zzz, xxxx, xxxy, xxxz, xxyy, xxyz, xxzz, xyyy, xyyz;
  double xyzz, xzzz, yyyy, yyyz, yyzz, yzzz, zzzz;
  double xxxxx, xxxxy, xxxxz, xxxyy, xxxyz, xxxzz, xxyyy, xxyyz, xxyzz, xxzzz;
  double xyyyy, xyyyz, xyyzz, xyzzz, xzzzz, yyyyy, yyyyz, yyyzz, yyzzz, yzzzz;
  double zzzzz;
  double xxxxxx, xxxxxy, xxxxxz, xxxxyy, xxxxyz, xxxxzz, xxxyyy, xxxyyz, xxxyzz;
  double xxxzzz, xxyyyy, xxyyyz, xxyyzz, xxyzzz, xxzzzz, xyyyyy, xyyyyz, xyyyzz;
  double xyyzzz, xyzzzz, xzzzzz, yyyyyy, yyyyyz, yyyyzz, yyyzzz, yyzzzz, yzzzzz;
  double zzzzzz;

  timer->tic();

  for (ii = 0; ii < inum; ii++) {
    for (elti = 0; elti < neltypes; elti++) neigh[elti] = 0;
    i = ilist[ii];
    elti = fmap[type[i]];
    xtmp = pos[i][0];
    ytmp = pos[i][1];
    ztmp = pos[i][2];
    // Find the row of the density matrix `rho` for atom `i`.
    // If `neltypes` is 1, `row == i` is always true.
    if (neltypes == 1) {
      a = i;
    } else {
      a = eltind[elti].row;
      eltind[elti].row++;
    }
    if (neltypes > 1) {
      i2row[i] = a;
      row2i[a] = i;
    }
    for (jn = 0; jn < numneigh[i]; jn++) {
      j = firstneigh[i][jn];
      j &= NEIGHMASK;
      dij[0] = pos[j][0] - xtmp;
      dij[1] = pos[j][1] - ytmp;
      dij[2] = pos[j][2] - ztmp;
      rij2 = dij[0] * dij[0] + dij[1] * dij[1] + dij[2] * dij[2];
      if (rij2 < this->cutforcesq) {
        eltj = fmap[type[j]];
        rij = sqrt(rij2);
        rijinv = 1.0 / rij;
        b = eltij_pos[elti][eltj];
        c = neigh[b];
        if (c < numneigh_max) {
          x = dij[0] * rijinv;
          y = dij[1] * rijinv;
          z = dij[2] * rijinv;
          R(c, b, a) = rij;
          mask(c, b, a) = 1;
          M(0, c, b, a) = 1.0;
#ifndef USE_FUSED_F2
          dMdrx(0, 0, c, b, a) = 0.0;
          dMdrx(0, 1, c, b, a) = 0.0;
          dMdrx(0, 2, c, b, a) = 0.0;
#endif
          drdrx(0, c, b, a) = x;
          drdrx(1, c, b, a) = y;
          drdrx(2, c, b, a) = z;
          ijlist(0, c, b, a) = i;
          ijlist(1, c, b, a) = j;
          if (max_moment > 0) {
            xx = x * x;
            xy = x * y;
            xz = x * z;
            yy = y * y;
            yz = z * y;
            zz = z * z;
            M(1, c, b, a) = x;
            M(2, c, b, a) = y;
            M(3, c, b, a) = z;
#ifndef USE_FUSED_F2
            dMdrx(1, 0, c, b, a) = -rijinv * (xx - 1);
            dMdrx(1, 1, c, b, a) = -rijinv * xy;
            dMdrx(1, 2, c, b, a) = -rijinv * xz;
            dMdrx(2, 0, c, b, a) = -rijinv * xy;
            dMdrx(2, 1, c, b, a) = -rijinv * (yy - 1);
            dMdrx(2, 2, c, b, a) = -rijinv * yz;
            dMdrx(3, 0, c, b, a) = -rijinv * xz;
            dMdrx(3, 1, c, b, a) = -rijinv * yz;
            dMdrx(3, 2, c, b, a) = -rijinv * (zz - 1);
#endif
            if (max_moment > 1) {
              xxx = xx * x;
              xxy = xx * y;
              xxz = xx * z;
              xyy = xy * y;
              xyz = xy * z;
              xzz = xz * z;
              yyy = yy * y;
              yyz = yy * z;
              yzz = yz * z;
              zzz = zz * z;
              M(4, c, b, a) = xx;
              M(5, c, b, a) = xy;
              M(6, c, b, a) = xz;
              M(7, c, b, a) = yy;
              M(8, c, b, a) = yz;
              M(9, c, b, a) = zz;
#ifndef USE_FUSED_F2
              dMdrx(4, 0, c, b, a) = -rijinv * (2 * xxx - 2 * x);
              dMdrx(4, 1, c, b, a) = -rijinv * 2 * xxy;
              dMdrx(4, 2, c, b, a) = -rijinv * 2 * xxz;
              dMdrx(5, 0, c, b, a) = -rijinv * (2 * xxy - y);
              dMdrx(5, 1, c, b, a) = -rijinv * (2 * xyy - x);
              dMdrx(5, 2, c, b, a) = -rijinv * 2 * xyz;
              dMdrx(6, 0, c, b, a) = -rijinv * (2 * xxz - z);
              dMdrx(6, 1, c, b, a) = -rijinv * 2 * xyz;
              dMdrx(6, 2, c, b, a) = -rijinv * (2 * xzz - x);
              dMdrx(7, 0, c, b, a) = -rijinv * 2 * xyy;
              dMdrx(7, 1, c, b, a) = -rijinv * (2 * yyy - 2 * y);
              dMdrx(7, 2, c, b, a) = -rijinv * 2 * yyz;
              dMdrx(8, 0, c, b, a) = -rijinv * 2 * xyz;
              dMdrx(8, 1, c, b, a) = -rijinv * (2 * yyz - z);
              dMdrx(8, 2, c, b, a) = -rijinv * (2 * yzz - y);
              dMdrx(9, 0, c, b, a) = -rijinv * 2 * xzz;
              dMdrx(9, 1, c, b, a) = -rijinv * 2 * yzz;
              dMdrx(9, 2, c, b, a) = -rijinv * (2 * zzz - 2 * z);
#endif
              if (max_moment > 2) {
                xxxx = xxx * x;
                xxxy = xxx * y;
                xxxz = xxx * z;
                xxyy = xxy * y;
                xxyz = xxy * z;
                xxzz = xxz * z;
                xyyy = xyy * y;
                xyyz = xyy * z;
                xyzz = xyz * z;
                xzzz = xzz * z;
                yyyy = yyy * y;
                yyyz = yyy * z;
                yyzz = yyz * z;
                yzzz = yzz * z;
                zzzz = zzz * z;
                M(10, c, b, a) = xxx;
                M(11, c, b, a) = xxy;
                M(12, c, b, a) = xxz;
                M(13, c, b, a) = xyy;
                M(14, c, b, a) = xyz;
                M(15, c, b, a) = xzz;
                M(16, c, b, a) = yyy;
                M(17, c, b, a) = yyz;
                M(18, c, b, a) = yzz;
                M(19, c, b, a) = zzz;
#ifndef USE_FUSED_F2
                dMdrx(10, 0, c, b, a) = -rijinv * (3 * xxxx - 3 * xx);
                dMdrx(10, 1, c, b, a) = -rijinv * 3 * xxxy;
                dMdrx(10, 2, c, b, a) = -rijinv * 3 * xxxz;
                dMdrx(11, 0, c, b, a) = -rijinv * (3 * xxxy - 2 * xy);
                dMdrx(11, 1, c, b, a) = -rijinv * (3 * xxyy - xx);
                dMdrx(11, 2, c, b, a) = -rijinv * 3 * xxyz;
                dMdrx(12, 0, c, b, a) = -rijinv * (3 * xxxz - 2 * xz);
                dMdrx(12, 1, c, b, a) = -rijinv * 3 * xxyz;
                dMdrx(12, 2, c, b, a) = -rijinv * (3 * xxzz - xx);
                dMdrx(13, 0, c, b, a) = -rijinv * (3 * xxyy - yy);
                dMdrx(13, 1, c, b, a) = -rijinv * (3 * xyyy - 2 * xy);
                dMdrx(13, 2, c, b, a) = -rijinv * 3 * xyyz;
                dMdrx(14, 0, c, b, a) = -rijinv * (3 * xxyz - yz);
                dMdrx(14, 1, c, b, a) = -rijinv * (3 * xyyz - xz);
                dMdrx(14, 2, c, b, a) = -rijinv * (3 * xyzz - xy);
                dMdrx(15, 0, c, b, a) = -rijinv * (3 * xxzz - zz);
                dMdrx(15, 1, c, b, a) = -rijinv * 3 * xyzz;
                dMdrx(15, 2, c, b, a) = -rijinv * (3 * xzzz - 2 * xz);
                dMdrx(16, 0, c, b, a) = -rijinv * 3 * xyyy;
                dMdrx(16, 1, c, b, a) = -rijinv * (3 * yyyy - 3 * yy);
                dMdrx(16, 2, c, b, a) = -rijinv * 3 * yyyz;
                dMdrx(17, 0, c, b, a) = -rijinv * 3 * xyyz;
                dMdrx(17, 1, c, b, a) = -rijinv * (3 * yyyz - 2 * yz);
                dMdrx(17, 2, c, b, a) = -rijinv * (3 * yyzz - yy);
                dMdrx(18, 0, c, b, a) = -rijinv * 3 * xyzz;
                dMdrx(18, 1, c, b, a) = -rijinv * (3 * yyzz - zz);
                dMdrx(18, 2, c, b, a) = -rijinv * (3 * yzzz - 2 * yz);
                dMdrx(19, 0, c, b, a) = -rijinv * 3 * xzzz;
                dMdrx(19, 1, c, b, a) = -rijinv * 3 * yzzz;
                dMdrx(19, 2, c, b, a) = -rijinv * (3 * zzzz - 3 * zz);
#endif
                if (max_moment > 3) {
                  xxxxx = xxxx * x;
                  xxxxy = xxxx * y;
                  xxxxz = xxxx * z;
                  xxxyy = xxxy * y;
                  xxxyz = xxxy * z;
                  xxxzz = xxxz * z;
                  xxyyy = xxyy * y;
                  xxyyz = xxyy * z;
                  xxyzz = xxyz * z;
                  xxzzz = xxzz * z;
                  xyyyy = xyyy * y;
                  xyyyz = xyyy * z;
                  xyyzz = xyyz * z;
                  xyzzz = xyzz * z;
                  xzzzz = xzzz * z;
                  yyyyy = yyyy * y;
                  yyyyz = yyyy * z;
                  yyyzz = yyyz * z;
                  yyzzz = yyzz * z;
                  yzzzz = yzzz * z;
                  zzzzz = zzzz * z;
                  M(20, c, b, a) = xxxx;
                  M(21, c, b, a) = xxxy;
                  M(22, c, b, a) = xxxz;
                  M(23, c, b, a) = xxyy;
                  M(24, c, b, a) = xxyz;
                  M(25, c, b, a) = xxzz;
                  M(26, c, b, a) = xyyy;
                  M(27, c, b, a) = xyyz;
                  M(28, c, b, a) = xyzz;
                  M(29, c, b, a) = xzzz;
                  M(30, c, b, a) = yyyy;
                  M(31, c, b, a) = yyyz;
                  M(32, c, b, a) = yyzz;
                  M(33, c, b, a) = yzzz;
                  M(34, c, b, a) = zzzz;
#ifndef USE_FUSED_F2
                  dMdrx(20, 0, c, b, a) = -rijinv * (4 * xxxxx - 3 * xxx);
                  dMdrx(20, 1, c, b, a) = -rijinv * 4 * xxxxy;
                  dMdrx(20, 2, c, b, a) = -rijinv * 4 * xxxxz;
                  dMdrx(21, 0, c, b, a) = -rijinv * (4 * xxxxy - 3 * xxy);
                  dMdrx(21, 1, c, b, a) = -rijinv * 4 * xxxyy - xxx;
                  dMdrx(21, 2, c, b, a) = -rijinv * 3 * xxyz;
                  dMdrx(22, 0, c, b, a) = -rijinv * (3 * xxxz - 2 * xz);
                  dMdrx(22, 1, c, b, a) = -rijinv * 3 * xxyz;
                  dMdrx(22, 2, c, b, a) = -rijinv * (3 * xxzz - xx);
                  dMdrx(23, 0, c, b, a) = -rijinv * (3 * xxyy - yy);
                  dMdrx(23, 1, c, b, a) = -rijinv * (3 * xyyy - 2 * xy);
                  dMdrx(23, 2, c, b, a) = -rijinv * 3 * xyyz;
                  dMdrx(24, 0, c, b, a) = -rijinv * (3 * xxyz - yz);
                  dMdrx(24, 1, c, b, a) = -rijinv * (3 * xyyz - xz);
                  dMdrx(24, 2, c, b, a) = -rijinv * (3 * xyzz - xy);
                  dMdrx(25, 0, c, b, a) = -rijinv * (3 * xxzz - zz);
                  dMdrx(25, 1, c, b, a) = -rijinv * 3 * xyzz;
                  dMdrx(25, 2, c, b, a) = -rijinv * (3 * xzzz - 2 * xz);
                  dMdrx(26, 0, c, b, a) = -rijinv * 3 * xyyy;
                  dMdrx(26, 1, c, b, a) = -rijinv * (3 * yyyy - 3 * yy);
                  dMdrx(26, 2, c, b, a) = -rijinv * 3 * yyyz;
                  dMdrx(27, 0, c, b, a) = -rijinv * 3 * xyyz;
                  dMdrx(27, 1, c, b, a) = -rijinv * (3 * yyyz - 2 * yz);
                  dMdrx(27, 2, c, b, a) = -rijinv * (3 * yyzz - yy);
                  dMdrx(28, 0, c, b, a) = -rijinv * 3 * xyzz;
                  dMdrx(28, 1, c, b, a) = -rijinv * (3 * yyzz - zz);
                  dMdrx(28, 2, c, b, a) = -rijinv * (3 * yzzz - 2 * yz);
                  dMdrx(29, 0, c, b, a) = -rijinv * 3 * xzzz;
                  dMdrx(29, 1, c, b, a) = -rijinv * 3 * yzzz;
                  dMdrx(29, 2, c, b, a) = -rijinv * (3 * zzzz - 3 * zz);
                  dMdrx(30, 0, c, b, a) = -rijinv * (4 * xxxxx - 3 * xxx);
                  dMdrx(30, 1, c, b, a) = -rijinv * 3 * xxxy;
                  dMdrx(30, 2, c, b, a) = -rijinv * 3 * xxxz;
                  dMdrx(31, 0, c, b, a) = -rijinv * (3 * xxxy - 2 * xy);
                  dMdrx(31, 1, c, b, a) = -rijinv * (3 * xxyy - xx);
                  dMdrx(31, 2, c, b, a) = -rijinv * 3 * xxyz;
                  dMdrx(32, 0, c, b, a) = -rijinv * (3 * xxxz - 2 * xz);
                  dMdrx(32, 1, c, b, a) = -rijinv * 3 * xxyz;
                  dMdrx(32, 2, c, b, a) = -rijinv * (3 * xxzz - xx);
                  dMdrx(33, 0, c, b, a) = -rijinv * (3 * xxyy - yy);
                  dMdrx(33, 1, c, b, a) = -rijinv * (3 * xyyy - 2 * xy);
                  dMdrx(33, 2, c, b, a) = -rijinv * 3 * xyyz;
                  dMdrx(34, 0, c, b, a) = -rijinv * (3 * xxyz - yz);
                  dMdrx(34, 1, c, b, a) = -rijinv * (3 * xyyz - xz);
                  dMdrx(34, 2, c, b, a) = -rijinv * (3 * xyzz - xy);
#endif
                  if (max_moment > 4) {
                    xxxxxx = xxxxx * x;
                    xxxxxy = xxxxx * y;
                    xxxxxz = xxxxx * z;
                    xxxxyy = xxxxy * y;
                    xxxxyz = xxxxy * z;
                    xxxxzz = xxxxz * z;
                    xxxyyy = xxxyy * y;
                    xxxyyz = xxxyy * z;
                    xxxyzz = xxxyz * z;
                    xxxzzz = xxxzz * z;
                    xxyyyy = xxyyy * y;
                    xxyyyz = xxyyy * z;
                    xxyyzz = xxyyz * z;
                    xxyzzz = xxyzz * z;
                    xxzzzz = xxzzz * z;
                    xyyyyy = xyyyy * y;
                    xyyyyz = xyyyy * z;
                    xyyyzz = xyyyz * z;
                    xyyzzz = xyyzz * z;
                    xyzzzz = xyzzz * z;
                    xzzzzz = xzzzz * z;
                    yyyyyy = yyyyy * y;
                    yyyyyz = yyyyy * z;
                    yyyyzz = yyyyz * z;
                    yyyzzz = yyyzz * z;
                    yyzzzz = yyzzz * z;
                    yzzzzz = yzzzz * z;
                    zzzzzz = zzzzz * z;
                    M(35, c, b, a) = xxxxx;
                    M(36, c, b, a) = xxxxy;
                    M(37, c, b, a) = xxxxz;
                    M(38, c, b, a) = xxxyy;
                    M(39, c, b, a) = xxxyz;
                    M(40, c, b, a) = xxxzz;
                    M(41, c, b, a) = xxyyy;
                    M(42, c, b, a) = xxyyz;
                    M(43, c, b, a) = xxyzz;
                    M(44, c, b, a) = xxzzz;
                    M(45, c, b, a) = xyyyy;
                    M(46, c, b, a) = xyyyz;
                    M(47, c, b, a) = xyyzz;
                    M(48, c, b, a) = xyzzz;
                    M(49, c, b, a) = xzzzz;
                    M(50, c, b, a) = yyyyy;
                    M(51, c, b, a) = yyyyz;
                    M(52, c, b, a) = yyyzz;
                    M(53, c, b, a) = yyzzz;
                    M(54, c, b, a) = yzzzz;
                    M(55, c, b, a) = zzzzz;
#ifndef USE_FUSED_F2
                    dMdrx(35, 0, c, b, a) = -rijinv * (4 * xxxxx - 3 * xxx);
                    dMdrx(35, 1, c, b, a) = -rijinv * 4 * xxxxy;
                    dMdrx(35, 2, c, b, a) = -rijinv * 4 * xxxxz;
                    dMdrx(36, 0, c, b, a) = -rijinv * (4 * xxxxy - 3 * xxy);
                    dMdrx(36, 1, c, b, a) = -rijinv * (4 * xxxyy - xxx);
                    dMdrx(36, 2, c, b, a) = -rijinv * 3 * xxyz;
                    dMdrx(37, 0, c, b, a) = -rijinv * (3 * xxxz - 2 * xz);
                    dMdrx(37, 1, c, b, a) = -rijinv * 3 * xxyz;
                    dMdrx(37, 2, c, b, a) = -rijinv * (3 * xxzz - xx);
                    dMdrx(38, 0, c, b, a) = -rijinv * (3 * xxyy - yy);
                    dMdrx(38, 1, c, b, a) = -rijinv * (3 * xyyy - 2 * xy);
                    dMdrx(38, 2, c, b, a) = -rijinv * 3 * xyyz;
                    dMdrx(39, 0, c, b, a) = -rijinv * (3 * xxyz - yz);
                    dMdrx(39, 1, c, b, a) = -rijinv * (3 * xyyz - xz);
                    dMdrx(39, 2, c, b, a) = -rijinv * (3 * xyzz - xy);
                    dMdrx(40, 0, c, b, a) = -rijinv * (3 * xxzz - zz);
                    dMdrx(40, 1, c, b, a) = -rijinv * 3 * xyzz;
                    dMdrx(40, 2, c, b, a) = -rijinv * (3 * xzzz - 2 * xz);
                    dMdrx(41, 0, c, b, a) = -rijinv * 3 * xyyy;
                    dMdrx(41, 1, c, b, a) = -rijinv * (3 * yyyy - 3 * yy);
                    dMdrx(41, 2, c, b, a) = -rijinv * 3 * yyyz;
                    dMdrx(42, 0, c, b, a) = -rijinv * 3 * xyyz;
                    dMdrx(42, 1, c, b, a) = -rijinv * (3 * yyyz - 2 * yz);
                    dMdrx(42, 2, c, b, a) = -rijinv * (3 * yyzz - yy);
                    dMdrx(43, 0, c, b, a) = -rijinv * 3 * xyzz;
                    dMdrx(43, 1, c, b, a) = -rijinv * (3 * yyzz - zz);
                    dMdrx(43, 2, c, b, a) = -rijinv * (3 * yzzz - 2 * yz);
                    dMdrx(44, 0, c, b, a) = -rijinv * 3 * xzzz;
                    dMdrx(44, 1, c, b, a) = -rijinv * 3 * yzzz;
                    dMdrx(44, 2, c, b, a) = -rijinv * (3 * zzzz - 3 * zz);
                    dMdrx(45, 0, c, b, a) = -rijinv * (4 * xxxxx - 3 * xxx);
                    dMdrx(45, 1, c, b, a) = -rijinv * 3 * xxxy;
                    dMdrx(45, 2, c, b, a) = -rijinv * 3 * xxxz;
                    dMdrx(46, 0, c, b, a) = -rijinv * (3 * xxxy - 2 * xy);
                    dMdrx(46, 1, c, b, a) = -rijinv * (3 * xxyy - xx);
                    dMdrx(46, 2, c, b, a) = -rijinv * 3 * xxyz;
                    dMdrx(47, 0, c, b, a) = -rijinv * (3 * xxxz - 2 * xz);
                    dMdrx(47, 1, c, b, a) = -rijinv * 3 * xxyz;
                    dMdrx(47, 2, c, b, a) = -rijinv * (3 * xxzz - xx);
                    dMdrx(48, 0, c, b, a) = -rijinv * (3 * xxyy - yy);
                    dMdrx(48, 1, c, b, a) = -rijinv * (3 * xyyy - 2 * xy);
                    dMdrx(48, 2, c, b, a) = -rijinv * 3 * xyyz;
                    dMdrx(49, 0, c, b, a) = -rijinv * (3 * xxyz - yz);
                    dMdrx(49, 1, c, b, a) = -rijinv * (3 * xyyz - xz);
                    dMdrx(49, 2, c, b, a) = -rijinv * (3 * xyzz - xy);
                    dMdrx(50, 0, c, b, a) = -rijinv * (3 * xxzz - zz);
                    dMdrx(50, 1, c, b, a) = -rijinv * 3 * xyzz;
                    dMdrx(50, 2, c, b, a) = -rijinv * (3 * xzzz - 2 * xz);
                    dMdrx(51, 0, c, b, a) = -rijinv * 3 * xyyy;
                    dMdrx(51, 1, c, b, a) = -rijinv * (3 * yyyy - 3 * yy);
                    dMdrx(51, 2, c, b, a) = -rijinv * 3 * yyyz;
                    dMdrx(52, 0, c, b, a) = -rijinv * 3 * xyyz;
                    dMdrx(52, 1, c, b, a) = -rijinv * (3 * yyyz - 2 * yz);
                    dMdrx(52, 2, c, b, a) = -rijinv * (3 * yyzz - yy);
                    dMdrx(53, 0, c, b, a) = -rijinv * 3 * xyzz;
                    dMdrx(53, 1, c, b, a) = -rijinv * (3 * yyzz - zz);
                    dMdrx(53, 2, c, b, a) = -rijinv * (3 * yzzz - 2 * yz);
                    dMdrx(54, 0, c, b, a) = -rijinv * 3 * xzzz;
                    dMdrx(54, 1, c, b, a) = -rijinv * 3 * yzzz;
                    dMdrx(54, 2, c, b, a) = -rijinv * (3 * zzzz - 3 * zz);
                    dMdrx(55, 0, c, b, a) = -rijinv * (4 * xxxxx - 3 * xxx);
                    dMdrx(55, 1, c, b, a) = -rijinv * 3 * xxxy;
                    dMdrx(55, 2, c, b, a) = -rijinv * 3 * xxxz;
#endif
                  }
                }
              }
            }
          }
        }
        neigh[b]++;
      }
    }

    size_t n, offset;
    size_t bytes = sizeof(Scalar);
    auto bm = static_cast<size_t>(neltypes);
    auto cm = static_cast<size_t>(numneigh_max);
    auto dm = static_cast<size_t>(nv_dims);
    size_t xm = 3;

    for (b = 0; b < neltypes; b++) {
      for (c = neigh[b]; c < numneigh_max; c++) {
        ijlist(0, c, b, a) = -1;
        ijlist(1, c, b, a) = -1;
        mask(c, b, a) = 0;
        R(c, b, a) = 0.0;
        drdrx(0, c, b, a) = 0.0;
        drdrx(1, c, b, a) = 0.0;
        drdrx(2, c, b, a) = 0.0;
        for (d = 0; d < nv_dims; d++) {
          M(d, c, b, a) = 0.0;
        }
#ifndef USE_FUSED_F2
        for (d = 0; d < nv_dims; d++) {
          dMdrx(d, 0, c, b, a) = 0.0;
          dMdrx(d, 1, c, b, a) = 0.0;
          dMdrx(d, 2, c, b, a) = 0.0;
        }
#endif
      }
    }
  }
  timer->record(TIMER::SETUP);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::calc_tensor_density(
    int *ilist, const int inum, const int *type, const int *fmap, double **pos,
    int numneigh_max, int *numneigh, int **firstneigh)
{
  int a, b, c, d, k, m;

  // Setup the R, M, dMdx, dMdrx tensors
  setup_tensors(ilist, inum, type, fmap, pos, numneigh_max, numneigh,
                firstneigh);

  // Subsets
  a = alocal;
  b = static_cast<int>(R.dimension(1));
  c = cmax;
  d = nv_dims;
  k = num_filters;
  m = max_moment + 1;
  TensorMap<Tensor<Scalar, 3>> R_{R.data(), {c, b, a}};
  TensorMap<Tensor<Scalar, 3>> sij_{sij.data(), {c, b, a}};
  TensorMap<Tensor<Scalar, 3>> dsij_{dsij.data(), {c, b, a}};
  TensorMap<Tensor<Scalar, 4>> H_{H.data(), {k, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> dHdr_{dHdr.data(), {k, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> M_{M.data(), {d, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> P_{P.data(), {d, k, b, a}};
  TensorMap<Tensor<Scalar, 4>> G_{G.data(), {m, k, b, a}};

  // Calculate descriptors: FNN or analytic functions
  if (this->fnn && this->fnn->is_interpolatable()) {
    timer->tic();
    this->fnn->interpolate(R_.data(), a * b * c, H_.data(), dHdr_.data());
    timer->record(TIMER::FNN_INTERP);
  } else if (this->descriptor && this->descriptor->is_interpolatable()) {
    timer->tic();
    this->descriptor->interpolate(R_.data(), a * b * c, H_.data(),
                                  dHdr_.data());
    timer->record(TIMER::DESCRIPTOR);
  } else {
    Tensor<Scalar, 3> mask_f = mask.cast<Scalar>();
    TensorMap<Tensor<Scalar, 3>> mask_{mask_f.data(), {c, b, a}};

    // Calculate cutoff coefficients
    timer->tic();
    cutoff->compute(R_, mask_, &sij_, &dsij_);
    timer->record(TIMER::CUTOFF);

    timer->tic();
    if (this->fnn) {
      Eigen::array<int, 4> bcast{num_filters, 1, 1, 1};
      Eigen::array<int, 4> shape{1, c, b, a};
      auto tiled_sij = sij_.reshape(shape).broadcast(bcast);
      auto tiled_dsij = dsij_.reshape(shape).broadcast(bcast);
      this->fnn->forward(R_.data(), a * b * c, H_.data());
      timer->record(TIMER::FNN_FORWARD);
      timer->tic();
      dHdr_ = tiled_dsij * H_;
      H_ = tiled_sij * H_;
    } else {
      this->descriptor->compute(R_, sij_, dsij_, &H_, &dHdr_);
    }
    timer->record(TIMER::DESCRIPTOR);
  }

  // P_dkba = M_dcba * H_kcba
  timer->tic();
  kernel_P(M_, H_, &P_);
  timer->record(TIMER::P);

  // +/- sign for m = 0
  Tensor<Scalar, 3> sign = P_.chip(0, 0).sign();

  // Q_mkba = T_md * (P_dkba)**2
  timer->tic();
  Eigen::array<IndexPair<int>, 1> contract_dims = {IndexPair<int>(1, 0)};
  G_ = T.contract(P_.square(), contract_dims);
  timer->record(TIMER::Q);

  // G_mkba
  timer->tic();
  G_.chip(0, 0) = G_.chip(0, 0).sqrt() * sign;
  timer->record(TIMER::G);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
void TensorMD<Scalar>::setup_interpolation(Scalar delta, int algo)
{
  if (rmax == 0.0) error->all(FLERR, "Invalid rmax for interpolation");

  if (this->fnn)
    this->fnn->setup_ration(delta, rmax, algo, cutoff);
  else
    this->descriptor->setup_ration(delta, rmax, algo, cutoff);
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
Scalar TensorMD<Scalar>::run(int *typenums, double etemp,
                                      double *eatom, double **f, double **vatom)
{
  timer->tic();

  int k, a, b, c, d, x, m;
  int i, j, elti;
  Scalar etotal;
  Scalar *x_in;
  Scalar *y;
  Scalar *eentropy;
  double df[3], virial[6];

  etotal = 0.0;
  y = eatom ? new Scalar[alocal] : nullptr;

  eentropy = nullptr;

  // Forward propagation: compute E and dEdQ
  for (elti = 0; elti < neltypes; elti++) {
    x_in = &G.data()[eltind[elti].offset * total_dims];
    if (this->nnp) {
      etotal += this->nnp->compute(
          elti, x_in, typenums[elti], eatom ? &y[eltind[elti].offset] : nullptr,
          &dEdG.data()[eltind[elti].offset * total_dims]);
    }
  }
  if (eatom) {
    for (a = 0; a < alocal; a++) {
      if (neltypes == 1) {
        eatom[a] = static_cast<double>(y[a]);
      } else {
        eatom[row2i[a]] = static_cast<double>(y[a]);
      }
    }
  }
  timer->record(TIMER::NN_COMPUTE);

  // Declare the tensor dimensions
  a = alocal;
  b = neltypes;
  c = static_cast<int>(M.dimension(1));
  d = nv_dims;
  k = num_filters;
  x = 3;
  m = max_moment + 1;
  TensorMap<Tensor<Scalar, 3>> R_{R.data(), {c, b, a}};
  TensorMap<Tensor<Scalar, 3>> sij_{sij.data(), {c, b, a}};
  TensorMap<Tensor<Scalar, 4>> drdrx_{drdrx.data(), {x, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> F1_{F1.data(), {x, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> H_{H.data(), {k, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> U_{U.data(), {k, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> dHdr_{dHdr.data(), {k, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> M_{M.data(), {d, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> V_{V.data(), {d, c, b, a}};
  TensorMap<Tensor<Scalar, 4>> P_{P.data(), {d, k, b, a}};
  TensorMap<Tensor<Scalar, 4>> G_{G.data(), {m, k, b, a}};
  TensorMap<Tensor<Scalar, 4>> dEdG_{dEdG.data(), {m, k, b, a}};

  timer->tic();

  // dEdG_mkba = dEdG_mkba * dGdQ_mkba
  // dGdQ.chip(0, 0) = 0.5 / G.chip(0, 0)
  // dGdQ.chip(m>0, 0) = 1.0
  kernel_dEdG(G_, &dEdG_);

  // dEdP_dkba = 2 * T_md * dGdQ_mkba * P_dkba
  // Eigen::array<IndexPair<int>, 1> contract_dims_1{IndexPair<int>{0, 0}};
  TensorMap<Tensor<Scalar, 4>> dEdS_{dEdS.data(), {d, k, b, a}};
  kernel_dEdS(T, dEdG_, P_, &dEdS_);
  timer->record(TIMER::DEDP);

  // U_kcba = dEdS_dkba * M_dcba
  // V_dcba = dEdS_dkba * H_kcba
  timer->tic();
  kernel_U<Scalar>(dEdS_, M_, &U_);
  kernel_V<Scalar>(dEdS_, H_, &V_);
  timer->record(TIMER::UV);

  // F1_xcba = U_kcba * dHdr_kcba * drdrx_xcba
  timer->tic();
  Eigen::array<int, 1> sum_axis{0};
  Eigen::array<int, 4> shape{1, c, b, a};
  Eigen::array<int, 4> bcast_xcba{x, 1, 1, 1};
  if (this->fnn && !this->fnn->is_interpolatable()) {
    Eigen::array<int, 4> bcast_kcba{num_filters, 1, 1, 1};
    Tensor<Scalar, 4> dHdrp{1, c, b, a};
    Tensor<Scalar, 4> Up = U * sij_.reshape(shape).broadcast(bcast_kcba);
    this->fnn->backward(Up.data(), dHdrp.data());
    auto grad = (U * dHdr_).sum(sum_axis).reshape(shape) + dHdrp;
    F1_ = grad.broadcast(bcast_xcba) * drdrx_;
    timer->record(TIMER::FNN_BACKWARD);
  } else {
#ifndef USE_FUSED_F1
    kernel_F1<Scalar>(U, dHdr_, drdrx_, &F1_);
#else
    TensorMap<Tensor<int, 3>> mask_{mask.data(), {c, b, a}};
    kernel_F1<Scalar>(U, dHdr_, drdrx_, mask_, &F1_);
#endif
    timer->record(TIMER::F1);
  }

  // F2_xcba = dMdrx_dxcba * V_dcba
  timer->tic();
#ifndef USE_FUSED_F2
  TensorMap<Tensor<Scalar, 5>> dMdrx_{dMdrx.data(), {d, x, c, b, a}};
  TensorMap<Tensor<int, 3>> mask_{mask.data(), {c, b, a}};
  kernel_F2<Scalar>(dMdrx_, V_, mask_, &F2);
#else
  TensorMap<Tensor<int, 3>> mask_{mask.data(), {c, b, a}};
  kernel_fused_F2<Scalar>(max_moment, V_, R_, drdrx_, mask_, &F2);
#endif
  timer->record(TIMER::F2);

  // Scatter update forces
  timer->tic();
  for (a = 0; a < alocal; a++) {
    for (b = 0; b < neltypes; b++) {
      for (c = 0; c < R.dimension(0); c++) {
        if (mask(c, b, a) == 0) continue;
        i = ijlist(0, c, b, a);
        j = ijlist(1, c, b, a);
        if (i >= 0 && j >= 0) {
          for (x = 0; x < 3; x++) {
            df[x] = F1(x, c, b, a) + F2(x, c, b, a);
            f[i][x] += df[x];
            f[j][x] -= df[x];
          }
          if (vatom) {
            virial[0] = -drdrx(0, c, b, a) * df[0] * R(c, b, a);
            virial[1] = -drdrx(1, c, b, a) * df[1] * R(c, b, a);
            virial[2] = -drdrx(2, c, b, a) * df[2] * R(c, b, a);
            virial[3] = -drdrx(0, c, b, a) * df[1] * R(c, b, a);
            virial[4] = -drdrx(0, c, b, a) * df[2] * R(c, b, a);
            virial[5] = -drdrx(1, c, b, a) * df[2] * R(c, b, a);
            vatom[i][0] += 0.5 * virial[0];
            vatom[i][1] += 0.5 * virial[1];
            vatom[i][2] += 0.5 * virial[2];
            vatom[i][3] += 0.5 * virial[3];
            vatom[i][4] += 0.5 * virial[4];
            vatom[i][5] += 0.5 * virial[5];
            vatom[j][0] += 0.5 * virial[0];
            vatom[j][1] += 0.5 * virial[1];
            vatom[j][2] += 0.5 * virial[2];
            vatom[j][3] += 0.5 * virial[3];
            vatom[j][4] += 0.5 * virial[4];
            vatom[j][5] += 0.5 * virial[5];
          }
        }
      }
    }
  }
  timer->record(TIMER::FORCES);
  timer->begin();
  delete[] y;
  delete[] eentropy;
  return etotal;
}

/* ---------------------------------------------------------------------- */

template <typename Scalar>
double TensorMD<Scalar>::memory_usage() const
{
  size_t size = 0;
  size += T.size();
  size += M.size();
  size += V.size();
#ifndef USE_FUSED_F2
  size += dMdrx.size();
#endif
  size += R.size();
  size += H.size();
  size += U.size();
  size += dHdr.size();
  size += G.size() * 2;
  size += dEdS.size();
  size += P.size();
  size += drdrx.size();
  size += F1.size();
  size += F2.size();
  size += sij.size();
  size += dsij.size();
  auto bytes = static_cast<double>(sizeof(Scalar) * size);
  bytes += static_cast<double>(sizeof(int) * nmax * 2);
  bytes += ijlist.size() * sizeof(int);
  bytes += mask.size() * sizeof(int);
  if (fnn) bytes += fnn->get_memory_usage();
  if (nnp) bytes += nnp->get_memory_usage();
  return bytes;
}

/* ----------------------------------------------------------------------
Explicit instantiation
------------------------------------------------------------------------- */

template class LAMMPS_NS::TensorMD<float>;
template class LAMMPS_NS::TensorMD<double>;
