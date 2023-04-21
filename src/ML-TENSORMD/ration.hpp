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

#ifndef LAMMPS_TENSORMD_RATION1D_H
#define LAMMPS_TENSORMD_RATION1D_H

#include "error.h"

namespace LAMMPS_NS {

/*
 * Rational interpolation.
 * @ref: Rational function method of interpolation, G. Kerley, 1977.
 * */
template <typename Scalar> class Ration1d {

 public:
  explicit Ration1d(Error *err, bool endpoints_force_zero_ = true)
  {
    error = err;
    y = nullptr;
    endpoints_force_zero = endpoints_force_zero_;
    size = 0;
    xdx = 0.0;
    dx = 0.0;
  };

  /* ---------------------------------------------------------------------- */

  ~Ration1d() { delete[] this->y; };

  /* ---------------------------------------------------------------------- */

  virtual double get_memory_usage()
  {
    return static_cast<double>(size * sizeof(Scalar));
  }

  /* ---------------------------------------------------------------------- */

  void setup(int n, Scalar delta, Scalar *f)
  {
    this->size = n;
    this->dx = delta;
    this->xdx = 1.0 / delta;
    this->y = new Scalar[n];
    for (int i = 0; i < n; ++i) { this->y[i] = f[i]; }
  }

  /* ---------------------------------------------------------------------- */

  virtual void compute(Scalar *x, int n, Scalar *f, Scalar *df, int stride)
  {
    int i, j, offset;
    Scalar q, r, s, sm, sp, c1, c2, c3, c4;
    Scalar fj, dfj;
    offset = 0;
    for (j = 0; j < n; j++) {
      i = static_cast<int>(x[j] * this->xdx);
      i = MIN(this->size - 2, i);
      q = x[j] - static_cast<Scalar>(i) * dx;
      r = dx - q;
      s = (y[i + 1] - y[i]) / dx;
      if (i > 0) {
        sm = (y[i] - y[i - 1]) / dx;
        c1 = (s - sm) / dx / 2.0;
        if (i < this->size - 2) {
          if (i > 1 && sm * (sm - dx * c1) <= 0.0) c1 = (s - sm * 2) / dx;
          sp = (y[i + 2] - y[i + 1]) / dx;
          c2 = (sp - s) / 2 / dx;
          c3 = abs(c2 * r);
          c4 = c3 + abs(c1 * q);
          if (c4 > 0.0) c3 /= c4;
          c4 = c2 + c3 * (c1 - c2);
          fj = y[i] + q * (s - r * c4);
          dfj = s + (q - r) * c4 + dx * (c4 - c2) * (1 - c3);
        } else {
          if (endpoints_force_zero) {
            fj = dfj = 0.0;
          } else {
            fj = y[i] + q * (s - r * c1);
            dfj = s + (q - r) * c1;
          }
        }
      } else {
        if (endpoints_force_zero) {
          fj = dfj = 0.0;
        } else {
          sp = (y[i + 2] - y[i + 1]) / dx;
          c2 = (sp - s) / 2 / dx;
          if (s * (s - dx * c2) <= 0.0) c2 = s / dx;
          fj = y[i] + q * (s - r * c2);
          dfj = s + (q - r) * c2;
        }
      }
      if (f) f[j + offset] = fj;
      if (df) df[j + offset] = dfj;
      offset += stride - 1;
    }
  }

  virtual void compute(Scalar *x, int n, Scalar *f, Scalar *df)
  {
    compute(x, n, f, df, 1);
  }

  /* ---------------------------------------------------------------------- */

 protected:
  Scalar *y;
  Scalar dx, xdx;
  int size;
  bool endpoints_force_zero;

  Error *error;
};

/*
 * Batch implementation of Ration1d.
 * */
template <typename Scalar> class BatchRation1d : Ration1d<Scalar> {

 public:
  explicit BatchRation1d(Error *err) : Ration1d<Scalar>(err)
  {
    params = nullptr;
    n_out = 0;
  };

  /* ---------------------------------------------------------------------- */

  ~BatchRation1d() { delete[] this->params; };

  /* ---------------------------------------------------------------------- */

  double get_memory_usage()
  {
    return static_cast<double>(this->size * 4 * n_out * sizeof(Scalar));
  }

  /* ---------------------------------------------------------------------- */

  void setup_direct(int ndim, int n, Scalar delta, Scalar *f)
  {
    this->n_out = ndim;
    this->size = n;
    this->dx = delta;
    this->xdx = 1.0 / delta;
    params = new Scalar[this->size * n_out];
    for (int i = 0; i < this->size * n_out; i++) {
      params[i] = f[i];
    }
  }

  void setup(int ndim, int n, Scalar delta, Scalar *f)
  {
    this->n_out = ndim;
    this->size = n;
    this->dx = delta;
    this->xdx = 1.0 / delta;
    params = new Scalar[this->size * 4 * n_out];
    for (int i = 0; i < this->size; ++i) {
      Scalar *y = &params[i * 4 * n_out];
      for (int j = 0; j < n_out; j++) y[j] = f[i * n_out + j];
    }
    for (int i = 0; i < this->size - 1; i++) {
      Scalar *y = &params[i * 4 * n_out];
      Scalar *y1 = &params[(i + 1) * 4 * n_out];
      Scalar *s = &params[(i * 4 + 1) * n_out];
      for (int j = 0; j < n_out; j++) s[j] = (y1[j] - y[j]) * this->xdx;
    }
    Scalar c1, c2, s, sm, sp;
    for (int i = 0; i < this->size - 1; i++) {
      Scalar *p_c1 = &params[(i * 4 + 2) * n_out];
      Scalar *p_c2 = &params[(i * 4 + 3) * n_out];
      Scalar *p_s = &params[(i * 4 + 1) * n_out];
      Scalar *p_sm = &params[((i - 1) * 4 + 1) * n_out];
      Scalar *p_sp = &params[((i + 1) * 4 + 1) * n_out];
      if (i == 0) {
        for (int j = 0; j < n_out; j++) {
          s = p_s[j];
          sp = p_sp[j];
          if (s * (s - (sp - s) / 2) <= 0)
            c2 = s * this->xdx;
          else
            c2 = (sp - s) * this->xdx / 2;
          p_c1[j] = 0;
          p_c2[j] = c2;
        }
      } else if (i == this->size - 2) {
        for (int j = 0; j < n_out; j++) {
          s = p_s[j];
          sm = p_sm[j];
          c1 = (s - sm) * this->xdx / 2;
          p_c1[j] = c1;
          p_c2[j] = 0;
        }
      } else {
        for (int j = 0; j < n_out; j++) {
          s = p_s[j];
          sm = p_sm[j];
          sp = p_sp[j];
          if (i == 1 && sm * (sm - (s - sm) / 2) <= 0)
            c1 = (s - 2 * sm) * this->xdx;
          else
            c1 = (s - sm) * this->xdx / 2;
          c2 = (sp - s) * this->xdx / 2;
          p_c1[j] = c1;
          p_c2[j] = c2;
        }
      }
    }
  }

 /* ---------------------------------------------------------------------- */

  void compute_direct(Scalar *x, int n, Scalar *f, Scalar *df)
  {
    int i, ip1, ip2, im1;
    Scalar q, r, s, c1, c2, c3, c4, sm, sp;
    for (int k = 0; k < n; k++) {
      i = static_cast<int>(x[k] * this->xdx);
      i = MIN(this->size - 2, i);
      q = x[k] - static_cast<Scalar>(i) * this->dx;
      r = this->dx - q;
      ip1 = MIN(this->size - 1, i + 1);
      ip2 = MIN(this->size - 1, i + 2);
      im1 = MAX(0, i - 1);
      Scalar *y = &params[i * n_out];
      Scalar *yp1 = &params[ip1 * n_out];
      Scalar *yp2 = &params[ip2 * n_out];
      Scalar *ym1 = &params[im1 * n_out];
      Scalar *p_f = &f[k * n_out];
      Scalar *p_df = &df[k * n_out];
      if (i == 0) {
        if (Ration1d<Scalar>::endpoints_force_zero) {
          for (int j = 0; j < n_out; j++) {
            p_f[j] = 0.0;
            p_df[j] = 0.0;
          }
        } else {
          for (int j = 0; j < n_out; j++) {
            s = (yp1[j] - y[j]) / this->dx;
            sp = (yp2[j] - yp1[j]) / this->dx;
            c2 = (sp - s) / 2 / this->dx;
            if (s * (s - this->dx * c2) <= 0.0) c2 = s / this->dx;
            p_f[j] = y[j] + q * (s - r * c2);
            p_df[j] = s + (q - r) * c2;
          }
        }
      } else {
        for (int j = 0; j < n_out; j++) {
          s = (yp1[j] - y[j]) / this->dx;
          sm = (y[j] - ym1[j]) / this->dx;
          c1 = (s - sm) / this->dx / 2.0;
          if (i == this->size - 2) {
            if (Ration1d<Scalar>::endpoints_force_zero) {
              p_f[j] = 0.0;
              p_df[j] = 0.0;
            } else {
              p_f[j] = y[j] + q * (s - r * c1);
              p_df[j] = s + (q - r) * c1;
            }
          } else {
            if (i > 1 && sm * (sm - this->dx * c1) <= 0.0) {
              c1 = (s - sm * 2.0) / this->dx;
            }
            sp = (yp2[j] - yp1[j]) / this->dx;
            c2 = (sp - s) / 2 / this->dx;
            c3 = abs(c2 * r);
            c4 = c3 + abs(c1 * q);
            if (c4 > 0.0) c3 /= c4;
            c4 = c2 + c3 * (c1 - c2);
            p_f[j] = y[j] + q * (s + q * (c1 + q * c2));
            p_df[j] = s + (q - r) * c4 + this->dx * (c4 - c2) * (1 - c3);
          }
        }
      }
    }
  }

  void compute(Scalar *x, int n, Scalar *f, Scalar *df)
  {
    int i, k;
    Scalar q, r, s, c1, c2, c3, c4;
#if defined(USE_OPENMP)
#if defined(USE_OMP_RATION)
#pragma omp parallel for private(i, k, q, r, s, c1, c2, c3, c4)
#else
    omp_set_num_threads(1);
#endif
#endif
    for (k = 0; k < n; k++) {
      i = static_cast<int>(x[k] * this->xdx);
      i = MIN(this->size - 2, i);
      q = x[k] - static_cast<Scalar>(i) * this->dx;
      r = this->dx - q;
      Scalar *p_y = &params[(i * 4 + 0) * n_out];
      Scalar *p_s = &params[(i * 4 + 1) * n_out];
      Scalar *p_c1 = &params[(i * 4 + 2) * n_out];
      Scalar *p_c2 = &params[(i * 4 + 3) * n_out];
      Scalar *p_f = &f[k * n_out];
      Scalar *p_df = &df[k * n_out];
      if (i == 0) {
        if (Ration1d<Scalar>::endpoints_force_zero) {
          for (int j = 0; j < n_out; j++) {
            p_f[j] = 0.0;
            p_df[j] = 0.0;
          }
        } else {
          for (int j = 0; j < n_out; j++) {
            s = p_s[j];
            c2 = p_c2[j];
            p_f[j] = p_y[j] + q * (s - r * c2);
            p_df[j] = s + (q - r) * c2;
          }
        }
      } else if (i == this->size - 2) {
        if (Ration1d<Scalar>::endpoints_force_zero) {
          for (int j = 0; j < n_out; j++) {
            p_f[j] = 0.0;
            p_df[j] = 0.0;
          }
        } else {
          for (int j = 0; j < n_out; j++) {
            s = p_s[j];
            c1 = p_c1[j];
            p_f[j] = p_y[j] + q * (s - r * c1);
            p_df[j] = s + (q - r) * c1;
          }
        }
      } else if (i > this->size - 2 || i < 0) {
        Ration1d<Scalar>::error->all(
            FLERR,
            fmt::format(
                "ERROR: i not in [0, size - 2] where i = {}, size = {}\n", i,
                this->size)
                .c_str());
      } else {
        for (int j = 0; j < n_out; j++) {
          s = p_s[j];
          c1 = p_c1[j];
          c2 = p_c2[j];
          c3 = abs(c2 * r);
          c4 = c3 + abs(c1 * q);
          if (c4 > 0.0) c3 /= c4;
          c4 = c2 + c3 * (c1 - c2);
          p_f[j] = p_y[j] + q * (s - r * c4);
          p_df[j] = s + (q - r) * c4 + this->dx * (c4 - c2) * (1 - c3);
        }
      }
    }
  }

  void compute_ghost(Scalar *x, int n, Scalar *f, Scalar *df)
  {
    Scalar q, r, s, c1, c2, c3, c4;
    for (int k = 0; k < n; k++) {
      int i = static_cast<int>(x[k] * this->xdx);
      q = x[k] - static_cast<Scalar>(i) * this->dx;
      r = this->dx - q;
      // -2dx, dx, 0, dx, 2dx, ...,     rmax, rmax + dx, rmax + 2dx
      //   -2, -1, 0,  1,   2, ...,     npts,  npts + 1,   npts + 2
      //    0,  1, 2,  3,   4, ..., npts + 2,  npts + 3,   npts + 4
      // size = npts + 5
      // assert(i >= 0 && i <= size - 5);
      i += 2;
      Scalar *p_y = &params[(i * 4 + 0) * n_out];
      Scalar *p_s = &params[(i * 4 + 1) * n_out];
      Scalar *p_c1 = &params[(i * 4 + 2) * n_out];
      Scalar *p_c2 = &params[(i * 4 + 3) * n_out];
      Scalar *p_f = &f[k * n_out];
      Scalar *p_df = &df[k * n_out];
      if (2 <= i && i <= this->size - 3) {
        for (int j = 0; j < n_out; j++) {
          if (Ration1d<Scalar>::endpoints_force_zero &&
              (i == 2 || i == this->size - 3)) {
            p_f[j] = 0.0;
            p_df[j] = 0.0;
          } else {
            s = p_s[j];
            c1 = p_c1[j];
            c2 = p_c2[j];
            c3 = abs(c2 * r);
            c4 = c3 + abs(c1 * q);
            if (c4 > 0.0) c3 /= c4;
            c4 = c2 + c3 * (c1 - c2);
            p_f[j] = p_y[j] + q * (s - r * c4);
            p_df[j] = s + (q - r) * c4 + this->dx * (c4 - c2) * (1 - c3);
          }
        }
      } else {
        Ration1d<Scalar>::error->all(
            FLERR,
            fmt::format("ERROR: i not in [0, npts] where i = {}, npts = {}\n",
                        i - 2, this->size - 5)
                .c_str());
      }
    }
  }

  /* ---------------------------------------------------------------------- */

 protected:
  Scalar *params;
  int n_out;
};

}    // namespace LAMMPS_NS

#endif
