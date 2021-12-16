/* ----------------------------------------------------------------------
    This is the

    ██╗     ██╗ ██████╗  ██████╗  ██████╗ ██╗  ██╗████████╗███████╗
    ██║     ██║██╔════╝ ██╔════╝ ██╔════╝ ██║  ██║╚══██╔══╝██╔════╝
    ██║     ██║██║  ███╗██║  ███╗██║  ███╗███████║   ██║   ███████╗
    ██║     ██║██║   ██║██║   ██║██║   ██║██╔══██║   ██║   ╚════██║
    ███████╗██║╚██████╔╝╚██████╔╝╚██████╔╝██║  ██║   ██║   ███████║
    ╚══════╝╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝®

    DEM simulation engine, released by
    DCS Computing Gmbh, Linz, Austria
    http://www.dcs-computing.com, office@dcs-computing.com

    LIGGGHTS® is part of CFDEM®project:
    http://www.liggghts.com | http://www.cfdem.com

    Core developer and main author:
    Christoph Kloss, christoph.kloss@dcs-computing.com

    LIGGGHTS® is open-source, distributed under the terms of the GNU Public
    License, version 2 or later. It is distributed in the hope that it will
    be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. You should have
    received a copy of the GNU General Public License along with LIGGGHTS®.
    If not, see http://www.gnu.org/licenses . See also top-level README
    and LICENSE files.

    LIGGGHTS® and CFDEM® are registered trade marks of DCS Computing GmbH,
    the producer of the LIGGGHTS® software and the CFDEM®coupling software
    See http://www.cfdem.com/terms-trademark-policy for details.

-------------------------------------------------------------------------
    Contributing author and copyright for this file:
    This file is from LAMMPS, but has been modified. Copyright for
    modification:

    Copyright 2012-     DCS Computing GmbH, Linz
    Copyright 2009-2012 JKU Linz

    Copyright of original file:
    LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
    http://lammps.sandia.gov, Sandia National Laboratories
    Steve Plimpton, sjplimp@sandia.gov

    Copyright (2003) Sandia Corporation.  Under the terms of Contract
    DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
    certain rights in this software.  This software is distributed under
    the GNU General Public License.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(nve/sphere,FixNVESphere)

#else

#ifndef LMP_FIX_NVE_SPHERE_H
#define LMP_FIX_NVE_SPHERE_H

#include "fix_nve.h"

namespace LAMMPS_NS {

class FixNVESphere : public FixNVE {
 public:
  FixNVESphere(class LAMMPS *, int, char **);
  virtual ~FixNVESphere() {}
  void init();
  virtual void initial_integrate(int);
  virtual void final_integrate();

// ========== Additional create_atom1 function included by Tarun ======
  virtual double pair_dist(double *, double *);
  virtual void sort_neigh(int *, int,int);
  virtual void place_CG(int *, int);
  int hasOverlap2(double *, double);
  //virtual void second_cg(int *, int);
  //virtual void Third_cg(int *, int);
  virtual void coarsen_particle(int *,int);
  virtual void refine_particle(int);
  virtual void avg(int *,int ,double *,double *);
  virtual void del(int *, int);
  virtual void create_cg(double *,double *,double *,double,double,double,int);
// ===================================================================================================

// ===================================================================================================
//=========== newly added by tarun ======================================
        // value of a fix property/atoms at insertion
        class FixPropertyAtom **fix_property;
        int n_fix_property;
        int *fix_property_nentry;
        double **fix_property_value;

        // a custom id for the history writing
        // in fix insert/stream/predefined
        int id_ins=-1;
        class FixPropertyAtom * const fix_release;

        void setFixTemplate(FixPropertyAtom* fix_template)
        { fix_template_ = fix_template; }

//=========================================================================




// ========== Additional variables included by Tarun ======
 private:
  int count1, count2=0, Neigh_List[1000][15];
  double positionsFisrtStage[1000][3],positions_SecondStage[1000][3];
  double **Distances;
  int *dlist;
int itype, tag1;
  FixPropertyAtom *fix_template_;

 protected:
  int extra;

  bool   useAM_;
  double CAddRhoFluid_;   //Added mass coefficient times relative fluid density (C_add*rhoFluid/rhoP)
  double onePlusCAddRhoFluid_;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Fix nve/sphere requires atom style sphere

Self-explanatory.

E: Fix nve/sphere requires atom attribute mu

An atom style with this attribute is needed.

E: Fix nve/sphere requires extended particles

This fix can only be used for particles of a finite size.

*/
