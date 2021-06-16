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

#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "fix_nve_sphere.h"
#include "atom.h"
#include "atom_vec.h"
#include "update.h"
#include "respa.h"
#include "force.h"
#include "error.h"
#include "domain.h" 

// ========== Modification by Tarun (Part-1)  ================
// ========== Additional header files included by Tarun ======
#include "group.h"
#include "region.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "random_mars.h"
#include "memory.h"
#include "atom_vec.h"
#include "fix_property_atom.h"
#include "vector_liggghts.h"
#include "math_extra_liggghts.h"
#include "modify.h"
#include "particleToInsert.h"
// ========== Modification by Tarun (Part-1) ends =============

using namespace LAMMPS_NS;
using namespace FixConst;

#define INERTIA 0.4          // moment of inertia prefactor for sphere

enum{NONE,DIPOLE};

/* ---------------------------------------------------------------------- */

FixNVESphere::FixNVESphere(LAMMPS *lmp, int narg, char **arg) :
  FixNVE(lmp, narg, arg),
  useAM_(false),
  CAddRhoFluid_(0.0),

// ========== Modification by Tarun (Part-2) =============
  fix_release(NULL), 
// ========== Modification by Tarun (Part-2) ends =============

  onePlusCAddRhoFluid_(1.0)
{
  if (narg < 3) error->all(FLERR,"Illegal fix nve/sphere command");

  time_integrate = 1;

  // process extra keywords

  extra = NONE;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"update") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix nve/sphere command");
      if (strcmp(arg[iarg+1],"dipole") == 0) extra = DIPOLE;
      else if (strcmp(arg[iarg+1],"CAddRhoFluid") == 0)
      {
            if(narg < iarg+2)
                error->fix_error(FLERR,this,"not enough arguments for 'CAddRhoFluid'");
            iarg+=2;
            useAM_ = true;
            CAddRhoFluid_        = atof(arg[iarg]);
            onePlusCAddRhoFluid_ = 1.0 + CAddRhoFluid_;
            fprintf(screen,"cfd_coupling_force_implicit will consider added mass with CAddRhoFluid = %f\n",
                    CAddRhoFluid_);
      }
      else error->all(FLERR,"Illegal fix nve/sphere command");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix nve/sphere command");
  }

  // error checks

  if (!atom->sphere_flag)
    error->all(FLERR,"Fix nve/sphere requires atom style sphere");
  if (extra == DIPOLE && !atom->mu_flag)
    error->all(FLERR,"Fix nve/sphere requires atom attribute mu");
}

/* ---------------------------------------------------------------------- */

void FixNVESphere::init()
{
  FixNVE::init();

  // check that all particles are finite-size spheres
  // no point particles allowed

  double *radius = atom->radius;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      if (radius[i] == 0.0)
        error->one(FLERR,"Fix nve/sphere requires extended particles");
}

/* ---------------------------------------------------------------------- */

void FixNVESphere::initial_integrate(int vflag)
{
  double dtfm,dtirotate,msq,scale;
  double g[3];

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *radius = atom->radius;
  double *rmass = atom->rmass;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

// ========== Modification by Tarun (Part-3) ==================
  int ntimestep = update->ntimestep;
  int n_fine=0, to_coarsen[10000];
// ========== Modification by Tarun (Part-3) ends =============

  // set timestep here since dt may have changed or come via rRESPA

  double dtfrotate; 
  if (domain->dimension == 2) dtfrotate = dtf / 0.5; // for discs the formula is I=0.5*Mass*Radius^2
  else dtfrotate  = dtf / INERTIA;

  // update 1/2 step for v and omega, and full step for  x for all particles
  // d_omega/dt = torque / inertia

  for (int i = 0; i < nlocal; i++) {
    //if (mask[i] & groupbit) {//tarun

      // velocity update for 1/2 step
      dtfm = dtf / (rmass[i]*onePlusCAddRhoFluid_);
      v[i][0] += dtfm * f[i][0];
      v[i][1] += dtfm * f[i][1];
      v[i][2] += dtfm * f[i][2];

      // position update
      x[i][0] += dtv * v[i][0];
      x[i][1] += dtv * v[i][1];
      x[i][2] += dtv * v[i][2];
      
      // rotation update
      dtirotate = dtfrotate / (radius[i]*radius[i]*rmass[i]);
      omega[i][0] += dtirotate * torque[i][0];
      omega[i][1] += dtirotate * torque[i][1];
      omega[i][2] += dtirotate * torque[i][2];

// ========== Modification by Tarun (Part-4) =============

	if (x[i][2]>=0.025 && x[i][2]<0.04 && radius[i]==0.0006) //Refinement step
	{
		refine_particle(i);
	} 

	if (x[i][2]<=0.020 && x[i][2]>=0 && radius[i]==0.0003){
		to_coarsen[n_fine]=i;
		n_fine++;
	} 
      }

	if(n_fine>0)
	coarsen_particle(to_coarsen,n_fine);

// ========== Modification by Tarun (Part-4) ends =============

  // update mu for dipoles
  // d_mu/dt = omega cross mu
  // renormalize mu to dipole length

  if (extra == DIPOLE) {
    double **mu = atom->mu;
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
        if (mu[i][3] > 0.0) {
          g[0] = mu[i][0] + dtv * (omega[i][1]*mu[i][2]-omega[i][2]*mu[i][1]);
          g[1] = mu[i][1] + dtv * (omega[i][2]*mu[i][0]-omega[i][0]*mu[i][2]);
          g[2] = mu[i][2] + dtv * (omega[i][0]*mu[i][1]-omega[i][1]*mu[i][0]);
          msq = g[0]*g[0] + g[1]*g[1] + g[2]*g[2];
          scale = mu[i][3]/sqrt(msq);
          mu[i][0] = g[0]*scale;
          mu[i][1] = g[1]*scale;
          mu[i][2] = g[2]*scale;
        }
  }
}


// ========== Modification by Tarun (Part-5) =============

void FixNVESphere::refine_particle(int ith)
{
  //int nlocal = atom->nlocal;
		//printf("insertion starts\n");
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *radius = atom->radius;
  double *rmass = atom->rmass;
  int *mask = atom->mask;
  int *tag = atom->tag;
  int *type = atom->type;
  double *density = atom->density;

  int itype=1;
  double delta = radius[ith]/2;

//inserting child particles
  for (int i=-1; i<2; i=i+2){
    for (int k=-1; k<2 ; k=k+2){
	for (int l=-1; l<2 ; l=l+2){
	double x_new[3]= {x[ith][0]+l*delta, x[ith][1]+k*delta, x[ith][2]+i*delta};

        //insertion of child particles starts
        int distorder = -1;
    	int groupbit = 0;
	atom->avec->create_atom(itype ,x_new); 
                int m = atom->nlocal - 1;
                atom->mask[m] = 1 | groupbit;
                vectorCopy3D(v[ith],atom->v[m]);
                vectorCopy3D(omega[ith],atom->omega[m]);
                atom->radius[m] = radius[ith]/2;
                atom->density[m] = density[ith];
                atom->rmass[m] = rmass[ith]/8;

                //pre_set_arrays() called via FixParticleDistribution
                for (int j = 0; j < modify->nfix; j++)
                   if (modify->fix[j]->create_attribute) modify->fix[j]->set_arrays(m);

                // apply fix property setting coming from fix insert
                // this overrides the set_arrays call above
                if(fix_property)
                {
                    for (int j = 0; j < n_fix_property; j++)
                    {
                        if (fix_property_nentry[j] == 1)
                            fix_property[j]->vector_atom[m] = fix_property_value[j][0];
                        else
                        {
                            for (int k = 0; k < fix_property_nentry[j]; k++)
                                fix_property[j]->array_atom[m][k] = fix_property_value[j][k];
                        }
                    }
                }
                if (fix_template_)
                    fix_template_->vector_atom[m] = (double)distorder;
                if (fix_release)
                    fix_release->array_atom[m][14] = (double) id_ins;
	}
     }
  }


//deleting mother particle (inspired from command() function of delete_atoms.cpp)

  int nlocal = atom->nlocal;
  AtomVec *avec = atom->avec;
  avec->copy(nlocal-1,ith,1);
  nlocal--;
  atom->nlocal = nlocal;

		//printf("insertion ends\n");

}


void FixNVESphere::coarsen_particle(int *to_coarsen,int n_fine)
{
  int nlocal = atom->nlocal;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *radius = atom->radius;
  double *rmass = atom->rmass;
  int *mask = atom->mask;
  int *tag = atom->tag;
  int *type = atom->type;
  double *density = atom->density;

  int n_coarse=n_fine/8;

  double rad_cg=2*radius[to_coarsen[0]]; 
  double density_cg=density[to_coarsen[0]];
  double rmass_cg=8*rmass[to_coarsen[0]]; 
  double x_cg[3]={0,0,0}; 
  double v_cg[3]={0,0,0}; 
  double omega_cg[3]={0,0,0}; 

  if (n_coarse>0){
  for (int i=0; i<n_coarse; i++)
  {
	int del_particles[8];
	for (int j=0;j<8;j++)
	del_particles[j]=to_coarsen[i*8+j];

	avg(del_particles,x_cg,v_cg,omega_cg);
	del(del_particles);
	create_cg(x_cg,v_cg,omega_cg,rad_cg,density_cg,rmass_cg);


  }
  }
	
}

void FixNVESphere::create_cg(double *x_cg,double *v_cg,double *omega_cg,double rad_cg,double density_cg,double rmass_cg)
{
	int itype=1;
        int distorder = -1;
    	int groupbit = 0;
	atom->avec->create_atom(itype ,x_cg); 
  	printf("New position of particle: (%f, %f, %f) radius= %f, density= %f\n",x_cg[0],x_cg[1],x_cg[2],rad_cg,density_cg);
                int m = atom->nlocal - 1;
                atom->mask[m] = 1 | groupbit;
                vectorCopy3D(v_cg,atom->v[m]);
                vectorCopy3D(omega_cg,atom->omega[m]);
                atom->radius[m] = rad_cg;
                atom->density[m] = density_cg;
                atom->rmass[m] = rmass_cg;
	//printf("I was here1\n");
                //pre_set_arrays() called via FixParticleDistribution
                for (int j = 0; j < modify->nfix; j++)
                   if (modify->fix[j]->create_attribute) modify->fix[j]->set_arrays(m);

                // apply fix property setting coming from fix insert
                // this overrides the set_arrays call above
                if(fix_property)
                {
                    for (int j = 0; j < n_fix_property; j++)
                    {
                        if (fix_property_nentry[j] == 1)
                            fix_property[j]->vector_atom[m] = fix_property_value[j][0];
                        else
                        {
                            for (int k = 0; k < fix_property_nentry[j]; k++)
                                fix_property[j]->array_atom[m][k] = fix_property_value[j][k];
                        }
                    }
                }
                if (fix_template_)
                    fix_template_->vector_atom[m] = (double)distorder;
                if (fix_release)
                    fix_release->array_atom[m][14] = (double) id_ins;
	//printf("I was here2\n");
}



void FixNVESphere::del(int *del_particles)
{
  int nlocal = atom->nlocal;
  AtomVec *avec = atom->avec;
  	for (int j=0; j< 8; j++)
	{
	avec->copy(nlocal-1,del_particles[j],1);
  	nlocal--;
  	atom->nlocal = nlocal;
	}
}


void FixNVESphere::avg(int *del_particles,double *x_cg,double *v_cg,double *omega_cg)
{
  int nlocal = atom->nlocal;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *radius = atom->radius;
  double *rmass = atom->rmass;
  int *mask = atom->mask;
  int *tag = atom->tag;
  int *type = atom->type;
  double *density = atom->density;

  for (int i=0; i<3; i++){
      for (int j=0; j<8; j++){
	x_cg[i]+=x[del_particles[j]][i];
	v_cg[i]+=v[del_particles[j]][i];
	omega_cg[i]+=omega[del_particles[j]][i];
      }
  x_cg[i]=x_cg[i]/8;
  v_cg[i]=v_cg[i]/8;
  omega_cg[i]=omega_cg[i]/8;
  }
}


// ========== Modification by Tarun (Part-5) ends =============



/* ---------------------------------------------------------------------- */

void FixNVESphere::final_integrate()
{
  double dtfm,dtirotate;

  double **v = atom->v;
  double **f = atom->f;
  double **omega = atom->omega;
  double **torque = atom->torque;
  double *rmass = atom->rmass;
  double *radius = atom->radius;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  // set timestep here since dt may have changed or come via rRESPA

  double dtfrotate; 
  if (domain->dimension == 2) dtfrotate = dtf / 0.5; // for discs the formula is I=0.5*Mass*Radius^2
  else dtfrotate  = dtf / INERTIA;

  // update 1/2 step for v,omega for all particles
  // d_omega/dt = torque / inertia

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {

      // velocity update for 1/2 step
      dtfm = dtf / (rmass[i]*onePlusCAddRhoFluid_);
      v[i][0] += dtfm * f[i][0];
      v[i][1] += dtfm * f[i][1];
      v[i][2] += dtfm * f[i][2];

      // rotation update
      dtirotate = dtfrotate / (radius[i]*radius[i]*rmass[i]);
      omega[i][0] += dtirotate * torque[i][0];
      omega[i][1] += dtirotate * torque[i][1];
      omega[i][2] += dtirotate * torque[i][2];
    }
}
