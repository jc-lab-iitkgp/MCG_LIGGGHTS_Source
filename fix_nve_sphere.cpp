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
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
// ========== End of change by Tarun ======

using namespace LAMMPS_NS;
using namespace FixConst;

#define INERTIA 0.4          // moment of inertia prefactor for sphere

enum{NONE,DIPOLE};

/* ---------------------------------------------------------------------- */

FixNVESphere::FixNVESphere(LAMMPS *lmp, int narg, char **arg) :
  FixNVE(lmp, narg, arg),
  useAM_(false),
  CAddRhoFluid_(0.0),
  fix_release(NULL),
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
  int *type=atom->type;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;
  int n_fine=0, to_coarsen[200];//tarun
  int ntimesteps= update->ntimestep;
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

// ========== Modification by Tarun Starts (part-1 of 2)  ===================================
	if (x[i][2]<0.014 && radius[i]==0.0006) //Refinement step
	{
		refine_particle(i);
	} 
	if(ntimesteps%100==0)
	if (x[i][2]<=0.020 && x[i][2]>=00.015 && radius[i]==0.0003){
		to_coarsen[n_fine]=i;
		n_fine++;
	} 

	if (type[i]==2 && radius[i]<0.0006)
	radius[i]+=0.0000002;
	if (radius[i]>0.00058 && radius[i]<0.0006)
	radius[i]=0.0006;

	if (type[i]==1 && radius[i]>0.0002 && radius[i]<0.0003)
	radius[i]+=0.0000000804;
	}
	if(ntimesteps%100==0)
	if(n_fine>0)
	coarsen_particle(to_coarsen,n_fine);


// ========== Modification by Tarun ends (part-1 of 2)  ===================================

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



// ========== Modification by Tarun Starts (part-2 of 2)  ===================================

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
  double delta = radius[ith]*0.366;
  double vel_revised[3]={0,0,v[ith][2]};
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
                vectorCopy3D(vel_revised,atom->v[m]);
                vectorCopy3D(omega[ith],atom->omega[m]);
                atom->radius[m] = radius[ith]*0.366;
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


  double density_cg=density[to_coarsen[0]];
  double rmass_cg=8*rmass[to_coarsen[0]]; 
  double x_cg[3]={0,0,0}; 
  double v_cg[3]={0,0,0}; 
  double omega_cg[3]={0,0,0}; 
  int **nghbr;
 
  place_CG(to_coarsen,n_fine);
 
}

void FixNVESphere::place_CG(int *to_coarsen, int n_fine)
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
  int itype= type[to_coarsen[0]];
  int n_coarse=n_fine/8;
  n_fine=n_coarse*8;
  int n_fine_del=n_coarse*8;

  double rad_cg=2*radius[to_coarsen[0]]; 
  double density_cg=density[to_coarsen[0]];
  double rmass_cg=8*rmass[to_coarsen[0]]; 
  double x_cg[3]={0,0,0}; 
  double v_cg[3]={0,0,0}; 
  double omega_cg[3]={0,0,0}; 
  avg(to_coarsen, n_fine,v_cg,omega_cg);
  v_cg[0]=0; v_cg[1]=0;
  printf("n_fine=%d\n",n_fine);
  for (int i=0; i<1000; i++)
    for (int j=0; j<15; j++)
	Neigh_List[i][j]=0;


	

  for (int i=0; i<n_fine; i++)
    for (int j=0; j<3; j++)
	positionsFisrtStage[i][j]=x[to_coarsen[i]][j];



  for (int c=0;c<3;c++){
    count2=0;
    for (int i=0; i< n_fine; i++) // finding neighbours
    {
    count1=0;
    //printf("i=%d Neighbors are: ",to_coarsen[i]);
    for (int j=0; j< n_fine;j++){
	if(i!=j){
	double DistSq=0.0, RadSumSq=0.0;
	double skin=rad_cg/2;

	DistSq = pair_dist(positionsFisrtStage[i],positionsFisrtStage[j]);

	RadSumSq=(rad_cg+skin)*(rad_cg+skin);
	if (DistSq<=RadSumSq){

	Neigh_List[i][count1]=j;
	//Distances[i][count1]=DistSq; 
	//printf("%d ",j);
	count1=count1+1;
	}
	}
	}
	sort_neigh(Neigh_List[i],i,count1);
	printf("\n i=%d and the 1st neighbor is %d \n",i,Neigh_List[i][0]);
	if(Neigh_List[i][0]!=0 && positionsFisrtStage[Neigh_List[i][0]][0]!=0){
  		for (int j=0;j<3;j++) { 
  		positions_SecondStage[count2][j]=(positionsFisrtStage[i][j]+positionsFisrtStage[Neigh_List[i][0]][j])/2;   
		positionsFisrtStage[i][0]=0; positionsFisrtStage[Neigh_List[i][0]][0]=0;
  		}
	count2++;
	printf("\n");
    }
  }
  printf("\n \n \n count2 is %d \n\n\n\n\n\n\n\n",count2);
  while(count2<n_fine/2){
  for (int i=0; i< n_fine; i++){
	if (positionsFisrtStage[i][0]!=0){
	for (int j=0; j<n_fine; j++){
		if (i!=j){
		if (positionsFisrtStage[j][0]!=0){
		for (int k=0;k<3;k++) 
		positions_SecondStage[count2][k]=(positionsFisrtStage[i][k]+positionsFisrtStage[j][k])/2;
		positionsFisrtStage[i][0]=0; positionsFisrtStage[j][0]=0;
		count2++;
  }
  }
  }
  }
  }	
  }
  printf("\n \n \n count2 is %d \n\n\n\n\n\n\n\n",count2);
  for (int i=0;i<n_fine/2;i++){
	for (int j=0;j<3;j++){
	printf("%f ",positions_SecondStage[i][j]);
	}
	printf("\n");
  }


  for (int j=0; j<1000; j++)
    for (int k=0; k<15; k++)
	Neigh_List[j][k]=0;

  for (int j=0; j<n_fine; j++)
    for (int k=0; k<3; k++)
	positionsFisrtStage[j][k]=positions_SecondStage[j][k];

  n_fine=n_fine/2;
  }

  for (int i=0;i<n_coarse;i++)
	positionsFisrtStage[i][2]=positionsFisrtStage[i][2]-0.0006;

  for (int i=0;i<n_coarse;i++){
	if(hasOverlap2(positionsFisrtStage[i],0.0004002)==0){
	int flag1=0;
	double pos_part[3];
	do{
	  int seed1=rand();
	  const int k1 = seed1/IQ;
	  seed1 = IA*(seed1-k1*IQ) - IR*k1;
	  if (seed1 < 0) seed1 += IM;
	  double random1=AM*seed1;	
	 //m[0]=(rand()%28000-14000); //m[0]=(rand()%40000-20000);
	 pos_part[0]=-.0015+random1*(0.0015-(-.0015));
	  int seed2=rand();
	  const int k2 = seed2/IQ;
	  seed2 = IA*(seed2-k2*IQ) - IR*k2;
	  if (seed2 < 0) seed2 += IM;
	  double random2=AM*seed2;
	 //m[1]=(rand()%28000-14000); 
	 pos_part[1]=-.0015+random2*(0.0015-(-.0015));//+0.0002
	  int seed3=rand();
	  const int k3 = seed3/IQ;
	  seed3 = IA*(seed3-k3*IQ) - IR*k3;
	  if (seed3 < 0) seed3 += IM;
	  double random3=AM*seed3;
	 //m[2]=(rand()%300000); 
	 pos_part[2]=random3*(00.0015)+0.0185-0.0006;//+0.004;//0.014;//; //m[2]/10000;
	 //pos_part[2]=0.03;	 
	if(hasOverlap2(pos_part, 0.0004002)==0) {
             flag1=1;
         }
	 else {
             flag1=0;
         }
	positionsFisrtStage[i][0]=pos_part[0];
	positionsFisrtStage[i][1]=pos_part[1];
	positionsFisrtStage[i][2]=pos_part[2];
	}while(flag1==1);
	}


	create_cg(positionsFisrtStage[i],v_cg,omega_cg,0.0004002,density_cg,rmass_cg,itype);
    }

  del(to_coarsen,n_fine_del);
}


int FixNVESphere::hasOverlap2(double *pos_part, double rad_new)
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

	for (int i=0; i< nlocal; i++) {
	double del_x, del_y, del_z, dist_sq, sum_rad_sq;
	del_x=x[i][0]-pos_part[0];
	del_y=x[i][1]-pos_part[1];
	del_z=x[i][2]-pos_part[2];
	dist_sq=del_x*del_x+del_y*del_y+del_z*del_z;
	sum_rad_sq=(rad_new+radius[i])*(rad_new+radius[i]);
	if (dist_sq < sum_rad_sq)
	{
	return 0;
	break;
	}
	}
	return 1;
}




void FixNVESphere::sort_neigh(int *list, int ii, int count)
{
	for(int i=0;i<count;i++){
		for (int j=i+1; j<count;j++){
		if (pair_dist(positionsFisrtStage[list[ii]],positionsFisrtStage[list[i]])>pair_dist(positionsFisrtStage[list[ii]],positionsFisrtStage[list[j]])){
		int exch=list[i];
		list[i]=list[j];
		list[j]=exch;
		}
	}
  }
}

double FixNVESphere::pair_dist(double *first, double *second)
{
  double Dist;
  Dist=(first[0]-second[0])*(first[0]-second[0])+(first[1]-second[1])*(first[1]-second[1])+(first[2]-second[2])*(first[2]-second[2]);

  return Dist;

}

void FixNVESphere::create_cg(double *x_cg,double *v_cg,double *omega_cg,double rad_cg,double density_cg,double rmass_cg, int i_type)
{
	int itype=i_type+1;
        int distorder = -1;
    	int groupbit = 0;
	atom->avec->create_atom(itype ,x_cg); 
  	printf("(%f, %f, %f) %f, %f\n",x_cg[0],x_cg[1],x_cg[2],rad_cg,density_cg);
                int m = atom->nlocal - 1;
                atom->mask[m] = 1 | groupbit;
                vectorCopy3D(v_cg,atom->v[m]);
                vectorCopy3D(omega_cg,atom->omega[m]);
                atom->radius[m] = rad_cg;
                atom->density[m] = density_cg;
                atom->rmass[m] = rmass_cg;
	printf("I was here1\n");
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
	printf("I was here2\n");
}



void FixNVESphere::del(int *to_coarsen, int del_numbers)
{
  for (int i=0;i<del_numbers;i++){
  int nlocal = atom->nlocal;
  AtomVec *avec = atom->avec;
  avec->copy(nlocal-1,to_coarsen[i],1);
  nlocal--;
  atom->nlocal = nlocal;
  }
}


void FixNVESphere::avg(int *del_particles,int n_fine, double *v_cg, double *omega_cg)
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
      for (int j=0; j<n_fine; j++){
	v_cg[i]+=v[del_particles[j]][i];
	omega_cg[i]+=omega[del_particles[j]][i];
      }
  v_cg[i]=v_cg[i]/n_fine;
  omega_cg[i]=omega_cg[i]/n_fine;
  }
}



// ========== Modification by Tarun ends (part-2 of 2)  ===================================



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
