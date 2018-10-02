#include "boundary_pellet.h"
#include <iostream>
#include <cmath>
#include <cassert>
using namespace std;

PelletInflowBoundary::PelletInflowBoundary():Pinflow(30), Uinflow(0), Vinflow(100){}

double calculateMassFlowRate(double energy){
	return energy;
}

int PelletInflowBoundary::UpdateInflowBoundary(ParticleData* m_pParticleData, EOS* m_pEOS, double dt, double dx){
        size_t fluidStartIndex = m_pParticleData->m_iFluidStartIndex;
        size_t fluidEndIndex = m_pParticleData->m_iFluidStartIndex + m_pParticleData->m_iFluidNum;
        size_t inflowEndIndex = fluidEndIndex + m_pParticleData->m_iInflowNum;
        double *x = m_pParticleData->m_vPositionX;
        double *y = m_pParticleData->m_vPositionY;
        double *z = m_pParticleData->m_vPositionZ;
        double *pressure = m_pParticleData->m_vPressure;
        double *vx = m_pParticleData->m_vVelocityU;
        double *vy = m_pParticleData->m_vVelocityV;
        double *vz = m_pParticleData->m_vVelocityW;
        double *volume = m_pParticleData->m_vVolume;
        double *volumeold = m_pParticleData->m_vVolumeOld;
        double *localParSpacing = m_pParticleData->m_vLocalParSpacing;
        double *mass = m_pParticleData->m_vMass;
        double *sound = m_pParticleData->m_vSoundSpeed;

	int *pelletid = m_pParticleData->m_vPelletID;
	size_t pelletn = m_pParticleData->m_iNumberofPellet;
	double *pelletx = m_pParticleData->m_vPelletPositionX;
	double *pellety = m_pParticleData->m_vPelletPositionY;
	double *pelletz = m_pParticleData->m_vPelletPositionZ;
	double *pelletr = m_pParticleData->m_vPelletRadius;
	double *pelletir = m_pParticleData->m_vPelletInnerRadius;
	double *pellete = m_pParticleData->m_vPelletEnergy;
	double *qplusminus = m_pParticleData->m_vQplusminus;
	double sublimationenergy = m_pParticleData->sublimationenergy;
	double *pelletvelocity = m_pParticleData->m_vPelletVelocity;
	vector<double> pelletqsum(pelletn,0);
	vector<double> pelletvolumesum(pelletn,0);
	vector<int> pelletneighbor(pelletn,0);

	m_pParticleData->m_vMassFlowRate=0;

//	cout<<"insert particles"<<endl;
//	cout<<inflowEndIndex<<endl;
//	cout<<m_pParticleData->m_iCapacity<<endl;
	for(int pi=0;pi<pelletn;pi++){
		cout<<pi<<endl;
		double pir=pelletir[pi];
		double pr=pelletr[pi];
		//generate new inflow particles in region 0<r<pir
		int n=4.0*3.1416*pir*pr*pr/dx/dx/dx*sqrt(2.0);
		cout<<n<<endl;
		if(inflowEndIndex+n>=m_pParticleData->m_iCapacity) {
			cout<<"Error: too many inflow particles: n = "<<n<<endl;
			return 1;//too many
		}
		double newpir=pir-n*dx*dx*dx/4.0/3.1416/pr/pr/sqrt(2.0);
		

		//calculate ablation velocity
		for(size_t index=fluidStartIndex;index<fluidEndIndex;index++)
                {
                        double d_x=x[index]-pelletx[pi];
                        double d_y=y[index]-pellety[pi];
                        double d_z=z[index]-pelletz[pi];
                        double r=d_x*d_x+d_y*d_y+d_z*d_z;
                        if(r<(pr+dx)*(pr+dx) && r>pr*pr)
                        {
                                pelletqsum[pi]+=qplusminus[index];
                                pelletvolumesum[pi]+=volume[index];
			        pelletneighbor[pi]++;
                        }
                }

                if(pelletneighbor[pi]==0)
                {
                        cout<<"Error: cannot find neighbor for pellet"<<endl;
                        assert(false);
                        return 0;
                }
                cout<<"Number of neighbor for pellet = "<<pelletneighbor[pi]<<endl;
                pellete[pi]=pelletqsum[pi]/pelletneighbor[pi]*4*3.1416*pr*pr;
                double massflowrate=pellete[pi]/sublimationenergy;
                cout<<"Mass flow rate = "<<massflowrate<<endl;

                m_pParticleData->m_vMassFlowRate+=massflowrate;

                double oldv=pelletvelocity[pi];
                double ave_volume = pelletvolumesum[pi]/pelletneighbor[pi];
                cout<<"average specific volume of neighbors = "<<ave_volume<<endl;
                pelletvelocity[pi]=massflowrate*ave_volume/4.0/3.1416/pr/pr;
                cout<<"pellet ablation velocity = "<<pelletvelocity[pi]<<endl;
		cout<<"average pressure of neighbors = "<<(83.1446/(20.1797*ave_volume))*(20 - (1.4 - 1.0)*pelletvelocity[pi]*pelletvelocity[pi]/(2*1.4*83.1446))<<endl;
		for(int i=0;i<n;i++)
		{
//			if(i%1000==0)
//			cout<<i<<endl;
			double tx=1,ty=1,tz=1,tr;
			while(tx*tx+ty*ty+tz*tz>1){
				tx=2*(double)rand()/(double)RAND_MAX-1;
				ty=2*(double)rand()/(double)RAND_MAX-1;
				tz=2*(double)rand()/(double)RAND_MAX-1;
			}
			tr=sqrt(tx*tx+ty*ty+tz*tz);
			tx=tx/tr,ty=ty/tr,tz=tz/tr;
			tr=newpir+(double)rand()/(double)RAND_MAX*(pir-newpir);
			x[inflowEndIndex]=pelletx[pi]+tx*tr;
			y[inflowEndIndex]=pellety[pi]+ty*tr;
			z[inflowEndIndex]=pelletz[pi]+tz*tr;
			vx[inflowEndIndex]=vy[inflowEndIndex]=vz[inflowEndIndex]=0;
			pressure[inflowEndIndex]= (83.1446/(20.1797*ave_volume))*(20 - (1.4 - 1.0)*pelletvelocity[pi]*pelletvelocity[pi]/(2*1.4*83.1446)); 
			volumeold[inflowEndIndex]=volume[inflowEndIndex]=ave_volume;
			localParSpacing[inflowEndIndex]=dx;
			mass[inflowEndIndex]=dx*dx*dx/sqrt(2.0)/ave_volume;
//			mass[inflowEndIndex]=dx*dx*dx/sqrt(2.0)/Vinflow*tr/pr*tr/pr;
			sound[inflowEndIndex]=m_pEOS->getSoundSpeed(pressure[inflowEndIndex],1./volume[inflowEndIndex]);
			pelletid[inflowEndIndex]=pi;
			inflowEndIndex++;
		}
//		cout<<"calculate velocity"<<endl;
		pelletir[pi]=newpir;
//		pelletvelocity[pi]=15;
		pelletir[pi]+=dt*0.5*(oldv+pelletvelocity[pi]);
	}	
//	cout<<"move particles"<<endl;

	for(size_t index=fluidEndIndex;index<inflowEndIndex;index++)
	{
		//move particle[index]
		double d_x=x[index]-pelletx[pelletid[index]];
		double d_y=y[index]-pellety[pelletid[index]];
		double d_z=z[index]-pelletz[pelletid[index]];
		double dr=sqrt(d_x*d_x+d_y*d_y+d_z*d_z);
		double oldv=sqrt(vx[index]*vx[index]+vy[index]*vy[index]+vz[index]*vz[index]);
		double newv=pelletvelocity[pelletid[index]];
		x[index]+=dt*0.5*(oldv+newv)*d_x/dr;
		y[index]+=dt*0.5*(oldv+newv)*d_y/dr;
		z[index]+=dt*0.5*(oldv+newv)*d_z/dr;
		vx[index]=newv*d_x/dr;
		vy[index]=newv*d_y/dr;
		vz[index]=newv*d_z/dr;
		dr+=dt*0.5*(oldv+newv);
//                mass[index]=dx*dx*dx/sqrt(2.0)/Vinflow*dr/pelletr[pelletid[index]]*dr/pelletr[pelletid[index]];
		if(dr>pelletr[pelletid[index]])//change label from inflow to fluid if r>pr
		{
//			mass[index]=dx*dx*dx/sqrt(2.0)/Vinflow;
			if(index>fluidEndIndex)
                                        m_pParticleData->swap(index,fluidEndIndex);
			fluidEndIndex++;
		}
	}

//	cout<<"inflow finished"<<endl;
        m_pParticleData->m_iFluidNum=fluidEndIndex-fluidStartIndex;
        m_pParticleData->m_iBoundaryStartIndex=fluidEndIndex;
        m_pParticleData->m_iBoundaryNum=m_pParticleData->m_iInflowNum=inflowEndIndex-fluidEndIndex;
        m_pParticleData->m_iGhostStartIndex=inflowEndIndex;
        m_pParticleData->m_iTotalNum=inflowEndIndex-fluidStartIndex;
	return 0;

}

PelletOutflowBoundary::PelletOutflowBoundary():xmin(-20),xmax(20),ymin(-20),ymax(20),zmin(-20),zmax(20) {
}

int PelletOutflowBoundary::UpdateInflowBoundary(ParticleData *m_pParticleData, EOS* m_pEOS, double dt, double dx) {
        size_t fluidStartIndex = m_pParticleData->m_iFluidStartIndex;
        size_t fluidEndIndex = m_pParticleData->m_iFluidStartIndex + m_pParticleData->m_iFluidNum;
        size_t inflowEndIndex = fluidEndIndex + m_pParticleData->m_iInflowNum;
        double *x = m_pParticleData->m_vPositionX;
        double *y = m_pParticleData->m_vPositionY;
        double *z = m_pParticleData->m_vPositionZ;
//        double *volume = m_pParticleData->m_vVolume;
//        double volumeforvacuum = 1e6;
        for(size_t index=fluidStartIndex;index<fluidEndIndex;index++)
        {
                if(x[index]<xmin || x[index]>xmax || y[index]<ymin || y[index]>ymax || z[index]<zmin || z[index]>zmax)
                {

                        if(index+1<fluidEndIndex)
                        {
                                m_pParticleData->swap(index,fluidEndIndex-1);
                        }
                        if(fluidEndIndex<inflowEndIndex)
                        {
                                m_pParticleData->swap(fluidEndIndex-1,inflowEndIndex-1);
                        }
                        fluidEndIndex--;
                        inflowEndIndex--;
                }
        }
        m_pParticleData->m_iFluidNum=fluidEndIndex-fluidStartIndex;
        m_pParticleData->m_iBoundaryStartIndex=fluidEndIndex;
        m_pParticleData->m_iGhostStartIndex=inflowEndIndex;
        m_pParticleData->m_iTotalNum=inflowEndIndex-fluidStartIndex;
        return 0;
}

