#ifndef __BOUNDARY_PELLET_H__
#define __BOUNDARY_PELLET_H__

#include "boundary.h"
#include "eos.h"
#include "particle_data.h"
#include <vector>

class PelletInflowBoundary: public Boundary {
public:
	PelletInflowBoundary();
	virtual ~PelletInflowBoundary() {};
	virtual int UpdateInflowBoundary(ParticleData* ParticleData, EOS* m_pEOS, double dt, double m_fInitParticleSpacing);
	virtual int operator()(double x, double y, double z, double pressure, double vx, double vy, double vz, std::vector<double>& xb, std::vector<double>& vb, std::vector<double>& zb, std::vector<double>& pressureb, std::vector<double>& vxb, std::vector<double>& vyb, std::vector<double>& vzb){return 0;};
private:
	double Pinflow;//inflow pressure, constant
	double Uinflow;//inflow velocity, calculated using energy absorb rate
	double Vinflow;//inflow specific volume, constant
};

class PelletOutflowBoundary: public Boundary {
public:
        PelletOutflowBoundary();
        virtual ~PelletOutflowBoundary() {};
        virtual int UpdateInflowBoundary(ParticleData *ParticleData, EOS* m_pEOS, double dt, double m_fInitParticleSpacing);
        virtual int operator()(double x, double y, double z, double pressure, double vx, double vy, double vz,
        std::vector<double>& xb, std::vector<double>& yb, std::vector<double>& zb,
        std::vector<double>& pressureb, std::vector<double>& vxb, std::vector<double>& vyb, std::vector<double>& vzb){return 0;};
private:
        double xmin,xmax,ymin,ymax,zmin,zmax;
};

#endif
