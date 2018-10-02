#ifndef __GEOMETRY_RANDOM_H__
#define __GEOMETRY_RANDOM_H__

#include "geometry.h"
#include <vector>

class Uniform3D: public Geometry {
public:
	Uniform3D();
	virtual ~Uniform3D(){}
        virtual bool operator()(double x, double y, double z) const;
        virtual void getBoundingBox(double& xmin, double& xmax, double& ymin, double& ymax, double& zmin, double& zmax);
	virtual void randomlocation(double& x, double& y, double& z);
private:
	double length,xCen,yCen,zCen;
};

class Gaussian3D: public Geometry {
public:
        Gaussian3D();
        virtual ~Gaussian3D(){}
        virtual bool operator()(double x, double y, double z) const;
        virtual void getBoundingBox(double& xmin, double& xmax, double& ymin, double& ymax, double& zmin, double& zmax);
        virtual void randomlocation(double& x, double& y, double& z);
private:
        double sigma,radius,xCen,yCen,zCen;
};

class MultiGaussian3D: public Geometry {
public:
        MultiGaussian3D();
        virtual ~MultiGaussian3D(){}
        virtual bool operator()(double x, double y, double z) const;
        virtual void getBoundingBox(double& xmin, double& xmax, double& ymin, double& ymax, double& zmin, double& zmax);
        virtual void randomlocation(double& x, double& y, double& z);
private:
        double sigma,radius;
	std::vector<double> xCen,yCen,zCen;
};
#endif
