#include "geometry_random.h"
#include <iostream>
#include <cmath>

Uniform3D::Uniform3D():length(1.0), xCen(0), yCen(0), zCen(0) {}

bool Uniform3D::operator()(double x, double y, double z) const {
        return (x<xCen+length/2 && x>xCen-length/2 && y<yCen+length/2 && y>yCen-length/2 && z<zCen+length/2 && z>zCen-length/2);
}

void Uniform3D::getBoundingBox(double& xmin, double& xmax, double& ymin, double& ymax, double& zmin, double& zmax) {
        xmin = xCen-length/2;
        xmax = xCen+length/2;
        ymin = yCen-length/2;
        ymax = yCen+length/2;
        zmin = zCen-length/2;
        zmax = zCen+length/2;
}

void Uniform3D::randomlocation(double& x, double& y, double& z){
	x=xCen-length/2+length*((double)rand()/(double)RAND_MAX);
        y=yCen-length/2+length*((double)rand()/(double)RAND_MAX);
        z=zCen-length/2+length*((double)rand()/(double)RAND_MAX);
}

Gaussian3D::Gaussian3D():sigma(1),radius(3),xCen(0),yCen(0),zCen(0){}

bool Gaussian3D::operator()(double x, double y, double z) const {
        return ((x-xCen)*(x-xCen)+(y-yCen)*(y-yCen)+(z-zCen)*(z-zCen)<radius*radius);
}

void Gaussian3D::getBoundingBox(double& xmin, double& xmax, double& ymin, double& ymax, double& zmin, double& zmax) {
        xmin = xCen-radius;
        xmax = xCen+radius;
        ymin = yCen-radius;
        ymax = yCen+radius;
        zmin = zCen-radius;
        zmax = zCen+radius;
}

void Gaussian3D::randomlocation(double& x, double& y, double& z){

	double r1,r2,w=10;
	while(w>=1.0){
		r1=2.0*((double)rand()/(double)RAND_MAX)-1.0;
		r2=2.0*((double)rand()/(double)RAND_MAX)-1.0;
		w=r1*r1+r2*r2;
	}
	w=sqrt((-2.0*log(w))/w);
	x=r1*w*sigma+xCen;
	y=r2*w*sigma+yCen;
	w=10;
        while(w>=1.0){
                r1=2.0*((double)rand()/(double)RAND_MAX)-1.0;
                r2=2.0*((double)rand()/(double)RAND_MAX)-1.0;
                w=r1*r1+r2*r2;
        }
        w=sqrt((-2.0*log(w))/w);
        z=r1*w*sigma+zCen;
}

MultiGaussian3D::MultiGaussian3D():sigma(0.1),radius(3){xCen={0.,1.,-1.,0.5,-0.5},yCen={0.,0.5,-0.1,1,-1},zCen={0.,-1.,-0.1,0.5,1};}

bool MultiGaussian3D::operator()(double x, double y, double z) const {
        return ((x-xCen[0])*(x-xCen[0])+(y-yCen[0])*(y-yCen[0])+(z-zCen[0])*(z-zCen[0])<radius*radius);
}

void MultiGaussian3D::getBoundingBox(double& xmin, double& xmax, double& ymin, double& ymax, double& zmin, double& zmax) {
        xmin = xCen[0]-radius;
        xmax = xCen[0]+radius;
        ymin = yCen[0]-radius;
        ymax = yCen[0]+radius;
        zmin = zCen[0]-radius;
        zmax = zCen[0]+radius;
}

void MultiGaussian3D::randomlocation(double& x, double& y, double& z){

	int i=rand()%xCen.size();

        double r1,r2,w=10;
        while(w>=1.0){
                r1=2.0*((double)rand()/(double)RAND_MAX)-1.0;
                r2=2.0*((double)rand()/(double)RAND_MAX)-1.0;
                w=r1*r1+r2*r2;
        }
        w=sqrt((-2.0*log(w))/w);
        x=r1*w*sigma+xCen[i];
        y=r2*w*sigma+yCen[i];
        w=10;
        while(w>=1.0){
                r1=2.0*((double)rand()/(double)RAND_MAX)-1.0;
                r2=2.0*((double)rand()/(double)RAND_MAX)-1.0;
                w=r1*r1+r2*r2;
        }
        w=sqrt((-2.0*log(w))/w);
        z=r1*w*sigma+zCen[i];
}
