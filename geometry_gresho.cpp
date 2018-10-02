#include "geometry_gresho.h"
#include <iostream>
#include <cmath>


////////////////////////////////////////////////////////////////////////////////////////
// Start of Gresho2D
////////////////////////////////////////////////////////////////////////////////////////

Gresho2D::Gresho2D():radius(1.), xCen(0), yCen(0) {}

bool Gresho2D::operator()(double x, double y, double z) const {	
	return ((x-xCen)*(x-xCen)+(y-yCen)*(y-yCen)<radius*radius); 	
}

void Gresho2D::getBoundingBox(double& xmin, double& xmax, double& ymin, double& ymax, double& zmin, double& zmax) {
	xmin = xCen-radius;
	xmax = xCen+radius;
	ymin = yCen-radius;
	ymax = yCen+radius;
	zmin = 0;
	zmax = 0;
}

////////////////////////////////////////////////////////////////////////////////////////
// End of Gresho2D
////////////////////////////////////////////////////////////////////////////////////////



Yee2D::Yee2D():radius(5.), xCen(0), yCen(0) {}

bool Yee2D::operator()(double x, double y, double z) const {
        return ((x-xCen)*(x-xCen)+(y-yCen)*(y-yCen)<radius*radius);
}

void Yee2D::getBoundingBox(double& xmin, double& xmax, double& ymin, double& ymax, double& zmin, double& zmax) {
        xmin = xCen-radius;
        xmax = xCen+radius;
        ymin = yCen-radius;
        ymax = yCen+radius;
        zmin = 0;
        zmax = 0;
}

void Yee2D::randomlocation(double& x, double& y, double& z){
	z=1;
	while(z>0 || (x-xCen)*(x-xCen)+(y-yCen)*(y-yCen)>radius*radius){
	        x=xCen-radius+2*radius*((double)rand()/(double)RAND_MAX);
        	y=yCen-radius+2*radius*((double)rand()/(double)RAND_MAX);
        	z=0;
	}
}

Sedov2D::Sedov2D():radius(1.), xCen(0), yCen(0) {}

bool Sedov2D::operator()(double x, double y, double z) const {
        return ((x-xCen)*(x-xCen)+(y-yCen)*(y-yCen)<radius*radius);
}

void Sedov2D::getBoundingBox(double& xmin, double& xmax, double& ymin, double& ymax, double& zmin, double& zmax) {
        xmin = xCen-radius-0.0025;
        xmax = xCen+radius-0.0025;
        ymin = yCen-radius-0.00026;
        ymax = yCen+radius-0.00026;
        zmin = 0;
        zmax = 0;
}

Yee3D::Yee3D():radius(5.), xCen(0), yCen(0), zlength(7.) {}

bool Yee3D::operator()(double x, double y, double z) const {
        return ((x-xCen)*(x-xCen)+(y-yCen)*(y-yCen)<radius*radius);
}

void Yee3D::getBoundingBox(double& xmin, double& xmax, double& ymin, double& ymax, double& zmin, double& zmax) {
        xmin = xCen-radius;
        xmax = xCen+radius;
        ymin = yCen-radius;
        ymax = yCen+radius;
        zmin = -zlength/2;
        zmax = zlength/2;
}
