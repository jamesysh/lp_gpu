#include "geometry_shocktube3d.h"
#include <iostream>
#include <cmath>

////////////////////////////////////////////////////////////////////////////////////////
// Start of Shocktube3D
////////////////////////////////////////////////////////////////////////////////////////

Shocktube3D::Shocktube3D():lengthX(7), lengthY(2), lengthZ(2) {}

bool Shocktube3D::operator()(double x, double y, double z) const {

        return (x<=lengthX/2. && x>=-lengthX/2. && y<=lengthY/2. && y>=-lengthY/2. && z<=lengthZ/2. && z>=-lengthZ/2.);
}

void Shocktube3D::getBoundingBox(double& xmin, double& xmax, double& ymin, double& ymax, double& zmin, double& zmax) {
        xmin = -lengthX/2.;
        xmax = lengthX/2.;
        ymin = -lengthY/2.;
        ymax = lengthY/2.;
        zmin = -lengthZ/2.;
        zmax = lengthZ/2.;
}

Shocktube3D0025::Shocktube3D0025():lengthX(5), lengthY(0.49796), lengthZ(0.50157) {}

bool Shocktube3D0025::operator()(double x, double y, double z) const {

        return (x<=lengthX/2. && x>=-lengthX/2. && y<=lengthY/2. && y>=-lengthY/2. && z<=lengthZ/2. && z>=-lengthZ/2.);
}

void Shocktube3D0025::getBoundingBox(double& xmin, double& xmax, double& ymin, double& ymax, double& zmin, double& zmax) {
        xmin = -lengthX/2.;
        xmax = lengthX/2.;
        ymin = -lengthY/2.;
        ymax = lengthY/2.;
        zmin = -lengthZ/2.;
        zmax = lengthZ/2.;
}

Shocktube3D0005::Shocktube3D0005():lengthX(5), lengthY(0.099593), lengthZ(0.100315) {}

bool Shocktube3D0005::operator()(double x, double y, double z) const {

        return (x<=lengthX/2. && x>=-lengthX/2. && y<=lengthY/2. && y>=-lengthY/2. && z<=lengthZ/2. && z>=-lengthZ/2.);
}

void Shocktube3D0005::getBoundingBox(double& xmin, double& xmax, double& ymin, double& ymax, double& zmin, double& zmax) {
        xmin = -lengthX/2.;
        xmax = lengthX/2.;
        ymin = -lengthY/2.;
        ymax = lengthY/2.;
        zmin = -lengthZ/2.;
        zmax = lengthZ/2.;
}

RayleighTaylor3D::RayleighTaylor3D():lengthX(1), lengthY(1), lengthZ(1) {}

bool RayleighTaylor3D::operator()(double x, double y, double z) const {

        return (x<=lengthX/2. && x>=-lengthX/2. && y<=lengthY/2. && y>=-lengthY/2. && z<=lengthZ/2. && z>=-lengthZ/2.);
}

void RayleighTaylor3D::getBoundingBox(double& xmin, double& xmax, double& ymin, double& ymax, double& zmin, double& zmax) {
        xmin = -lengthX/2.;
        xmax = lengthX/2.;
        ymin = -lengthY/2.;
        ymax = lengthY/2.;
        zmin = -lengthZ/2.;
        zmax = lengthZ/2.;
}
