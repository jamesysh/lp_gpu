/**
 * \file   initializer.h
 *
 * \brief  This header file contains classes for the initialization of the simulation
 *
 * The task of initialization of the simulation includes:\n 
 * 1. Reading the input file \n
 * 2. Initializing the geometry of fluid objects\n
 * 3. Initializing the state of fluid objects\n
 * 4. Specifying some pre-determined parameters and calculate some parameters
 *
 * \author Chen, Hsin-Chiang (morrischen2008@gmail.com) 
 *
 * \version 1.0 
 *
 * \date 2014/06/09
 *
 * Created on: 2014/06/01 
 *
 */


#ifndef __INITIALIZER_H__
#define __INITIALIZER_H__

#include <cstddef>
#include <fstream>
#include <vector>
#include <map>

class EOS;
class Geometry;
class State;
class Boundary;

/**
 * \class BoundingBox
 * 
 * \brief This class is keeps the information of the boundaries of a fluid/boundary object
 *
 * The bounding box can be updated by the setter member methods
 *
 * \author Chen, Hsin-Chiang (morrischen2008@gmail.com)
 *
 * \version 1.0 
 *
 * \date 2014/06/09 
 *
 * Created on: 2014/06/01 
 *
 */
class BoundingBox {
public:
	/**
	 * \brief            Constructor
	 *
	 * Initializes the bounding box using the parameters in the argument list
	 *
	 * \param [in] xmin  The minimum value in the x-coordinate
 	 * \param [in] xmax  The maximum value in the x-coordinate
	 * \param [in] ymin  The minimum value in the y-coordinate
	 * \param [in] ymax  The maximum value in the y-coordinate
	 * \param [in] zmin  The minimum value in the z-coordinate
	 * \param [in] zmax  The maximum value in the z-coordinate
	 *
	 */
	BoundingBox(double xmin, double xmax, double ymin, double ymax, double zmin, double zmax): 
		m_fXmin(xmin), m_fXmax(xmax), m_fYmin(ymin), m_fYmax(ymax), m_fZmin(zmin), m_fZmax(zmax),
		m_iStartIndex(0), m_iNumber(0) {}
	
	/**
	 * \brief   Getter function of the minimum value in x-coordinate of the bounding box
	 * \param   None
	 * \return  The minimum value in x-coordinate of the bounding box
	 */
	double getXmin() {return m_fXmin;}
	
	/**
	 * \brief   Getter function of the maximum value in x-coordinate of the bounding box
	 * \param   None
	 * \return  The maximum value in x-coordinate of the bounding box
	 */
	double getXmax() {return m_fXmax;}
	
	/**
	 * \brief   Getter function of the minimum value in y-coordinate of the bounding box
	 * \param   None
	 * \return  The minimum value in y-coordinate of the bounding box
	 */
	double getYmin() {return m_fYmin;}
	
	/**
	 * \brief   Getter function of the maximum value in y-coordinate of the bounding box
	 * \param   None
	 * \return  The maximum value in y-coordinate of the bounding box
	 */
	double getYmax() {return m_fYmax;}
	
	/**
	 * \brief   Getter function of the minimum value in z-coordinate of the bounding box
	 * \param   None
	 * \return  The minimum value in z-coordinate of the bounding box
	 */
	double getZmin() {return m_fZmin;}
	
	/**
	 * \brief   Getter function of the maximum value in z-coordinate of the bounding box
	 * \param   None
	 * \return  The maximum value in z-coordinate of the bounding box
	 */
	double getZmax() {return m_fZmax;}
	
	/**
	 * \brief   Getter function of the start index in the particle arrays 
	 *          of the fluid/boundary object inside this bounding box 
	 * \param   None
	 * \return  The start index in the particle arrays of the fluid/boundary object inside this bounding box
	 * \note    The particle arrays refer to the major arrays like the x, y, z-coordinates 
	 *          of the ParticleData class
	 */
	size_t getStartIndex() {return m_iStartIndex;}
	
	/**
	 * \brief   Getter function of the number of particles  
	 *          of the fluid/boundary object inside this bounding box 
	 * \param   None
	 * \return  The number of particles of the fluid/boundary object inside this bounding box
	 */
	size_t getNumber() {return m_iNumber;}
	

	/**
	 * \brief   Getter function of the tag of this object  
	 *          
	 * \param   None
	 * \return  The tag
	 */
	size_t getObjectTag() {return m_iObjectTag;}


	/**
	 * \brief        Setter function of the minimum value in x-coordinate of the bounding box
	 * \param   xmin The minimum value in x-coordinate of the bounding box
	 * \return       None
	 */
	void setXmin(double xmin) {m_fXmin=xmin;}
	
	/**
	 * \brief        Setter function of the maximum value in x-coordinate of the bounding box
	 * \param   xmax The maximum value in x-coordinate of the bounding box
	 * \return       None
	 */
	void setXmax(double xmax) {m_fXmax=xmax;}
	
	/**
	 * \brief        Setter function of the minimum value in y-coordinate of the bounding box
	 * \param   ymin The minimum value in y-coordinate of the bounding box
	 * \return       None
	 */
	void setYmin(double ymin) {m_fYmin=ymin;}
	
	/**
	 * \brief        Setter function of the maximum value in y-coordinate of the bounding box
	 * \param   ymax The maximum value in y-coordinate of the bounding box
	 * \return       None
	 */
	void setYmax(double ymax) {m_fYmax=ymax;}
	
	/**
	 * \brief        Setter function of the minimum value in z-coordinate of the bounding box
	 * \param   zmin The minimum value in z-coordinate of the bounding box
	 * \return       None
	 */
	void setZmin(double zmin) {m_fZmin=zmin;}
	
	/**
	 * \brief        Setter function of the maximum value in z-coordinate of the bounding box
	 * \param   zmax The maximum value in z-coordinate of the bounding box
	 * \return       None
	 */
	void setZmax(double zmax) {m_fZmax=zmax;}
	
	/**
	 * \brief   Setter function of the start index in the particle arrays 
	 *          of the fluid/boundary object inside this bounding box 
	 * \param   The start index in the particle arrays of the fluid/boundary object inside this bounding box
	 * \return  None 
	 * \note    The particle arrays refer to the major arrays like the x, y, z-coordinates 
	 *          of the ParticleData class         
	 */
	void setStartIndex(size_t index) {m_iStartIndex=index;}
	
	/**
	 * \brief   Setter function of the number of particles of the fluid/boundary object inside this bounding box 
	 * \param   The number of particles of the fluid/boundary object inside this bounding box
	 * \return  None 
	 *          
	 */
	void setNumber(size_t num) {m_iNumber=num;}

	/**
	 * \brief   Setter function of the tag of this object 
	 * \param   The tag
	 * \return  None 
	 *          
	 */
	void setObjectTag(int tag) {m_iObjectTag=tag;}

private:
	double m_fXmin, m_fXmax, m_fYmin, m_fYmax, m_fZmin, m_fZmax;
	size_t m_iStartIndex;
	size_t m_iNumber;
	int m_iObjectTag;
};










/**
 * \class Initializer
 * 
 * \brief This class initializes the simulation  
 *
 * 
 * The task of initialization of the simulation includes:\n 
 * 1. Reading the input file \n
 * 2. Initializing the geometry of fluid objects\n
 * 3. Initializing the state of fluid objects\n
 * 4. Specifying some pre-determined parameters and calculate some parameters
 *
 *
 * \author Chen, Hsin-Chiang (morrischen2008@gmail.com)
 *
 * \version 1.0 
 *
 * \date 2014/06/09 
 *
 * Created on: 2014/06/01 
 *
 */
class Initializer {
public:
	/**
	 * \brief Constructor 
	 * 
	 * 1. Reads the input file\n 
	 * 2. Initializes the geometry of fluid objects\n
     * 3. Initializes the state of fluid objects\n
     * 4. Specifies some pre-determined parameters and calculate some parameters 
	 */	
	Initializer(const std::string& inputfileName, bool ifDebug=false, const std::string& debugfileName="debug");	
	
	/**
	 * \brief Constructor for restart 
	 * 
	 * 1. Reads the parameter file and data file\n 
	 * 2. Initializes the geometry of fluid objects\n
     * 3. Initializes the state of fluid objects 
	 */	
	Initializer(const std::string& param_filename, const std::string& data_filename, 
	bool ifDebug=false, const std::string& debugfileName="debug");	
	
	/**
	 * \brief Destructor 
	 *
	 */
	~Initializer();
	
	/**
	 * \brief   Getter function of the boolean value on whether to print debug information
	 * \param   None
	 * \return  The boolean value on whether to print debug information
	 */
	bool getIfDebug() const {return m_iIfDebug;}
	
	/**
	 * \brief   Getter function of the boolean value on whether a restart run
	 * \param   None
	 * \return  The boolean value on whether a restart run
	 */
	bool getIfRestart() const {return m_iIfRestart;}

	/**
	 * \brief   Getter function of the filename for printing debug information 
	 * \param   None
	 * \return  The filename for printing debug information
	 */
	std::string getDebugfileName() const {return m_sDebugfileName;}

	/**
	 * \brief   Getter function of the specified number of threads 
	 * \param   None
	 * \return  The specified number of threads
	 */
	int getNumThreads() const {return m_iNumThreads;}
	
	/**
	 * \brief   Getter function of the specified simulation physical start time 
	 * \param   None
	 * \return  The specified simulation physical start time specified
	 */
	double getStartTime() const {return m_fStartTime;}
	
	/**
	 * \brief   Getter function of the specified simulation physical end time 
	 * \param   None
	 * \return  The specified simulation physical end time
	 */
	double getEndTime() const {return m_fEndTime;}
	
	/**
	 * \brief   Getter function of the specified physical time interval between writes 
	 * \param   None
	 * \return  The specified physical time interval between writes
	 */
	double getWriteTimeInterval() const {return m_fWriteTimeInterval;}
	
	/**
	 * \brief   Getter function of the number of times results are written to the particle veiwer 
	 * \param   None
	 * \return  The number of times results are written to the particle veiwer
	 */
	double getWriteStep() const {return m_iWriteStep;}


	/**
	 * \brief   Getter function of the specified CFL coefficient 
	 * \param   None
	 * \return  The specified CFL coefficient
	 */
	double getCFLCoeff() const {return m_fCFLCoeff;}
	
	/**
	 * \brief   Getter function of the specified dimension of simulation 
	 * \param   None
	 * \return  The specified dimension of simulation
	 */
	int getDimension() const {return m_iDimension;}	
	
	/**
	 * \brief   Getter function of the specified boolean value on whether to use random order of Strang splitting or not
	 * \param   None
	 * \return  The specified boolean value on whether to use random order of Strang splitting or not
	 */
	bool getRandomDirSplitOrder() const {return m_iRandomDirSplitOrder;}
	
	/**
	 * \brief   Getter function of the specified order of Local Polynomial Fitting 
	 * \param   None
	 * \return  The specified order of Local Polynomial Fitting
	 */
	int getLPFOrder() const {return m_iLPFOrder;}	
	
	/**
	 * \brief   Getter function of the specified choice of equation of state (eos) 
	 * \param   None
	 * \return  The specified specified choice of equation of state (eos)
	 */
	int getEOSChoice() const {return m_iEOSChoice;} //TODO
	
	/**
	 * \brief   Getter function of the specified initial inter-particle spacing 
	 * \param   None
	 * \return  The specified initial inter-particle spacing
	 */
	double getInitParticleSpacing() const {return m_fInitParticleSpacing;}
	
	/**
	 * \brief   Getter function of the specified gravity value 
	 * \param   None
	 * \return  The specified gravity value
	 */
	double getGravity() const {return m_fGravity;}
	
	/**
	 * \brief   Getter function of the specified boolean value on whether to use limiter or not 
	 * \param   None
	 * \return  The specified boolean value on whether to use limiter or not
	 * \note    We only have 1D limiter now
	 */
	bool getUseLimiter() const {return m_iUseLimiter;}

        int getDensityEstimatorType() const {return m_iDensityEstimatorType;}
        int getInitialPerturbation() const {return m_fInitialPerturbation;}
        bool getFixParticles() const {return m_iFixParticles;}
        bool getOutputError() const {return m_iOutputError;}
        std::string getExactSolutionName() const {return m_sExactSolutionName;}
	std::string getAuxiliaryOutput() const {return m_sAuxiliaryOutput;}
	int getPelletDistribution() const {return m_iPelletDistribution;}
	int getNumberofPellet() const {return m_iNumberofPellet;}
	int getMaxParticlePerCell() const {return m_iMaxParticlePerCell;}
	double* getPelletPositionX() const {return m_vPelletPositionX;}
        double* getPelletPositionY() const {return m_vPelletPositionY;}
        double* getPelletPositionZ() const {return m_vPelletPositionZ;}
        double* getPelletRadius() const {return m_vPelletRadius;}
	double* getPelletInnerRadius() const {return m_vPelletInnerRadius;}
        double getMagx() {return Magx;}
        double getMagy() {return Magy;}
        double getMagz() {return Magz;}
        double getmasse() {return masse;}
        double getmassNe() {return massNe;}
        double getteinf() {return teinf;}
        double getINe() {return INe;}
        int getZNe() {return ZNe;}
        double getneinf() {return neinf;}
        double getheatK() {return heatK;}
        double getconductivity() {return conductivity;}
        double getsublimationenergy() {return sublimationenergy;}

	
	
	/**
	 * \brief   Getter function of the specified maximum number of neighbours of a particle  
	 * \param   None
	 * \return  The specified maximum number of neighbours of a particle 
	 */
	std::size_t getMaxNeighbourNum() const {return m_iMaxNeighbourNum;}
	
	/**
	 * \brief   Getter function of the specified maximum number of neighbours in one direction 
	 *          (eg. right-hand-side in the x-coordinate) of a particle  
	 * \param   None
	 * \return  The specified maximum number of neighbours in one direction 
	 *          (eg. right-hand-side in the x-coordinate) of a particle
	 */
	std::size_t getMaxNeighbourNumInOneDir() const {return m_iMaxNeighbourNumInOneDir;}
	
	/**
	 * \brief   Getter function of the specified minimum number of neighbours in one direction of a particle  
	 *          for using 2nd order local polynomial fitting to compute the one-sided spatial derivatives
	 * \param   None
	 * \return  The specified minimum number of neighbours in one direction of a particle  
	 *          for using 2nd order local polynomial fitting to compute the one-sided spatial derivatives
	 */
	std::size_t getNumRow2ndOrder() const {return m_iNumRow2ndOrder;}
	
	/**
	 * \brief   Getter function of the specified minimum number of neighbours in one direction of a particle  
	 *          for using 1st order local polynomial fitting to compute the one-sided spatial derivatives
	 * \param   None
	 * \return  The specified minimum number of neighbours in one direction of a particle  
	 *          for using 1st order local polynomial fitting to compute the one-sided spatial derivatives
	 */
	std::size_t getNumRow1stOrder() const {return m_iNumRow1stOrder;}
	
	/**
	 * \brief   Getter function of the theoretical minimum number of neighbours required 
	 *          in one direction of a particle for 2nd order local polynomial fitting for 
	 *          the computation of one-sided spatial derivatives
	 * \param   None
	 * \return  The theoretical minimum number of neighbours required 
	 *          in one direction of a particle for using 2nd order local polynomial fitting 
	 *          to compute the one-sided spatial derivatives
	 */
	std::size_t getNumCol2ndOrder() const {return m_iNumCol2ndOrder;}
	
	/**
	 * \brief   Getter function of the theoretical minimum number of neighbours required 
	 *          in one direction of a particle for using 1st order local polynomial fitting  
	 *          to compute the one-sided spatial derivatives  
	 * \param   None
	 * \return  The theoretical minimum number of neighbours required 
	 *          in one direction of a particle for using 1st order local polynomial fitting  
	 *          to compute the one-sided spatial derivatives
	 */
	std::size_t getNumCol1stOrder() const {return m_iNumCol1stOrder;}
	
	/**
	 * \brief   Getter function of the specified radius for neighbour search  
	 * \param   None
	 * \return  The specified radius for neighbour search
	 */
	double getNeiSearchRadius() const {return m_fNeiSearchRadius;}
	
	/**
	 * \brief   Getter function of the specified maximum/minimum pressure value a particle can possibly attain
	 *  
	 * \param   None
	 * \return  The specified maximum/minimum pressure value a particle can possibly attain
	 * \note    If this value is exceeded certain actions, such as re-calculating spatial derivatives using
	 *          a lower order of local polynomial fitting, needs to be carried to avoid the crash of the program
	 */
	double getInvalidPressure() const {return m_fInvalidPressure;}
	
	/**
	 * \brief   Getter function of the specified maximum/minimum volume value a particle can possibly attain 
	 * \param   None
	 * \return  The specified maximum/minimum volume value a particle can possibly attain
	 * \note    If this value is exceeded certain actions, such as re-calculating spatial derivatives using
	 *          a lower order of local polynomial fitting, needs to be carried to avoid the crash of the program
	 */
	double getInvalidDensity() const {return m_fInvalidDensity;}
	
	/**
	 * \brief   Getter function of how many times is the neighbour search radius wrt 
	 *          inter-particle spacing
	 *  
	 * \param   None
	 * \return  times
	 */
	double getTimesNeiSearchRadius() const {return m_fTimesNeiSearchRadius;}
	
	/**
	 * \brief   Getter function of how many times is the contact length wrt 
	 *          inter-particle spacing
	 *  
	 * \param   None
	 * \return  times
	 */
	double getTimesContactLength() const {return m_fTimesContactLength;}

	/**
	 * \brief   Getter function of times of buffer zone wrt inter-particle spacing 
	 *          
	 * \param   None
	 * \return  times
	 */
	double getTimesBoundingBox() const {return m_fTimesBoundingBox;}


	/**
	 * \brief   Getter function of the specified tree depth for the octree neighbour search algorithm  
	 * \param   None
	 * \return  The specified tree depth for the octree neighbour search algorithm
	 */
	int getTreeDepth() const {return m_iTreeDepth;}
	
	/**
	 * \brief   Getter function of the specified length such that if the distance between two particles
	 *          coming from different fluid objects is less than this length, they are considered as
	 *          "in contact"
	 * \param   None
	 * \return  The specified length such that if the distance between two particles
	 *          coming from different fluid objects is less than this length, they are considered as
	 *          "in contact"
	 *
	 * \note    The length is only valid in simulations with more than one fluid object. 
	 *          If two particles are in contact, they can start to take neighbours from 
	 *          the other fluid object. 
	 *          The length is specified as a number less than the initial inter-particle spacing
	 */
	double getContactLength() const {return m_fContactLength;}
	
	/**
	 * \brief   Getter function of the specified array length for particle data arrays 
	 *          such as x, y, and z-coordinates, pressure, volume, and sound speeds    
	 * \param   None
	 * \return  The specified array length for particle data arrays 
	 *          such as x, y, and z-coordinates, pressure, volume, and sound speeds 
	 * \note    Capacity is a number larger than the total number of \e all types of particles
	 */
	std::size_t getCapacity() const {return m_iCapacity;}
	
	/**
	 * \brief   Getter function of the number of \e fluid particles initialized 
	 * \param   None
	 * \return  The number of \e fluid particles initialized 
	 */
	std::size_t getFluidNum() const {return m_iFluidNum;}
	
	/**
	 * \brief   Getter function of the  number of \e boundary particles initialized
	 * \param   None
	 * \return  The number of \e boundary particles initialized
	 */
//	std::size_t getBoundaryNum() const {return m_iBoundaryNum;}
	
	/**
	 * \brief   Getter function of the start index of fluid particles in the particle data arrays  
	 * \param   None
	 * \return  The start index of fluid particles in the particle data arrays
	 */
	std::size_t getFluidStartIndex() const {return m_iFluidStartIndex;}
	
	/**
	 * \brief   Getter function of the start index of boundary particles in the particle data arrays  
	 * \param   None
	 * \return  The start index of boundary particles in the partical data arrays
	 */
//	std::size_t getBoundaryStartIndex() const {return m_iBoundaryStartIndex;}
	
	/**
	 * \brief   Getter function of the start index of ghost particles in the particle arrays of the ParticleData class  
	 * \param   None
	 * \return  The start index of ghost particles in the particle arrays
	 */
//	std::size_t getGhostStartIndex() const {return m_iGhostStartIndex;}
	
	/**
	 * \brief   Getter function of the number of particles located inside the specified radius of 
	 *          neighbour search at the time of initialization 
	 * \param   None
	 * \return  The number of particles located inside the specified radius of 
	 *          neighbour search at the time of initialization
	 * \note    This number is used in the context of ghost particle selection
	 */
	std::size_t getNumParticleWithinSearchRadius() const {return m_iNumParticleWithinSearchRadius;}	


	/**
	 * \brief   Getter function of the array of x-coordinates of the initialized particles  
	 * \param   None
	 * \return  Pointer pointing to the array of x-coordinates of the initialized particles
	 * \warning The array of x-coordinate is initialized on the heap in this class, 
	 *          and will be not deleted in the destructor of this class. 
	 *          The reason is that the ownership of this array will be 
	 *          transfered to the ParticleData class via this getter function. 
	 *          It is advised not to assign the ownership of this array to other classes
	 */
	double* getPositionX() const {return m_vPositionX;}
	
	/**
	 * \brief   Getter function of the array of y-coordinates of the initialized particles  
	 * \param   None
	 * \return  Pointer pointing to the array of y-coordinates of the initialized particles
	 * \warning The array of x-coordinate is initialized on the heap in this class, 
	 *          and will be not deleted in the destructor of this class. 
	 *          The reason is that the ownership of this array will be 
	 *          transfered to the ParticleData class via this getter function. 
	 *          It is advised not to assign the ownership of this array to other classes
	 */
	double* getPositionY() const {return m_vPositionY;}
	
	/**
	 * \brief   Getter function of the array of z-coordinates of the initialized particles  
	 * \param   None
	 * \return  Pointer pointing to the array of z-coordinates of the initialized particles
	 * \warning The array of x-coordinate is initialized on the heap in this class, 
	 *          and will be not deleted in the destructor of this class. 
	 *          The reason is that the ownership of this array will be 
	 *          transfered to the ParticleData class via this getter function. 
	 *          It is advised not to assign the ownership of this array to other classes
	 */
	double* getPositionZ() const {return m_vPositionZ;}
	
	/**
	 * \brief   Getter function of the array of velocity in x-coordinate of the initialized particles  
	 * \param   None
	 * \return  Pointer pointing to the array of velocity in x-coordinate of the initialized particles
	 * \warning The array of velocity in x-coordinate is initialized on the heap in this class, 
	 *          and will be not deleted in the destructor of this class. 
	 *          The reason is that the ownership of this array will be 
	 *          transfered to the ParticleData class via this getter function. 
	 *          It is advised not to assign the ownership of this array to other classes
	 */
	double* getVelocityU() const {return m_vVelocityU;}
	
	/**
	 * \brief   Getter function of the array of velocity in y-coordinate of the initialized particles  
	 * \param   None
	 * \return  Pointer pointing to the array of velocity in y-coordinate of the initialized particles
	 * \warning The array of velocity in x-coordinate is initialized on the heap in this class, 
	 *          and will be not deleted in the destructor of this class. 
	 *          The reason is that the ownership of this array will be 
	 *          transfered to the ParticleData class via this getter function. 
	 *          It is advised not to assign the ownership of this array to other classes
	 */
	double* getVelocityV() const {return m_vVelocityV;}
	
	/**
	 * \brief   Getter function of the array of velocity in z-coordinate of the initialized particles  
	 * \param   None
	 * \return  Pointer pointing to the array of velocity in z-coordinate of the initialized particles
	 * \warning The array of velocity in x-coordinate is initialized on the heap in this class, 
	 *          and will be not deleted in the destructor of this class. 
	 *          The reason is that the ownership of this array will be 
	 *          transfered to the ParticleData class via this getter function. 
	 *          It is advised not to assign the ownership of this array to other classes
	 */
	double* getVelocityW() const {return m_vVelocityW;}
	
	/**
	 * \brief   Getter function of the array of volume of the initialized particles  
	 * \param   None
	 * \return  Pointer pointing to the array of volume of the initialized particles
	 * \warning The array of volume is initialized on the heap in this class, 
	 *          and will be not deleted in the destructor of this class. 
	 *          The reason is that the ownership of this array will be 
	 *          transfered to the ParticleData class via this getter function. 
	 *          It is advised not to assign the ownership of this array to other classes
	 */
	double* getVolume() const {return m_vVolume;}
	double* getMass() const {return m_vMass;}

	/**
	 * \brief   Getter function of the array of pressure of the initialized particles  
	 * \param   None
	 * \return  Pointer pointing to the array of pressure of the initialized particles
	 * \warning The array of pressure is initialized on the heap in this class, 
	 *          and will be not deleted in the destructor of this class. 
	 *          The reason is that the ownership of this array will be 
	 *          transfered to the ParticleData class via this getter function. 
	 *          It is advised not to assign the ownership of this array to other classes
	 */
	double* getPressure() const {return m_vPressure;}
	
	/**
	 * \brief   Getter function of the array of sound speed of the initialized particles  
	 * \param   None
	 * \return  Pointer pointing to the array of sound speed of the initialized particles
	 * \warning The array of volume is initialized on the heap in this class, 
	 *          and will be not deleted in the destructor of this class. 
	 *          The reason is that the ownership of this array will be 
	 *          transfered to the ParticleData class via this getter function. 
	 *          It is advised not to assign the ownership of this array to other classes
	 */
	double* getSoundSpeed() const {return m_vSoundSpeed;}
	
	/**
	 * \brief   Getter function of the array of local inter-particle spacing 
	 * \param   None
	 * \return  Pointer pointing to the array of local inter-particle spacing
	 * \warning The array of volume is initialized on the heap in this class, 
	 *          and will be not deleted in the destructor of this class. 
	 *          The reason is that the ownership of this array will be 
	 *          transfered to the ParticleData class via this getter function. 
	 *          It is advised not to assign the ownership of this array to other classes
	 */
	double* getLocalParSpacing() const {return m_vLocalParSpacing;}

	/**
	 * \brief   Getter function of the array of "object tags" of the initialized particles  
	 * \param   None
	 * \return  Pointer pointing to the array of "object tags" of the initialized particles
	 * \note    The tag of a particle in a fluid object is the index of the fluid object 
	 *          (the index of fluid object is 1,2,...). 
	 *          The tag of a particle \b not in fluid object is 0
	 */
	int* getObjectTag() const {return m_vObjectTag;} 
	
	/**
	 * \brief   Getter function of the pointer to an object in the EOS family  
	 * \param   None
	 * \return  Pointer pointing to an object in the EOS family
	 * \warning The memory of the object pointed to by the pointer is on the heap.
	 *          This memory will not be deleted but the ownership will be transfered
	 *          to other classes. 
	 *          It is save to have multiple pointers to this object since this object 
	 *          does not modify its own data
	 */
	EOS* getEOS() const {return m_pEOS;}
	
	/**
	 * \brief   Getter function of the bounding boxes for the initialized fluid objects  
	 * \param   None
	 * \return  Vector of pointers to BoundingBox type 
	 */
	std::vector<BoundingBox*>& getFluidBoundingBox() {return m_vFluidBoundingBox;}	
	
	/**
	 * \brief   Getter function of the types of the initialized boundary objects  
	 * \param   None
	 * \return  Vector of strings of boundary object types
	 */
	std::vector<std::string>& getBoundaryObjTypes() {return m_vBoundaryObjTypes;}

	/**
	 * \brief   Getter function of the initialized boundary objects  
	 * \param   None
	 * \return  Vector of strings of boundary objects
	 */
	std::vector<Boundary*>& getBoundaryObj() {return m_vBoundaryObj;}
private:
	
	//--------------------------Data from arg list--------------------------------
	bool m_iIfDebug; ///< if true then print debug info
	const std::string m_sDebugfileName; ///< filename for printing debug info
	

	//----------------------------------------------------------------------------

	//--------------------------Data from the inputfile---------------------------	
	
	int m_iNumThreads;///< Number of threads 
	double m_fStartTime;///< simulation start time
	double m_fEndTime; ///< simulation end time	
	double m_fWriteTimeInterval;///< write time interval
	std::size_t m_iWriteStep; ///< write step
	double m_fCFLCoeff;///< CFL coeff
	int m_iDimension;///< dimension
	int m_iFluidObjNum;///< number of fluid objects		
	int m_iBoundaryObjNum;///< number of boundary objects
	bool m_iRandomDirSplitOrder;///< if true then the order of directional splitting is randomly set 1:yes 0:no	
	int m_iLPFOrder;///< the order of Local Polynomial Fitting (LPF)  	
	int m_iEOSChoice;///< choice of eos
	double m_fGamma;///< eos parameter gamma
	double m_fPinf;///< eos parameter pinf (stiffened poly gas) 
	double m_fEinf;///<eos parameter einf (stiffened poly gas) 	
	double m_fInitParticleSpacing;///< the initial particle spacing	
	double m_fGravity;///< gravity
	bool m_iUseLimiter;///< if use limiter or not 1:yes 0:no
	int m_iDensityEstimatorType;///< type of density estimator (0:pde, 1:SPH, 3: PDE with switch to SPH)
	double m_fInitialPerturbation;//<amount of maximal initial perturbation in dx
	bool m_iFixParticles;//<Fix the particles
	bool m_iOutputError;///<output the error compared to exact solution
	std::string m_sExactSolutionName;///<name of exact solution
	std::string m_sAuxiliaryOutput;////<name of auxiliary output type
	int m_iPelletDistribution;
	int m_iMaxParticlePerCell;	
	//-----------------------------------------------------------------------------

	
	//-------------Predetermined parameters specified in this class----------------
	
	std::size_t m_iMaxNeighbourNum;///< maximum number of neighbours of a particle
	std::size_t m_iMaxNeighbourNumInOneDir;///< maximum number of neighbours of a particle in one direction	
	std::size_t m_iNumRow2ndOrder;///< the smallest number of rows of A to solve 2nd order LPF
	std::size_t m_iNumRow1stOrder;///< the smallest number of rows of A to solve 1st order LPF
	std::size_t m_iNumCol2ndOrder;//TODO///< the number of columns of A when solving 2nd order LPF	
	std::size_t  m_iNumCol1stOrder;//TODO///< the number of columns of A when solving 1st order LPF
	double m_fNeiSearchRadius;///< the radius for neighbour search
	double m_fInvalidPressure;///< if p < invalid pressure => invalid state
	double m_fInvalidDensity;///< volume cannot be negative: if volume < invalid volume => invalid state	
	int m_iTreeDepth;///< the octree depth for neighbour search		
	double m_fContactLength;///< defined length such that for two fluid particles from different fluid object, if the distance from each other is shorter than the length the two fluid particles start to interact with each other
	double m_fTimesContactLength; ///< how many times larger than m_fInitParticleSpacing
	double m_fTimesNeiSearchRadius; ///< how many times is the neighbour search radius wrt average inter-particle spacing
	double m_fTimesCapacity; ///< how many times capacity is wrt m_iFluidNum+m_iBoundaryNum
	std::size_t m_fAdditionalCapacity; //< how many additional capacity is besides m_fTimesCapacity*(m_iFluidNum+m_iBoundaryNum)
	double m_fTimesBoundingBox; ///< how many times is the buffer zone wrt average inter-particle spacing
	// ----------------------------------------------------------------------------

	
	//------Parameters computed by the input parameters after initialization-------	
	
	std::size_t m_iCapacity;///< Maximum length of particle arrays (> the total number of all particles)  	
	std::size_t m_iFluidNum;///< Number of fluid particles
//	std::size_t m_iBoundaryNum;///< Number of boundary particles
	//std::size_t m_iGhostNum; ///< NUmber of ghost particles
	//std::size_t m_iTotalNum;///< Number of all particles
	std::size_t m_iFluidStartIndex;///< Start index of fluid particles in the particle array
//	std::size_t m_iBoundaryStartIndex;///< Start index of boundary particles in the particle array
//	std::size_t m_iGhostStartIndex;///< Start index of ghost particles in the particle array		
	std::vector<BoundingBox*> m_vFluidBoundingBox;///< Initial bounding box of the initialized fluid objects
	std::vector<Geometry*> m_vFluidObj;///< Vector of fluid objects
//	std::vector<BoundingBox*> m_vBoundaryBoundingBox;///< Initial bounding box of the initialized boundary objects
//	std::vector<Geometry*> m_vBoundaryObj;///< Vector of boundary objects 
	std::vector<Boundary*> m_vBoundaryObj;///< Vector of boundary objects
	std::vector<State*> m_vFluidObjState;///< Vector of objects of fluid state
	std::vector<std::string> m_vFluidObjNames; ///< Vector of fluid object names 	
	std::vector<std::string> m_vFluidObjStateNames; ///< Vector of fluid object state names 	
	std::vector<std::string> m_vBoundaryObjNames; ///< Vector of boundary object names
	std::vector<std::string> m_vBoundaryObjTypes; ///< Vector of boundary object types 

	size_t m_iNumParticleWithinSearchRadius;///< the number of particles within the search radius at initialization time
	bool m_iIfRestart; ///< 0:no restart 1:restart
	bool m_iIfHex;///< 0:random intialization 1:hexgonal pack

	//----------------------------------------------------------------------------
	
	//----------------------------Array data--------------------------------------
	
	double* m_vPositionX;///< x
	double* m_vPositionY;///< y
	double* m_vPositionZ;///< z
	double* m_vVelocityU;///< velocity in x-coordinate
	double* m_vVelocityV;///< velocity in y-coordinate
	double* m_vVelocityW;///< velocity in z-coordinate
	double* m_vVolume;///< volume
	double* m_vMass;
	double* m_vPressure;///< pressure	
	double* m_vSoundSpeed;///< sound speed
	//double* m_vEnergy;///< energy
	int* m_vObjectTag;///< tag=1,2,3,...:fluid objects; otherwise: boundary or ghost particles
	double* m_vLocalParSpacing;///< local particle spacing
	//-----------------------------------------------------------------------------

	int m_iNumberofPellet;
	double* m_vPelletPositionX;
	double* m_vPelletPositionY;
	double* m_vPelletPositionZ;
	double* m_vPelletRadius;
	double* m_vPelletInnerRadius;

        double Magx;
        double Magy;
        double Magz;
        double masse;
        double massNe;
        double teinf;
        double INe;
        int ZNe;
        double neinf;
        double heatK;
        double conductivity;
        double sublimationenergy;
	
	EOS* m_pEOS;///< pointer to the EOS object
	//std::ofstream debug;///< output information for debugging
	std::string m_sFilenameSaveInit; ///< the filename for saving initial setting for restart

private:
	//--------------------------------Methods--------------------------------------
	
	/// read input file
	void readInputfile(const std::string& inputfileName);
	
	/// set the params that should be optimal and should not be determined by the user
	void setParams();
	
	/// set the EOS model
	void setEOS();
	
	/// set fluid geometry & state objects and boundary geometry objects
	void setObjs();
	
	/// Set the bounding boxes of fluid and boundary objects 
	void setBoundingBox();		

	/// Allocate memory for particle array after the total number of fluid particles is determined
	void initParticleDataMemory(); 	
	
	/// Initialize fluid and boundary object geoemtry and state (calls initGeometryAndStateOnHexPacking())
	void initGeometryAndState();		
	
	/// Build fluid object geoemtry and assign state in 1D 
	size_t initGeometryAndState1D(bool saveData);

	/// Build fluid object geoemtry and assign state on hexagonal packing based on level set function 
	size_t initGeometryAndStateOnHexPacking(bool saveData);
        size_t initGeometryAndStateOnRandomDistribution(bool saveData);
	
	/// Build fluid object geoemtry and assign state on hexagonal packing based on level set function 
	size_t initGeometryAndStateOnHexPackingTemp(bool saveData);

	/// Compute number of particles within search radius wrt the specified initial particle spacing 
	void computeNumParticleWithinSearchRadius();
	
	/// Calculate and set boundary and fluid bounding box starting index in the BoundingBox objects 
	void setBoundingBoxStartIndex();
	
	/// Set up object tags (fluid objects: 1,2,3,...; boundary and ghost objects:otherwise) 
	void setObjectTag();
	
	/// Set the local inter-particle spacing
	void setLocalParSpacing();
	
	/// Set the local inter-particle spacing
	void setLocalParSpacingTemp();

	/// read parameters file (for restart)
	void readParamfile(const std::string& paramfileName);
	
	/// read data file (for restart)
	void readDatafile(const std::string& datafileName);

	/// Set the bounding boxes of fluid and boundary objects (for restart)
	void setBoundingBox(const int* tag, std::size_t num);
	
	/// modify the intial particle spacing by the average of inter particle spacing (for restart)
	void modifyInitParticleSpacing();

	/// modify the initial neighbour search radius by the modified initial inter particle spacing (for restart)
	void modifyInitNeighbourSearchRadius();

	/// modify the initial contact length by the modified initial inter particle spacing (for restart)
	void modifyInitContactLength();

	/// modify the number of particles within search radius (for restart)
	void modifyNumParticleWithinSearchRadius();

	/// modify the local inter-particle spacing
	void modifyLocalParSpacing();

	void initPellet();
};






#endif // __INITIALIZER_H__
