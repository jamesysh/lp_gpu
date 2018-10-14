/**
 * \file   lp_solver.h
 *
 * \brief  This header file contains classes of the main Lagrangian Particle solvers such as 
 *         the hyperbolic solvers and the elliptic solvers
 *
 * \author Chen, Hsin-Chiang (morrischen2008@gmail.com) 
 *
 * Co-author: Yu, Kwangmin (yukwangmin@gmail.com) on initial interface design 
 *            and the design of data pointer swaping algorithms in the Strang splitting method               
 *
 *
 * \version 1.0 
 *
 * \date 2014/10/09
 *
 * Created on: 2014/9/20 
 *
 */


#ifndef __LP_SOLVER_H__
#define __LP_SOLVER_H__

#include <cstddef>
#include <vector>
#include <string>
#include <fstream>
#include <utility>
#include <algorithm>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <magma_lapack.h>
#include <magma_v2.h>
#include "matrix_build.h"

class Initializer;
class ParticleData;
class NeighbourSearcher;
class EOS;


/**
 * \class LPSolver
 * 
 * \brief An abstract class for the family of Lagrangian Particle solvers
 *
 *
 * \author Chen, Hsin-Chiang (morrischen2008@gmail.com)
 *
 * Co-author: Yu, Kwangmin (yukwangmin@gmail.com) on initial interface design
 *
 * \version 1.0 
 *
 * \date 2014/10/09 
 *
 * Created on: 2014/09/20 
 *
 */
class LPSolver {

public:
	/// Destructor
	virtual ~LPSolver() {}

	/**
	 * \brief         The black box main Lagrangian particle solver for one iteration step
	 * 
	 * The method should be called by TimeController repeated at every time step
	 *
	 * \param [in] dt The length of physical time for this iteration
	 * \return        0 if the iteration is success 
	 * \warning       The function should always return 0 because all exceptions should be handled inside this class
	 */ 
	virtual int solve(double dt) = 0;
	
	/**
	 * \brief   Getter function of the minimum inter-particle distance among all fluid particles 
	 * \param   None
	 * \return  The minimum inter-particle distance among all fluid particles
	 */		
	virtual double getMinParticleSpacing() const {return m_fMinParticleSpacing;}
	
	/**
	 * \brief   Getter function of the maximum sound speed among all fluid particles 
	 * \param   None
	 * \return  The maximum sound speed among all fluid particles
	 */	
	virtual double getMaxSoundSpeed() const {return m_fMaxSoundSpeed;}

	/**
	 * \brief   Getter function of the maximum absolute value velocity among all fluid particles 
	 * \param   None
	 * \return  The maximum absolute value velocity among all fluid particles
	 */	
	virtual double getMaxFluidVelocity() const {return m_fMaxFluidVelocity;}
	virtual double getMinCFL() const {return m_fMinCFL;}
protected:

	double m_fMinParticleSpacing; ///< Minimum inter-particle spacing among fluid particles at a time step		
	double m_fMaxSoundSpeed; ///< Maximum sound speed of fluid particles at a time step	
	double m_fMaxFluidVelocity; ///< Maximum absolute value velocity of fluid particles at a time step
	double m_fMinCFL;
	bool m_iIfDebug;///< if true then print debug info
	std::ofstream debug;///< output information for debugging	
};


/**
 * \class HyperbolicLPSolver
 * 
 * \brief The default Lagrangian Particle solver for the compressible Euler's equation in 2D and 3D
 *
 * \author Chen, Hsin-Chiang (morrischen2008@gmail.com); Wang, Xingyu (xingyuwangcs@gmail.com)
 *
 * Co-author: Yu, Kwangmin (yukwangmin@gmail.com) on initial interface design
 *            and the design of data pointer swaping algorithms in the Strang splitting method
 *
 * \version 2.0 
 *
 * \date 2018/02/20 
 *
 * Created on: 2014/09/20 
 *
 */
class HyperbolicLPSolver : public LPSolver {

public:
	/**
	 * \brief       Constructor
	 * 
	 * Get and set up parameters and obtain access to objects needed for the main solver 
	 *
	 * \param [in] init   To retrieve information from \e init   
	 * \param [in] pData  To obtain access to an object of the PaticleData clas
	 * \param [in] ns     To obtain access to an object in the NeighbourSearcher class
	 */
	HyperbolicLPSolver(const Initializer& init, ParticleData* pData, NeighbourSearcher* ns);
	
	/**
	 * \brief         The Lagrangian particle solver for the compressible Euler's equations for one iteration step
	 * 
	 * The method should be called by TimeController repeated at every time step
	 *
	 * \param [in] dt The length of physical time for this iteration
	 * \return        0 if the iteration is success 
	 * \warning       The function should always return 0 because all exceptions should be handled inside this class
	 */
	virtual int solve(double dt);	
	
	virtual ~HyperbolicLPSolver();
private:

	//-----------------------------------------Data----------------------------------------

	//--------------------------Info got from input argument list---------------------------

	ParticleData* m_pParticleData; ///< Pointer to the object containing major particle data arrays 	
	NeighbourSearcher* m_pNeighbourSearcher; ///< Pointer to the object for the neighbour search task
	
	//--------------------------------------------------------------------------------------	

	//--------------------------Info get from Initializer class------------------------------
	
	EOS* m_pEOS; ///< Pointer to the object for computing eos
	double m_pGamma;///< gamma in polytropic and stiff polytropic eos
	double m_pPinf;///< p_inf in stiff polytropic eos
	double m_pEinf;///< e_inf in still polytropic eos
	int m_iNumThreads; ///< Number of threads	
	bool m_iIfMultiThreads;	///< true if use multithreads
	int m_iDimension; ///< dimension
	bool m_iRandomDirSplitOrder; ///< if true then the order of directional splitting is randomly set
	int m_iLPFOrder; ///< the order of Local Polynomial Fitting (LPF)
	std::size_t m_iNumRow2ndOrder; ///< the smallest number of rows of A to solve 2nd order LPF
	std::size_t m_iNumRow1stOrder; ///< the smallest number of rows of A to solve 1st order LPF
	std::size_t m_iNumCol2ndOrder; ///< the number of columns of A when solving 2nd order LPF
	std::size_t m_iNumCol1stOrder; ///< the number of columns of A when solving 1st order LPF	
	double m_fAvgParticleSpacing; ///< the average particle spacing
        double m_fInitParticleSpacing;//< the initial particle spacing for uniform density 
	double m_fGravity; ///< Gravity 
	double m_fInvalidPressure; ///< if p < invalid pressure => invalid state
	double m_fInvalidDensity; ///< volume cannot be negative: if volume < invalid volume => invalid state
	double m_fTimesNeiSearchRadius;///< how many times is the neighbour search radius wrt average inter-particle spacing	
	bool m_iIfRestart;///< if a restart run
	//--------------------------------------------------------------------------------------
	
	//---------------------------------Other parameters-------------------------------------

	int m_iNumPhase; ///< number of phases in directional splitting
	/**
	 *\brief 2D: A 2X3 table which maps direction split order and split phase to 0(x) or 1(y)\n
		     3D: A 6X5 table which maps direction split order and split phase to 0(x), 1(y), or 2(z)
	*/
	std::vector<std::vector<int> > m_vDirSplitTable; 

	int m_iDirSplitOrder;///< In 3D: 0=xyzyx, 1=xzyzx, 2=yxzxy, 3=yzxzy, 4=zxyxz, 5=zyx. In 2D: 0=xyx, 1=yxy	
	
	double m_fDt; ///< the time length of this iteration 

	//std::ofstream debug;///< output information for debugging
	//bool m_iIfDebug;///< if true then print debug info
	
	bool m_iFreeBoundary; ///< if there is free boundary condition
	bool m_iPeriodicBoundary; ///< if there is periodic boundary condition
	bool m_iSolidBoundary; ///< if there is solid boundary condition
	bool m_iUseLimiter; ///< if use limiter / switch
	int m_iDensityEstimatorType; //< if use SPH density estimator
	bool m_iFixParticles;//<if use fixed particles

	double m_fTotalTime;///< total CPU time
	double m_fSolverTime;///< CPU time to solve the sptial and temporal derivatives and update the states
	double m_fSPHTime;///< CPU time to calculate SPH density (only >0 when SPH density estimator is used)
	double m_fOctreeTime;///< CPU time to construct and search the octree
	double m_fNeighbourTime;///< CPU time to construct the GFD stencils given the octree neighbours
	double m_fBoundaryTime;///< CPU time to generate or update boundary (solid, ghost, periodic, inflow, outflow)particles
	int m_iCount;///< count of time steps

	std::vector<bool> m_vFillGhost;///< if each fludi particle has corresponding ghost particle(s) or not
	std::vector<std::size_t> m_vMirrorIndex; ///< The index of the corresponding fluid particle of a mirror particle
	//-------------------------------------------------------------------------------------

 //---------------------GPU ARRAYS NEEDED IN COMPUTATION----------------------------------
    double** d_A_LS;   //Matrices of LS linear system, stored in gpu
    double** A_temp;   //A_temp is cpu pointer but its elements are gpu pointers
    double* d_distance; 
    double** d_Tau;    //Needed in QR batched mode
    double** Tau_temp;
    int* d_info; //Test the error in qr and LS solving Process
    
    double** d_B_LS; //Right hand side of LS problem;
    double** B_temp;
    
    double** d_result;//Store the result of LS problem
    double** result_temp;
    
    double* d_vel_d_0;
    double* d_vel_dd_0;
    double* d_p_d_0;
    double* d_p_dd_0;
    
    double* d_vel_d_1;
    double* d_vel_dd_1;
    double* d_p_d_1;
    double* d_p_dd_1;


    int* d_particleOrder; //Store the order of particles in computation
    int* d_valueAssigned; //If value is assigned, it will be 0.

    int capacity = 50000;//copy from partical data

    int* d_warningCount;


	//-------------------------------------Methods-----------------------------------------
	
	/**
	 * \brief
	 *
	 */
	void checkInvalid();

	/**  
	 * \brief set ghost velocities as the corresponding fluid particle 
	 *
	 
	 */
	void setGhostVelocity(int phase);
	
	
	/**  
	 * \brief set the pressure and velocities of a mirror particle
	 *
	 */
	void setMirrorPressureAndVelocity(int phase);


	/**  
	 * \brief A composite function that calls a bunch of other methods in order to set up
	 *        the environment for the next iteration based on the update of this iteration
	 * 
	 */	
	void computeSetupsForNextIteration(); 
	
	/**  
	 * \brief Searches neighbours for fluid particles based on octree neighobur search  
	 */
	void searchNeighbourForFluidParticle();
        void searchNeighbourForFluidParticle(int choice);

        /**  
          * \brief SPH density estimator
          */
        void SPHDensityEstimatorForFluidParticle(int choice);
	
	/**  
	 * \brief Generate ghost particles for free boundary; locally generatye ghost particles for each fluid particle 
	 *         
	 */
	bool generateGhostParticleByFillingVacancy();
	
	/**
 	 * \brief Helper functions for generateGhostParticleByFilingVacancy 
	 */
	#ifdef _OPENMP
	void fillGhostParticle2D(int dir, int count[], size_t index, 
	std::vector<double>& gX, std::vector<double>& gY, std::vector<size_t>& fIndex);	
	
	void fillGhostParticle3D(int dir, int count[], size_t index, 
	std::vector<double>& gX, std::vector<double>& gY, std::vector<double>& gZ, std::vector<size_t>& fIndex);
	#else
	bool fillGhostParticle2D(int dir, int count[], size_t index, size_t* ghostIndex);
	
	bool fillGhostParticle3D(int dir, int count[], size_t index, size_t* ghostIndex);
	#endif
	
	

	/**  
	 * \brief Generate particles for solid boundary by reflecting each fluid particle 
	 *        across the solid boundary 
	 */
	bool generateSolidBoundaryByMirrorParticles();
	/**
 	 * \brief Move the out-of-boundary fluid particles to the other end of the domain and generate mirror particles for periodic boundary
 	 */ 
        bool generatePeriodicBoundaryByMirrorParticles();


	/**  
	 * \brief Set up one-sided neighbour list based on the entire neighbour list (for fluid particles only) 
	 *
	 *  
	 *
	 */
	void setUpwindNeighbourList();
	
	/**  
	* \brief Reset the order of local polynomial fitting to the pre-set value \e m_iLPFOrder  
	*
	* The order of local polynomial fitting are multiple arrays, each represent a direction 
	* (eg. two arrays for right and left in the x-coordinate) 
	*
	*/	
	void resetLPFOrder();
	
	/**  
	 * \brief Compute the minimum inter-particle spacing among fluid particles
	 *
	 * For the computation of next dt  
	 *
	 */
	void computeMinParticleSpacing();
	
	/**  
	 * \brief Compute the average inter-particle spacing among fluid particles
	 *
	 * For variable neighbour search radius  
	 *
	 */
	void computeAvgParticleSpacing();
	
	/**  
	 * \brief Update the local inter-particle spacing
	 *
	 * For variable neighbour search radius  
	 *
	 */
	void updateLocalParSpacingByVolume();
	
	/**  
	 * \brief Computes the maximum sound speed of fluid particles
	 *
	 * For the computation of next dt 
	 *
	 */
	void computeMaxSoundSpeed();
	
	/**  
	 * \brief Computes the maximum absolute value velocity of fluid particles
	 *
	 * For the computation of next dt 
	 *
	 */
	void computeMaxFluidVelocity();
	void computeMinCFL();	
	
	
	/**  
	 * \brief Calculates the GFD spatial derivatives of density
	 *
	 *
	 */
        bool density_derivative();

	/**  
	 * \brief Calculates the first order upwind spatial derivatives and performs time integration by the Strang splitting method for moving particles
	 *
	 *
	 */
	bool solve_upwind(int phase);

	/**  
	 * \brief Calculates the first and second order central spatial derivatives and performs time integration for moving particles
	 *
	 *
	 */
        bool solve_laxwendroff();

	/**  
	 * \brief Calculates the first and second order central spatial derivatives and performs time integration for fixed particles
	 *
	 *
	 */
        bool solve_laxwendroff_fix();

	/**  
	 * \brief Alias two one-sided neighbour lists based on if this round is on x, y, or z 
	 *
	 *
	 */	
	void setNeighbourListPointers(int dir, // input
		const int **neighbourList0, const int **neighbourList1, // output
		const int **neighbourListSize0, const int **neighbourListSize1);
	
	/**  
	 * \brief Alias input and output data pointers for this round 
	 *
	 * \note There are 3 rounds in total for 2D and 5 rounds in toal for 3D
	 */
	void setInAndOutDataPointers(int phase, int dir,
		const double** inVelocity, const double** inPressure, const double** inVolume, const double** inSoundSpeed, 
		double** outVelocity, double** outPressure, double** outVolume, double** outSoundSpeed);
		
	/**  
	 * \brief Alias the pointers to the arrays of the order of local polynomial fitting
	 *
	 *
	 */
	void setLPFOrderPointers(int dir, // input
		int** LPFOrder0, int** LPFOrder1, std::vector<int*>& LPFOrderOther); // output

	/**  
	 * \brief Compute the spatial derivatives by solving least squares problem in one direction for upwind method
	 *
	 *
	 */
	void computeSpatialDer(int dir, size_t index, // input
		int offset, void (HyperbolicLPSolver::*computeA) (size_t, const int *, const int*, size_t, size_t,double*, double*),
		const double* inPressure, const double* inVelocity,
		const int *neighbourList, const int *neighbourListSize,int additional,  
		int* LPFOrder, double* vel_d, double* vel_dd, double* p_d, double* p_dd); // output


	/**  
	 * \brief Compute the spatial derivatives by solving least squares problem in all directions for density_derivative
	 *
	 *
	 */

	void computeSpatialDer(size_t index, size_t offset, void (HyperbolicLPSolver::*computeA) (size_t, const int *, const int*, size_t, size_t,double*, double*),
                                                  const double* inVolume, const int *neighbourList, const int *neighbourListSize,
                                                  int *LPFOrder, double* volume_x, double* volume_y, double* volume_z);

        /**  
         * \brief Compute the spatial derivatives by solving least squares problem for all directions for Lax-Wendroff method
         *
         *
         */
	void computeSpatialDer(size_t index,  size_t offset, //input
void (HyperbolicLPSolver::*computeA) (size_t, const int *, const int*, size_t, size_t,double*, double*),  const double* inPressure, const double* inVelocityU, const double* inVelocityV, const double* inVelocityW, const double* inVolume, const int *neiList, const int *neiListSize,
                                                          int *LPFOrder, double* Pd, double* Ud, double* Vd, double* Wd, double* Volumed, int number_of_derivative);// output	
	/**  
	 * \brief Performs time integration for one direction in upwind method
	 *
	 * \note There are 3 rounds in total for 2D and 5 rounds in toal for 3D
	 */
	void timeIntegration(double real_dt, double multiplier1st, double multiplier2nd, 
		double gravity, double inVolume, double inVelocity, double inPressure,
		double inSoundSpeed, 
		double vel_d_0, double vel_dd_0, double p_d_0, double p_dd_0,
		double vel_d_1, double vel_dd_1, double p_d_1, double p_dd_1,
		double* outVolume, double* outVelocity, double* outPressure); // output

        /**  
         * \brief Performs time integration for Lax Wendroff method with moving particles
         *
         *
         */

        void timeIntegration(int index, double Dt, double gravity, double inVolume, double inVelocityU, double inVelocityV, double inVelocityW, double inPressure, double inSoundSpeed,//rotate
                                      double* Volumed, double* Ud, double* Vd, double *Wd, double *Pd,
                                                        double* outVolume, double* outVelocityU, double* outVelocityV, double* outVelocityW, double* outPressure);//output

	
        /**  
         * \brief Performs time integration for Lax Wendroff method with fixed particles
         *
         *
         */
        void timeIntegration_fix(int index, double Dt, double gravity, double inVolume, double inVelocityU, double inVelocityV, double inVelocityW, double inPressure, double inSoundSpeed,//rotate
                                      double* Volumed, double* Ud, double* Vd, double *Wd, double *Pd,
                                                        double* outVolume, double* outVelocityU, double* outVelocityV, double* outVelocityW, double* outPressure);//output

	/**  
	 * \brief Computes the number of rows and columns used for a particle at this round 
	 *
	 * \note A helper function of computeSpatialDer()
	 *
	 */
	void computeNumRowAndNumColAndLPFOrder(size_t index, // input
		const int *neighbourList, const int *neighbourListSize, size_t numRow2nd, size_t numRow1st,
		int* LPFOrder, size_t *numRow, size_t *numCol); // output


	/**  
	 * \brief Computes the matrix A in the least squares problem Ax~b in the 2D context 
	 *
	 * \note A helper function of computeSpatialDer()
	 */
	void computeA2D(size_t index, const int *neighbourList, const int* LPFOrder, size_t numRow, size_t numCol, // input
					double *A, double *distance); // output 
	
	/**  
	 * \brief Computes the matrix A in the least squares problem Ax~b in the 3D context
	 *
	 * \note A helper function of computeSpatialDer()
	 */
	void computeA3D(size_t index, const int *neighbourList, const int* LPFOrder, size_t numRow, size_t numCol, // input
					double *A, double *distance); // output
	
	/**  
	 * \brief Computes the vector b in the least squares problem Ax~b
	 *
	 * \note A helper function of computeSpatialDer()
	 */
	void computeB(size_t index, const int *neighbourList, size_t numRow, const double* inData, 
				  double *b); // output
	
	/**  
	 * \brief Updates the states of fluid particles at the end of one iteration by swapping pointers 
	 *
	 *
	 */
	void updateFluidState();
	
	/**  
	 * \brief Updates the location of fluid particles based on velocities at the end of one iteration
	 *
	 * Based on a combination of forward and backward Euler's method
	 */
	void moveFluidParticle();
	
	/**  
	 * \brief Update the velocities of fluid particles at the end of one iteration by swapping pointers
	 *
	 * 
	 */
	void updateFluidVelocity();

	void calculateHeatDeposition();

	void updateStatesByLorentzForce();

	void computeIntegralSpherical();

//GPU functions
   void computeSpatialDer_gpu(int dir, int offset,  void (HyperbolicLPSolver::*computeA) ( const int *,
const int*, int, int, int),
	const double* inPressure, const double* inVelocity,
	const int *neighbourList, const int *neighbourListSize,int additional,
	int* LPFOrder, double* vel_d, double* vel_dd, double* p_d, double* p_dd);

    void computeA3D_cpu( const int *neighbourList,  const int* LPFOrder, int numRow, int startIndex, int numComputingPrticle);
    void computeA2D_cpu(const int *neighbourList, const int*  LPFOrder,  int numRow,   int startIndex, int numComputingParticle); 
    void setLPFOrderPointers_gpu(int dir, int** LPFOrder0, int** LPFOrder1, std::vector<int*>& LPFOrderOther); 
    void setInAndOutDataPointers_gpu(int phase, int dir,
	const double** inVelocity, const double** inPressure, const double** inVolume, const double** inSoundSpeed, 
	double** outVelocity, double** outPressure, double** outVolume, double** outSoundSpeed); 
    void setNeighbourListPointers_gpu(int dir, const int **neiList0, const int **neiList1, // output
	const int **neiListSize0, const int **neiListSize1);

};
#endif
