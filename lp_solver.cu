#include "lp_solver.h"
#include "boundary.h"
#include "neighbour_searcher.h"
#include "eos.h"
#include "particle_data.h"
#include "initializer.h"
#include "ls_solver.h"
#include "hexagonal_packing.h"
#include "omp.h"
//#include "voronoi_area_estimator.h"
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <iostream>
#include <memory> // shared_ptr
#include <iomanip> // setw
#include "matrix_build.h"
using namespace std;


////////////////////////////////////////////////////////////////////////////////////////
// Start of HyperbolicLPSolver
////////////////////////////////////////////////////////////////////////////////////////


HyperbolicLPSolver::HyperbolicLPSolver(const Initializer& init, ParticleData* pData, NeighbourSearcher* ns) {
	
	srand(time(0));

	m_pParticleData = pData; 
	m_pNeighbourSearcher = ns;
	m_pEOS = init.getEOS();
	std::vector<double> eos_parameters;
	m_pEOS->getParameters(eos_parameters);

	if(eos_parameters.size()==1)
	{
		m_pGamma=eos_parameters[0];
		m_pPinf=0.;
		m_pEinf=0.;
		cout<<"Polytropic EOS, gamma = "<<m_pGamma<<endl;	
	}
	else if(eos_parameters.size()==3)
        {
                m_pGamma=eos_parameters[0];
                m_pPinf=eos_parameters[1];
                m_pEinf=eos_parameters[2];
                cout<<"Stiff polytropic EOS, gamma = "<<m_pGamma<<", P_inf = "<<m_pPinf<<", E_inf = "<<m_pEinf<<endl;
        }
	else
	{
		m_pGamma=0.;
		m_pPinf=0.;
		m_pEinf=0.;
		cout<<"Warning: Cannot recognize EOS."<<endl;
	}
	
	// get parameters from init
	m_iNumThreads = init.getNumThreads();
	m_iDimension = init.getDimension();
	m_iNumPhase = m_iDimension==3? 5:3;
	m_iRandomDirSplitOrder = init.getRandomDirSplitOrder();
	m_iLPFOrder = init.getLPFOrder(); 
	m_iNumRow1stOrder = init.getNumRow1stOrder();
	m_iNumRow2ndOrder = init.getNumRow2ndOrder();
	m_iNumCol1stOrder = init.getNumCol1stOrder(); 	
	m_iNumCol2ndOrder = init.getNumCol2ndOrder();
	m_fAvgParticleSpacing = init.getInitParticleSpacing();
        m_fInitParticleSpacing = m_fAvgParticleSpacing;
	m_fGravity = init.getGravity(); 
	m_fInvalidPressure = init.getInvalidPressure(); 
	m_fInvalidDensity = init.getInvalidDensity();
	m_fTimesNeiSearchRadius = init.getTimesNeiSearchRadius();
	m_iIfRestart = init.getIfRestart();
	m_iUseLimiter = init.getUseLimiter();
	m_iDensityEstimatorType=init.getDensityEstimatorType();
	m_iFixParticles=init.getFixParticles();
	if(m_iLPFOrder==2)
	{
		if(m_iUseLimiter)
			printf("Second order Lax-Wendroff type method with switch to first order\n");
		else
			printf("Second order Lax-Wendroff type method\n");
	}
	else
	{
		printf("First order upwind method\n");
	}
	if(m_iDensityEstimatorType==1)
		printf("SPH density estimator\n");
	else
	{
		if(m_iDensityEstimatorType==0)
			printf("PDE density updator\n");
		else
			printf("PDE density updator with switch to SPH density estimator\n");

	}
	if(m_iFixParticles)
		printf("Fixed particles\n");
	else
		printf("Moving particles\n");

	//m_iIfDebug = init.getIfDebug();
	//debug.open(init.getDebugfileName(), std::ofstream::out | std::ofstream::app);	
	
	// all fluid objects should be distant at initialization
	// this variable should always be false in the single fluid object case

	// set OpenMP environment
	m_iIfMultiThreads = false;
	if(m_iNumThreads > 0) {//> 1
		// set the number of threads
		omp_set_num_threads(min(omp_get_max_threads(), m_iNumThreads));	
		m_iIfMultiThreads = true;	
		
		cout<<"-------HyperbolicLPSolver::HyperbolicLPSolver()-------"<<endl;
		cout<<"m_iNumThreads = "<<m_iNumThreads<<endl;
		cout<<"omp_get_num_procs() = "<<omp_get_num_procs()<<endl;
		cout<<"omp_get_max_threads() = "<<omp_get_max_threads()<<" (after omp_set_num_threads())"<<endl;
		cout<<"------------------------------------------------------"<<endl;
	}

	if(m_iDimension==2) {
		m_vDirSplitTable = vector<vector<int> >({{0,1,0},{1,0,1}});
	}
	else if(m_iDimension==3)
		m_vDirSplitTable = vector<vector<int> >
		({{0,1,2,1,0},
		  {0,2,1,2,0},
		  {1,0,2,0,1},
		  {1,2,0,2,1},
		  {2,0,1,0,2},
		  {2,1,0,1,2}});
	
	// for completeness initialize to zero
	m_iDirSplitOrder = 0;//default: 0
	m_fDt = 0;
		
	
	m_iFreeBoundary = false;
	m_iPeriodicBoundary = false;
	m_iSolidBoundary = false;
	for(auto s:m_pParticleData->m_vBoundaryObjTypes) {
		if(s=="free") {
			if(m_iFreeBoundary) continue; // avoid re-initialize memory
			m_iFreeBoundary = true;
			m_vFillGhost = vector<bool>(m_pParticleData->m_iCapacity,false);
		}
		else if(s=="periodic") {
			m_iPeriodicBoundary = true;
		}
		else if(s=="solid" || s=="inflow" || s=="outflow") {
			m_iSolidBoundary = true;
		}
	}	

	m_fTotalTime=0;
	m_fSolverTime=0;
        m_fSPHTime=0;
        m_fOctreeTime=0;
        m_fNeighbourTime=0;
        m_fBoundaryTime=0;

        searchNeighbourForFluidParticle(0);

	computeSetupsForNextIteration();
		
        m_fTotalTime=0;
        m_fSolverTime=0;
        m_fSPHTime=0;
        m_fOctreeTime=0;
        m_fNeighbourTime=0;
        m_fBoundaryTime=0;
	m_iCount=0;

// --------------------GPU ARRAYS MEMORY ALLOCATION--------------------------------
    int numNeighbourInOneDir = m_pParticleData->m_iMaxNeighbourNumInOneDir;
    int m_iCapacity = m_pParticleData->m_iCapacity;
    
    cudaMalloc((void**)&d_valueAssigned,sizeof(int)*m_iCapacity);
    cudaMemset(d_valueAssigned, 0, sizeof(int)*m_iCapacity); 
    cudaMalloc((void**)&d_A_LS,sizeof(double*)*capacity);
    A_temp = new double*[capacity];
    #ifdef _OPENMP
    #pragma omp parallel for 
    #endif
    for(int i=0;i<capacity;i++){
        cudaMalloc((void**)&A_temp[i],sizeof(double)*10*numNeighbourInOneDir);
    }
    cudaMemcpy(d_A_LS, A_temp,sizeof(double*)*capacity,cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&d_distance,sizeof(double)*capacity);
    
    cudaMalloc((void**)&d_Tau,capacity*sizeof(double*));
    Tau_temp = new double*[capacity];
    #ifdef _OPENMP
    #pragma omp parallel for 
    #endif
    for(int i=0;i<capacity;i++){
        cudaMalloc((void**)&Tau_temp[i],sizeof(double)*10);
    }

    cudaMemcpy(d_Tau, Tau_temp, sizeof(double*)*capacity, cudaMemcpyHostToDevice);  
 

    cudaMalloc((void**)&d_info,capacity*sizeof(int));
   
    cudaMalloc((void**)&d_B_LS,sizeof(double*)*capacity);
    B_temp = new double*[capacity];
    #ifdef _OPENMP
    #pragma omp parallel for 
    #endif
    for(int i=0;i<capacity;i++){
        cudaMalloc((void**)&B_temp[i],sizeof(double)*numNeighbourInOneDir);
    }
    cudaMemcpy(d_B_LS,B_temp,sizeof(double*)*capacity,cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&d_result,capacity*sizeof(double*));
    result_temp = new double*[capacity];
    #ifdef _OPENMP
    #pragma omp parallel for 
    #endif
    for(int i=0;i<capacity;i++){
        cudaMalloc((void**)&result_temp[i],sizeof(double)*10);
    }

    cudaMemcpy(d_result, result_temp, sizeof(double*)*capacity, cudaMemcpyHostToDevice);  
   
   cudaMalloc((void**)&d_warningCount,sizeof(int));
   
    cudaMalloc((void**)&d_info_single,sizeof(int));
    cudaError err = cudaGetLastError();
        if(cudaSuccess != err){
            printf("Error occurs when setting up lp_solver!!! MSG: %s\n",cudaGetErrorString(err));
            assert(false);
        }
        cout<<"-----------------allocate end-----------------------"<<endl;

}

HyperbolicLPSolver::~HyperbolicLPSolver() {
	delete m_pEOS;

    delete[] A_temp;
    delete[] B_temp;
    delete[] Tau_temp;
    delete[] result_temp;
    cudaFree(d_A_LS);
    cudaFree(d_B_LS);
    cudaFree(d_Tau);
    cudaFree(d_info);
    cudaFree(d_result);
    cudaFree(d_valueAssigned);
    cudaFree(d_warningCount);   
    cudaFree(d_info_single);
    #ifdef _OPENMP
    #pragma omp parallel for 
    #endif
    for(int i=0;i<capacity;i++){
        cudaFree(A_temp[i]);
        cudaFree(B_temp[i]);
        cudaFree(result_temp[i]);
        cudaFree(Tau_temp);

    }
cout<<"memory release end-------------------------------"<<endl;



}


void HyperbolicLPSolver::computeSetupsForNextIteration() {
		
	double startTime;

	startTime = omp_get_wtime();

	updateStatesByLorentzForce();

	if(m_iSolidBoundary) generateSolidBoundaryByMirrorParticles();
	if(m_iPeriodicBoundary) generatePeriodicBoundaryByMirrorParticles();
	if(m_iSolidBoundary || m_iPeriodicBoundary) 
	{
        m_fBoundaryTime+=omp_get_wtime() - startTime;
//		printf("Create boundary particles takes %.16g seconds\n", omp_get_wtime() - startTime);
	}
//Octree: build octree and use it to search neighbours
	startTime = omp_get_wtime();
	searchNeighbourForFluidParticle(0);
        m_fOctreeTime+=omp_get_wtime() - startTime;
//	printf("Search neighbours for all fluid particles takes %.16g seconds\n", omp_get_wtime() - startTime);

        updateLocalParSpacingByVolume();

        checkInvalid();

	startTime = omp_get_wtime();
	if(m_iFreeBoundary) generateGhostParticleByFillingVacancy();
	if(m_iFreeBoundary)
	{ 
                m_fBoundaryTime+=omp_get_wtime() - startTime;
//		printf("Fill ghost particles takes %.16g seconds\n", omp_get_wtime() - startTime);
	}
//Stencil: set upwind and central GFD stencils
	startTime = omp_get_wtime();
setUpwindNeighbourList();
m_fNeighbourTime+=omp_get_wtime() - startTime;
//	printf("Set upwind neighbour list takes %.16g seconds\n", omp_get_wtime() - startTime);

startTime = omp_get_wtime();
if(m_iDensityEstimatorType) density_derivative();
SPHDensityEstimatorForFluidParticle(m_iDensityEstimatorType);
if(m_iDensityEstimatorType) 
{
	m_fSPHTime+=omp_get_wtime() - startTime;
//		printf("SPH density estimator for all fluid particles takes %.16g seconds\n", omp_get_wtime() - startTime);
}
// initialize the LPF order (1 or 2) in directions right, left, north, and south
resetLPFOrder();


// to determine the dt for next step
computeMinParticleSpacing();
computeMaxSoundSpeed();
computeMaxFluidVelocity();
computeMinCFL();

}

int HyperbolicLPSolver::solve(double dt) {	
//cout<<"--------------HyperbolicLPSolver::solve()--------------"<<endl;
//	cout<<"---------------------------------------------------------"<<endl;
double currentstepstartTime;
currentstepstartTime = omp_get_wtime();
// dt for this time step 
m_fDt = dt;
#ifdef _OPENMP
#pragma omp parallel for
#endif
for(size_t index=m_pParticleData->m_iFluidStartIndex;
index<m_pParticleData->m_iFluidStartIndex+m_pParticleData->m_iFluidNum; index++) {
	m_pParticleData->m_vVolumeOld[index] = m_pParticleData->m_vVolume[index];
}
double startTime;
startTime = omp_get_wtime();
bool phase_success;
if(m_iFixParticles==0){
//Solver: upwind scheme
    magma_init();
for(int phase=0; phase<m_iNumPhase; ) {

	cout<<"upwind phase="<<phase<<endl;
    m_pParticleData->cpyFromHostToDevice();


double startTime1,endTime;
startTime1 = omp_get_wtime();
	phase_success = solve_upwind(phase);
    cudaDeviceSynchronize();
endTime = omp_get_wtime();
printf("The computation time is %.5f.\n",endTime-startTime1);
    m_pParticleData->cpyFromDeviceToHost();
    if(!phase_success) {
		phase = 0;
		//numParNotUpdate++;
		cout<<"GO BACK TO PHASE 0!!!!!!!"<<endl;
		continue;
	}

	if(m_iSolidBoundary || m_iPeriodicBoundary) setMirrorPressureAndVelocity(phase);
	if(m_iFreeBoundary) setGhostVelocity(phase);


	phase++;
}
    magma_finalize();
//Solver: Lax-Wendroff scheme
if(m_iLPFOrder==2)
	phase_success = solve_laxwendroff();
}
else
{
phase_success = solve_laxwendroff_fix();
}
m_fSolverTime+=omp_get_wtime() - startTime;
//        printf("No directioinal Splitting takes %.16g seconds\n", omp_get_wtime() - startTime);
if(!phase_success) {
	cout<<"Error in nodirectionalSplitting"<<endl;
	return 1;
}
updateFluidState();
if(m_iFixParticles==0)
	moveFluidParticle();
//moveFluidParticleAdjusted();
updateFluidVelocity();	

computeSetupsForNextIteration();
m_fTotalTime+=omp_get_wtime() - currentstepstartTime;
m_iCount++;
cout<<"****************************************************************************"<<endl;
cout<< setw(60) <<"Number of time steps = "<<m_iCount<<endl;
cout<< setw(60) <<"Total running time = "<<m_fTotalTime<<endl;
cout<< setw(60) <<"Time to solve GFD derivative and update states = "<<m_fSolverTime<<endl;
cout<< setw(60) <<"Time to estimate SPH density = "<<m_fSPHTime<<endl;
cout<< setw(60) <<"Time to bluid octree and search neighbours = "<<m_fOctreeTime<<endl;
cout<< setw(60) <<"Time to bluid GFD stencil = "<<m_fNeighbourTime<<endl;	
cout<< setw(60) <<"Time to generate boundary particles = "<<m_fBoundaryTime<<endl;
cout<<"****************************************************************************"<<endl;
//cout<<"-------------------------------------------------------"<<endl;

return 0;
}


void HyperbolicLPSolver::checkInvalid() {
size_t fluidStartIndex = m_pParticleData->m_iFluidStartIndex;
size_t fluidEndIndex = m_pParticleData->m_iFluidStartIndex + m_pParticleData->m_iFluidNum;
#ifdef _OPENMP
#pragma omp parallel for
#endif
for(size_t index=fluidStartIndex; index<fluidEndIndex; index++) {
	if(std::isnan(m_pParticleData->m_vPositionX[index]) || std::isinf(m_pParticleData->m_vPositionX[index])) {
		cout<<"invalid x"<<endl;
		assert(false);
	}
	if(std::isnan(m_pParticleData->m_vPositionY[index]) || std::isinf(m_pParticleData->m_vPositionY[index])) {
		cout<<"invalid y"<<endl;
		assert(false);
	}
	if(std::isnan(m_pParticleData->m_vVelocityU[index]) || std::isinf(m_pParticleData->m_vVelocityU[index])) {
		cout<<"invalid u"<<endl;
		assert(false);
	}
	if(std::isnan(m_pParticleData->m_vVelocityV[index]) || std::isinf(m_pParticleData->m_vVelocityV[index])) {
		cout<<"invalid v"<<endl;
		assert(false);
	}
	if(std::isnan(m_pParticleData->m_vPressure[index]) || std::isinf(m_pParticleData->m_vPressure[index])) {
		cout<<"invalid p"<<endl;
		assert(false);
	}
	if(std::isnan(m_pParticleData->m_vVolume[index]) || std::isinf(m_pParticleData->m_vVolume[index])) {
		cout<<"invalid vol"<<endl;
		assert(false);
	}
	if(std::isnan(m_pParticleData->m_vSoundSpeed[index]) || std::isinf(m_pParticleData->m_vSoundSpeed[index])) {
		cout<<"invalid cs"<<endl;
		assert(false);
	}
	if(std::isnan(m_pParticleData->m_vLocalParSpacing[index]) || std::isinf(m_pParticleData->m_vLocalParSpacing[index]            || m_pParticleData->m_vLocalParSpacing[index] <=0)) {
		cout<<"invalid spacing, spacing="<<m_pParticleData->m_vLocalParSpacing[index]<<endl;
		assert(false);
	}
}
}

void HyperbolicLPSolver::searchNeighbourForFluidParticle() {
	searchNeighbourForFluidParticle(0);
}

void HyperbolicLPSolver::computeIntegralSpherical(){
        const double *positionX = m_pParticleData->m_vPositionX;
        const double *positionY = m_pParticleData->m_vPositionY;
        const double *positionZ = m_pParticleData->m_vPositionZ; // is all zero for the 2D case 
        const double *mass = m_pParticleData->m_vMass;
        double *leftintegral = m_pParticleData->m_vLeftIntegral;

        int fluidStartIndex = m_pParticleData->getFluidStartIndex();
        int fluidEndIndex = fluidStartIndex + m_pParticleData->getFluidNum();

	std::vector<std::pair<double,int>> vec(m_pParticleData->m_iFluidNum);
	for(int index=fluidStartIndex; index<fluidEndIndex; index++)
	{
		double r2=positionX[index]*positionX[index]+positionY[index]*positionY[index]+positionZ[index]*positionZ[index];
		vec[index]={r2,index};
	}
	std::sort(vec.begin(),vec.end());
	double integral=0;
	for(int index=fluidEndIndex-1; index>=fluidStartIndex; index--)
	{
		double temp=mass[vec[index].second]/4.0/3.1416/vec[index].first;
		leftintegral[vec[index].second]=integral+0.5*temp;
		integral+=temp;
	}
}

void HyperbolicLPSolver::searchNeighbourForFluidParticle(int choice) {
	
	cout<<"-------HyperbolicLPSolver::searchNeighbourForFluidParticle()-------"<<endl;

	const double *positionX = m_pParticleData->m_vPositionX;
	const double *positionY = m_pParticleData->m_vPositionY;
	const double *positionZ = m_pParticleData->m_vPositionZ; // is all zero for the 2D case	
	const double *mass = m_pParticleData->m_vMass;
	double *leftintegral = m_pParticleData->m_vLeftIntegral;
	double *rightintegral = m_pParticleData->m_vRightIntegral;
	double *VolumeVoronoi= m_pParticleData->m_vVolumeVoronoi;


	if(choice==2)
	{
		std::cout<<"Voronoi density estimator has been removed from LP code. Please use another density estimator. In this simulation, PDE density updator will be used."<<std::endl;
		choice=0;
/*		std::cout<<"Calculating Voronoi Area"<<std::endl;
	        VoronoiAreaEstimator voronoi(m_iDimension, m_pParticleData->m_iFluidNum + m_pParticleData->m_iBoundaryNum, positionX, positionY, positionZ, mass, VolumeVoronoi);
		int voronoi_error=voronoi.ComputeVoronoiArea();
		if (voronoi_error){
			std::cout<<"Error in voronoi area estimator"<<std::endl;
		}*/
	}
	if(choice!=0 && choice!=1 && choice!=3)
	{
		std::cout<<"This is not a valid choice of density estimator. In this simulation, PDE density updator will be used."<<std::endl;
		choice=0;
	}

	double startTime = omp_get_wtime();
//	cout<<"Start to build octree"<<endl;
	
	m_pNeighbourSearcher->buildSearchStructure(positionX, positionY, positionZ, mass, VolumeVoronoi,
	m_pParticleData->m_iFluidStartIndex ,m_pParticleData->m_iFluidNum + m_pParticleData->m_iBoundaryNum);
//	cout<<"Calculate integral"<<endl;
//        printf("Build octree takes %.16g seconds\n", omp_get_wtime() - startTime);
	if(m_pParticleData->m_iNumberofPellet){
	        startTime = omp_get_wtime();

//Compute integral from x+ and x- directions using octree
//		m_pNeighbourSearcher->computeIntegralAPCloud(mass, leftintegral, rightintegral, m_pParticleData->m_iFluidNum + m_pParticleData->getInflowNum(),m_pParticleData->getMaxParticlePerCell());

//Compute integral for spherical symmetry case
		computeIntegralSpherical();

//		cout<<"Integral calculated"<<endl;	
//		printf("Calculate integral takes %.16g seconds\n", omp_get_wtime() - startTime);
		startTime = omp_get_wtime();
		calculateHeatDeposition();
//		cout<<"Heat deposition calculated"<<endl;      
//              	printf("Calculate head deposition takes %.16g seconds\n", omp_get_wtime() - startTime);

	}
//	cout<<"end building octree"<<endl;

	int *neighbourList = m_pParticleData->m_vNeighbourList;
	int *neighbourListSize = m_pParticleData->m_vNeighbourListSize;	

	size_t fluidStartIndex = m_pParticleData->getFluidStartIndex();
	size_t fluidEndIndex = fluidStartIndex + m_pParticleData->getFluidNum();
	
	size_t maxNeiNum = m_pParticleData->m_iMaxNeighbourNum;		
	 
//	startTime = omp_get_wtime();
//	cout<<"Search neighbours and calculate density"<<endl;
	#ifdef _OPENMP
	#pragma omp parallel
	{
	
	int tid = omp_get_thread_num();
	
	#endif	
	
	double neiListDist[maxNeiNum]; // a temp array for dist between a particle and its neighbours
	size_t numNeiFound;	

	// fluid
	if(m_pParticleData->m_vFluidBoundingBox.size() > 1) { // multiple fluid objects
		
		#ifdef _OPENMP
		#pragma omp for
		#endif
		for(size_t index=fluidStartIndex; index<fluidEndIndex; index++) { 
			size_t neiListStartIndex = index*maxNeiNum;		
			
				
			double radius = m_fTimesNeiSearchRadius * sqrt(2*mass[index]*m_pParticleData->m_vVolumeOld[index]/1.7321);
			#ifdef _OPENMP
			m_pNeighbourSearcher->searchNeighbourQuadrant(
				positionX[index],positionY[index],positionZ[index],radius, 
				neighbourList+neiListStartIndex,neiListDist,numNeiFound,tid,index); // output	
			#else
			m_pNeighbourSearcher->searchNeighbourQuadrant(
				positionX[index],positionY[index],positionZ[index],radius, 
				neighbourList+neiListStartIndex,neiListDist,numNeiFound,index); // output
			#endif

//sph density estimator, the option is keeped but not used. The current version call the SPH density estimator using a seperate function, SPHDensityEstimatorForFluidParticle 
			if(choice==1){
                        double radius2multiplier=3.0;
                        double radius2 = radius2multiplier*sqrt(2*mass[index]*m_pParticleData->m_vVolumeOld[index]/1.7321);
                        double count_density=1.0/m_pParticleData->m_vVolumeOld[index];
                        m_pNeighbourSearcher->densityEstimator(
                                positionX[index],positionY[index],positionZ[index],radius2,
                                &count_density,m_pParticleData->m_vVolume_x[index],m_pParticleData->m_vVolume_y[index],m_pParticleData->m_vVolume_z[index]);
                        if(m_iDimension==2){
                                m_pParticleData->m_vVolume[index]=1.0/count_density;
                        }
			else{//3d TODO
			}
			}
//end of sph density estimator

			if(numNeiFound > maxNeiNum) {
				cout<<"numNeiFound="<<numNeiFound<<" > maxNeiNum="<<maxNeiNum<<endl;
				assert(false);
			}

			neighbourListSize[index] = numNeiFound;		    
		}

	}
	else { // single fluid object
		#ifdef _OPENMP
		#pragma omp for
		#endif
		for(size_t index=fluidStartIndex; index<fluidEndIndex; index++) { 
			size_t neiListStartIndex = index*maxNeiNum;	
				
			double radius;
                        if(m_iDimension==2) radius = m_fTimesNeiSearchRadius * sqrt(2*mass[index]*m_pParticleData->m_vVolumeOld[index]/1.7321);
			if(m_iDimension==3) radius = m_fTimesNeiSearchRadius * cbrt(1.4142*mass[index]*m_pParticleData->m_vVolumeOld[index]);	

//Using octree to search neighbours. The neighbourhood should contains similar number of particles from each direction.
			if(m_iDimension==3) {
	                        #ifdef _OPENMP
        	                m_pNeighbourSearcher->searchNeighbour(//Direction
                	                positionX[index],positionY[index],positionZ[index],radius, 
                        	        neighbourList+neiListStartIndex,neiListDist,numNeiFound,tid,index); // output   
                        	#else
	                        m_pNeighbourSearcher->searchNeighbour(//Direction
        	                        positionX[index],positionY[index],positionZ[index],radius, 
                	                neighbourList+neiListStartIndex,neiListDist,numNeiFound,index); // output
                        	#endif
			}
                        if(m_iDimension==2) {
                                #ifdef _OPENMP
                                m_pNeighbourSearcher->searchNeighbour(//Quadrant
                                        positionX[index],positionY[index],positionZ[index],radius,
                                        neighbourList+neiListStartIndex,neiListDist,numNeiFound,tid,index); // output   
                                #else
                                m_pNeighbourSearcher->searchNeighbour(//Quadrant
                                        positionX[index],positionY[index],positionZ[index],radius,
                                        neighbourList+neiListStartIndex,neiListDist,numNeiFound,index); // output
                                #endif
                        }


//SPH density estimator option is keeped but not used. The current LP algorithm call sthe SPH density estimator using a seperate function, SPHDensityEstimatorForFluidParticle 
			if (choice==1){
                        double radius2multiplier;
			if (m_iDimension==2)
				radius2multiplier=5.0;
			if(m_iDimension==3)
				radius2multiplier=5.0;
			double radius2;
			if(m_iDimension==2) radius2 = radius2multiplier*sqrt(2*mass[index]*m_pParticleData->m_vVolumeOld[index]/1.7321);
                        if(m_iDimension==3) radius2 = radius2multiplier*cbrt(1.4142*mass[index]*m_pParticleData->m_vVolumeOld[index]);
                        double count_density=1.0/m_pParticleData->m_vVolumeOld[index];
			if(m_iDimension==3) count_density=5.0*radius2/7.0*count_density;
                        m_pNeighbourSearcher->densityEstimator(
                                index, positionX[index],positionY[index],positionZ[index],radius2,
                                &count_density,m_pParticleData->m_vVolume_x[index],m_pParticleData->m_vVolume_y[index],m_pParticleData->m_vVolume_z[index]);
                        if(m_iDimension==2){
                                m_pParticleData->m_vVolume[index]=1.0/count_density;  
                        }
                        else{//3d
				m_pParticleData->m_vVolume[index]=5.0*radius2/7.0/count_density;
                        }
			}
//end of sph density estimator
//Voronoi density estimator option is keeped but not used.
                        if (choice==2){
                        double radius2multiplier=5.0;
                        double radius2 = radius2multiplier*sqrt(2*mass[index]*m_pParticleData->m_vVolumeOld[index]/1.7321);
			double count_density=0;
                        m_pNeighbourSearcher->VoronoiDensityEstimator(index,
                                positionX[index],positionY[index],positionZ[index],radius2,
                                &count_density);
                        if(m_iDimension==2){
                                m_pParticleData->m_vVolume[index]=1.0/count_density;
                        }
                        else{//3d TODO
                        }
                        }
//end of voronoi density estimator


			if(numNeiFound > maxNeiNum) {
				cout<<"numNeiFound="<<numNeiFound<<" > maxNeiNum="<<maxNeiNum<<endl;
				assert(false);
			}

			neighbourListSize[index] = numNeiFound;
			//cout<<"numNeiFound"<<numNeiFound<<endl;
			//cout<<"(x,y)=("<<positionX[index]<<","<<positionY[index]<<endl;
		}
	}
	
    #ifdef _OPENMP
	}
	#endif 
	
//	printf("Searching neighbours takes %.16g seconds\n", omp_get_wtime() - startTime);

	cout<<"-------END HyperbolicLPSolver::searchNeighbourForFluidParticle()-------"<<endl;

}

//Supportive Bessel function for heat deposition calculation

/* Bessel_I0 returns the modifies Bessel function I0(x) of positive real x  */

float           Bessel_I0(
	float           x)
{
        float   p1 = 1.0;
        float   p2 = 3.5156229;
        float   p3 = 3.0899424;
        float   p4 = 1.2067492;
        float   p5 = 0.2659732;
        float   p6 = 0.360768e-1;
        float   p7 = 0.45813e-2;
	
        float   q1 = 0.39894228;
        float   q2 = 0.1328592e-1;
        float   q3 = 0.225319e-2;
        float   q4 = -0.157565e-2;
        float   q5 = 0.916281e-2;
        float   q6 = -0.2057706e-1;
        float   q7 = 0.2635537e-1;
        float   q8 = -0.1647633e-1;
        float   q9 = 0.392377e-2;
	
	float   ax, y, value;
	
	if (fabs(x) < 3.75)
	  {
	    y = (x/3.75)*(x/3.75);//sqr
	    value = p1+y*(p2+y*(p3+y*(p4+y*(p5+y*(p6+y*p7)))));
	  }
	else
	  {
	    ax = fabs(x);
	    y = 3.75/ax;

	    value = (exp(ax)/sqrt(ax))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*(q7+y*(q8+y*q9))))))));
	  }

	return value;
}


/* Bessel_I1 returns the modifies Bessel function I1(x) of positive real x  */

float           Bessel_I1(
	float           x)
{
        float   p1 = 0.5;
        float   p2 = 0.87890594;
        float   p3 = 0.51498869;
        float   p4 = 0.15084934;
        float   p5 = 0.2658733e-1;
        float   p6 = 0.301532e-2;
        float   p7 = 0.32411e-3;
	
        float   q1 = 0.39894228;
        float   q2 = -0.3988024e-1;
        float   q3 = -0.362018e-2;
        float   q4 = 0.163801e-2;
        float   q5 = -0.1031555e-1;
        float   q6 = 0.2282967e-1;
        float   q7 = -0.2895312e-1;
        float   q8 = 0.1787654e-1;
        float   q9 = -0.420059e-2;
	
	float   ax, y, value;
	
	if (fabs(x) < 3.75)
	  {
	    y = (x/3.75)*(x/3.75);//sqr
	    value = x*(p1+y*(p2+y*(p3+y*(p4+y*(p5+y*(p6+y*p7))))));
	  }
	else
	  {
	    ax = fabs(x);
	    y = 3.75/ax;

	    value = (exp(ax)/sqrt(ax))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*(q7+y*(q8+y*q9))))))));
	    if (x < 0)
	      value *= -1.0;
	  }
	return value;
}

/* Bessel_K0 returns the modifies Bessel function K0(x) of positive real x  */

float           Bessel_K0(
	float           x)
{
        float   p1 = -0.57721566;
	float   p2 = 0.4227842;
	float   p3 = 0.23069756;
	float   p4 = 0.348859e-1;
	float   p5 = 0.262698e-2;
	float   p6 = 0.1075e-3;
	float   p7 = 0.74e-5;

	float   q1 = 1.25331414;
	float   q2 = -0.7832358e-1;
	float   q3 = 0.2189568e-1;
	float   q4 = -0.1062446e-1;
	float   q5 = 0.587872e-2;
	float   q6 = -0.25154e-2;
	float   q7 = 0.53208e-3;

	float   y, value;

	if (x <= 2.0)
	  {
	    y = x*x/4.0;
	    value = (-log(x/2.0)*Bessel_I0(x))+(p1+y*(p2+y*(p3+y*(p4+y*(p5+y*(p6+y*p7))))));
	  }
	else
	  {
	    y = 2.0/x;
	    value = (exp(-x)/sqrt(x))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*q7))))));
	  }
	return value;
}

/* Bessel_K1 returns the modifies Bessel function K1(x) of positive real x  */

float           Bessel_K1(
	float           x)
{
        float   p1 = 1.0;
	float   p2 = 0.15443144;
	float   p3 = -0.67278579;
	float   p4 = -0.18156897;
	float   p5 = -0.01919402;
	float   p6 = -0.110404e-2;
	float   p7 = -0.4686e-4;

	float   q1 = 1.25331414;
	float   q2 = 0.23498619;
	float   q3 = -0.3655620e-1;
	float   q4 = 0.1504268e-1;
	float   q5 = -0.780353e-2;
	float   q6 = 0.325614e-2;
	float   q7 = -0.68245e-3;

	float   y, value;

	if (x <= 2.0)
	  {
	    y = x*x/4.0;
	    value = (log(x/2.0)*Bessel_I1(x))+(1.0/x)*(p1+y*(p2+y*(p3+y*(p4+y*(p5+y*(p6+y*p7))))));
	  }
	else
	  {
	    y = 2.0/x;
	    value = (exp(-x)/sqrt(x))*(q1+y*(q2+y*(q3+y*(q4+y*(q5+y*(q6+y*q7))))));
	  }
	return value;
}
				       		       

/* Bessel_Kn returns the modifies Bessel function Kn(x) of positive real x for n >= 2 */

float           Bessel_Kn(
        int             n,
	float           x)
{
        int    j;
        float  bk, bkm, bkp, tox;

	if (n < 2)
	  {
	    printf("Error in Bessel_Kn(), the order n < 2: n = %d\n",n);
	    assert(false);
	    return 0;
	  }

	tox = 2.0/x;
	bkm = Bessel_K0(x);
	bk = Bessel_K1(x);

	for (j = 1; j < n; j++)
	  {
	    bkp = bkm + j*tox*bk;
	    bkm = bk;
	    bk = bkp;
	  }

	return bk;
}
//Pellet heat deposition calculation
void HyperbolicLPSolver::calculateHeatDeposition() {
	const double* leftintegral = m_pParticleData->m_vLeftIntegral;
	const double* rightintegral = m_pParticleData->m_vRightIntegral;
	const double* volume = m_pParticleData->m_vVolume;
	double* Deltaq = m_pParticleData->m_vDeltaq;
	double* Qplusminus = m_pParticleData->m_vQplusminus;
	double masse = m_pParticleData->masse;
	double massNe = m_pParticleData->massNe;
	double teinf = m_pParticleData->teinf;
	double INe = m_pParticleData->INe;
	int ZNe = m_pParticleData->ZNe;
	double neinf = m_pParticleData->neinf;
	double heatK = m_pParticleData->heatK;
	double e = heatK*(2.99792458e7)/100;
        size_t fluidStartIndex = m_pParticleData->getFluidStartIndex();
        size_t ghostStartIndex = fluidStartIndex + m_pParticleData->getGhostStartIndex();

//previous used lnlambda
//	double lnLambda = log(2.5*teinf/(INe));

//new lnlambda
	double lnLambda = log(2.0*teinf/(9.0*ZNe*(1.0 + 1.8*pow(ZNe,-2./3.))));

//        #ifdef _OPENMP
//        #pragma omp parallel for
//        #endif
	for(size_t index=fluidStartIndex; index<ghostStartIndex; index++){
		double tauleft = leftintegral[index]/massNe*ZNe;
		double tauright = rightintegral[index]/massNe*ZNe;
		double tauinf = heatK*heatK*teinf*teinf/(8.0*3.1416*e*e*e*e*lnLambda);
		double taueff = tauinf*sqrt(2.0/(1.0+ZNe));
		double uleft = tauleft/taueff;
		double uright = tauright/taueff;
		double qinf=sqrt(2.0/3.1416/masse)*neinf*pow(heatK*teinf,1.5);
		double guleft = sqrt(uleft)*Bessel_K1(sqrt(uleft))/4;
		double guright = sqrt(uright)*Bessel_K1(sqrt(uright))/4;
		double nt=1.0/volume[index]/massNe;
//parallel line case
//		Deltaq[index] = qinf*nt/tauinf*(guleft+guright);
//		Qplusminus[index] = qinf*0.5*(uleft*Bessel_Kn(2,sqrt(uleft))+uright*Bessel_Kn(2,sqrt(uright)));
//spherical symmetry case
		Deltaq[index]=qinf*nt/tauinf*guleft;
		Qplusminus[index] = qinf*0.5*uleft*Bessel_Kn(2,sqrt(uleft));
	}

}

void HyperbolicLPSolver::updateStatesByLorentzForce() {
	if(m_iDimension==2)
		return;
        const double *positionX = m_pParticleData->m_vPositionX;
        const double *positionY = m_pParticleData->m_vPositionY;
        const double *positionZ = m_pParticleData->m_vPositionZ;	
        double *velocityU = m_pParticleData->m_vVelocityU;
        double *velocityV = m_pParticleData->m_vVelocityV;
        double *velocityW = m_pParticleData->m_vVelocityW;

	double Magneticfield=0.0;//placeholder

        size_t fluidStartIndex = m_pParticleData->getFluidStartIndex();
        size_t fluidEndIndex = fluidStartIndex + m_pParticleData->getFluidNum();

//      #ifdef _OPENMP
//      #pragma omp parallel for
//      #endif
        for(size_t index=fluidStartIndex; index<fluidEndIndex; index++){
		double y=positionY[index];
		double z=positionZ[index];
		double vy=velocityV[index];
		double vz=velocityW[index];
		double r=sqrt(y*y+z*z);
		if(r==0)
			continue;
		double vradial=vy*y/r+vz*z/r;
		double vtheta=vy*(-z)/r+vz*y/r;

		vradial=vradial+m_fDt*Magneticfield*vtheta;

		velocityV[index]=vradial*y/r+vtheta*(-z)/r;
		velocityW[index]=vradial*z/r+vtheta*y/r;
	}
}


void HyperbolicLPSolver::SPHDensityEstimatorForFluidParticle(int choice) {
	if(choice)
	        cout<<"-------HyperbolicLPSolver::SPHDensityEstimatorForFluidParticle()-------"<<endl;

        const double *positionX = m_pParticleData->m_vPositionX;
        const double *positionY = m_pParticleData->m_vPositionY;
        const double *positionZ = m_pParticleData->m_vPositionZ; // is all zero for the 2D case 
        const double *mass = m_pParticleData->m_vMass;
        double *VolumeVoronoi= m_pParticleData->m_vVolumeVoronoi;

        if(choice==2)
        {
                std::cout<<"Voronoi density estimator has been removed from LP code. SPH density estimator is used.\n"<<std::endl;
                choice=1;
        }
	if(choice==1||choice==3){
        m_pNeighbourSearcher->buildSearchStructure(positionX, positionY, positionZ, mass, VolumeVoronoi,
        m_pParticleData->m_iFluidStartIndex ,m_pParticleData->m_iFluidNum + m_pParticleData->m_iBoundaryNum);

//        double *localParSpacing = m_pParticleData->m_vLocalParSpacing;
        size_t fluidStartIndex = m_pParticleData->getFluidStartIndex();
        size_t fluidEndIndex = fluidStartIndex + m_pParticleData->getFluidNum();

        cout<<"Calculate density"<<endl;
        #ifdef _OPENMP
        #pragma omp parallel
        {

//        int tid = omp_get_thread_num();

        #endif

//        double neiListDist[maxNeiNum]; // a temp array for dist between a particle and its neighbours
//        size_t numNeiFound;

        if(m_pParticleData->m_vFluidBoundingBox.size() > 1) { // multiple fluid objects

                #ifdef _OPENMP
                #pragma omp for
                #endif
                for(size_t index=fluidStartIndex; index<fluidEndIndex; index++) {
                        if(choice==1){
                        double radius2multiplier=5.0;
//                        double radius2 = sqrt(m_pParticleData->m_vVolume[index])*m_fInitParticleSpacing*radius2multiplier;
                        double radius2 = radius2multiplier*sqrt(2*mass[index]*m_pParticleData->m_vVolumeOld[index]/1.7321);
                        double count_density=1.0/m_pParticleData->m_vVolumeOld[index];
                        m_pNeighbourSearcher->densityEstimator(
                                positionX[index],positionY[index],positionZ[index],radius2,
                                &count_density,m_pParticleData->m_vVolume_x[index],m_pParticleData->m_vVolume_y[index],m_pParticleData->m_vVolume_z[index]);
                        if(m_iDimension==2){
                                m_pParticleData->m_vVolume[index]=1.0/count_density;
                        }
                        else{//3d TODO
                        }
                        }
//end of sph density estimator
		}
	}
        else { // single fluid object
                #ifdef _OPENMP
                #pragma omp for
                #endif
//                for(size_t index=fluidStartIndex; index<fluidEndIndex+m_pParticleData->getInflowNum(); index++) {
                for(size_t index=fluidStartIndex; index<fluidEndIndex+m_pParticleData->getInflowNum(); index++) {
//sph density estimator
//			double tolerance=1;
//			double gra_mag=sqrt(m_pParticleData->m_vVolume_x[index]*m_pParticleData->m_vVolume_x[index]+m_pParticleData->m_vVolume_y[index]*m_pParticleData->m_vVolume_y[index]+m_pParticleData->m_vVolume_z[index]*m_pParticleData->m_vVolume_z[index]);
//			m_pParticleData->m_vIfSPHDensity[index]=0;
//                        if (choice==1||((choice==3)&&(gra_mag>tolerance))){
			double alpha=3.0;
			double max_gradient=3.0;
			double vmin=1e10;
			double vmax=0;
			double ratio=1.5;
			if(choice==1||choice==3){
//                        m_pParticleData->m_vIfSPHDensity[index]+=1;
                        double radius2multiplier;
                        if (m_iDimension==2)
                                radius2multiplier=5.0;
                        if(m_iDimension==3)
                                radius2multiplier=5.0;
			if(choice==3)
				radius2multiplier=2.0;
                        double gra_mag=sqrt(m_pParticleData->m_vVolume_x[index]*m_pParticleData->m_vVolume_x[index]+m_pParticleData->m_vVolume_y[index]*m_pParticleData->m_vVolume_y[index]+m_pParticleData->m_vVolume_z[index]*m_pParticleData->m_vVolume_z[index]);
                        double radius2;
                        if(m_iDimension==2) radius2 = radius2multiplier*sqrt(2*mass[index]*m_pParticleData->m_vVolumeOld[index]/1.7321);
                        if(m_iDimension==3) radius2 = radius2multiplier*cbrt(1.4142*mass[index]*m_pParticleData->m_vVolumeOld[index]);
//                      cout<<radius2<<endl;
                        double count_density=1.0/m_pParticleData->m_vVolumeOld[index];
                        if(m_iDimension==3) count_density=5.0*radius2/7.0*count_density;
			if(0)
	                        m_pNeighbourSearcher->densityEstimator(
        	                        index, positionX[index],positionY[index],positionZ[index],radius2,
                	                &count_density,m_pParticleData->m_vVolume_x[index],m_pParticleData->m_vVolume_y[index],m_pParticleData->m_vVolume_z[index]);
			if(choice==3 || choice==1)
                                m_pNeighbourSearcher->densityEstimator(
                                        index, positionX[index],positionY[index],positionZ[index],radius2,
                                        &count_density,m_pParticleData->m_vVolumeOld,&vmin,&vmax);
//                                        &count_density,m_pParticleData->m_vVolume_x[index],m_pParticleData->m_vVolume_y[index],m_pParticleData->m_vVolume_z[index]);
                        if(m_iDimension==2){
				if(choice==1)
	                                m_pParticleData->m_vVolume[index]=1.0/count_density;
				if(choice==3)
				{
					if(1|| vmax/vmin>ratio)
					{
                                                m_pParticleData->m_vIfSPHDensity[index]+=1;
                                                m_pParticleData->m_vVolume[index]=1.0/count_density;
					}
/*					if((m_pParticleData->m_vVolume[index]*count_density)>alpha||(m_pParticleData->m_vVolume[index]*count_density)<1.0/alpha)
					{
						m_pParticleData->m_vIfSPHDensity[index]+=1;
						m_pParticleData->m_vVolume[index]=1.0/count_density;
					}*/
/*					if(gra_mag>max_gradient)
                                        {
                                                m_pParticleData->m_vIfSPHDensity[index]+=1;
                                                m_pParticleData->m_vVolume[index]=1.0/count_density;
                                        }*/
				}	
                        }
                        else{//3d
				if(choice==1)
                                	m_pParticleData->m_vVolume[index]=5.0*radius2/7.0/count_density;
                                if(choice==3)
                                {
					count_density=count_density*7.0/5.0/radius2;
                                        if(1||vmax/vmin>ratio)
                                        {
                                                m_pParticleData->m_vIfSPHDensity[index]+=1;
                                                m_pParticleData->m_vVolume[index]=1.0/count_density;
                                        }
/*                                        if((m_pParticleData->m_vVolume[index]*count_density)>alpha||(m_pParticleData->m_vVolume[index]*count_density)<1.0/alpha)
                                        {
                                                m_pParticleData->m_vIfSPHDensity[index]+=1;
                                                m_pParticleData->m_vVolume[index]=1.0/count_density;
                                        }*/
/*                                        if(gra_mag>max_gradient)
                                        {
                                                m_pParticleData->m_vIfSPHDensity[index]+=1;
                                                m_pParticleData->m_vVolume[index]=1.0/count_density;
                                        }*/
                                }

                        }
                        }
//end of sph density estimator
                }
        }

        #ifdef _OPENMP
        }
        #endif
	}
	if(choice)
	        cout<<"-------END HyperbolicLPSolver::SPHDensityEstimatorForFluidParticle()-------"<<endl;

}

bool HyperbolicLPSolver::generateSolidBoundaryByMirrorParticles() {
	
	cout<<"-------HyperbolicLPSolver::generateSolidBoundaryByMirrorParticles()-------"<<endl;	

	size_t fluidStartIndex = m_pParticleData->m_iFluidStartIndex;
	size_t fluidEndIndex = m_pParticleData->m_iFluidStartIndex + m_pParticleData->m_iFluidNum;


	#ifdef _OPENMP
	size_t numThreads = min(omp_get_max_threads(), m_iNumThreads);
	vector<vector<double>> bX(numThreads);
	vector<vector<double>> bY(numThreads);
	vector<vector<double>> bZ(numThreads);
	vector<vector<double>> bPressure(numThreads);
	vector<vector<double>> bVx(numThreads);
	vector<vector<double>> bVy(numThreads);
	vector<vector<double>> bVz(numThreads);
	vector<vector<size_t>> fIndex(numThreads);// the corresponding fluid index of a mirror particle
/*        for(size_t p=0; p<m_pParticleData->m_vBoundaryObjTypes.size(); p++) {

                if(m_pParticleData->m_vBoundaryObjTypes[p]!="solid") continue;
                cout<<"m_pParticleData->m_vBoundaryObjTypes["<<p<<"]="<<m_pParticleData->m_vBoundaryObjTypes[p]<<endl;
	}*/

        for(size_t p=0; p<m_pParticleData->m_vBoundaryObjTypes.size(); p++) {
                if(m_pParticleData->m_vBoundaryObjTypes[p]!="outflow") continue;
//		std::cout<<"Outflow begin"<<p<<std::endl;
                m_pParticleData->m_vBoundaryObj[p]->UpdateInflowBoundary(m_pParticleData,m_pEOS,m_fDt,m_fInitParticleSpacing);
//                std::cout<<"Outflow end"<<std::endl;
        }

	for(size_t p=0; p<m_pParticleData->m_vBoundaryObjTypes.size(); p++) {
//		cout<<p<<" "<<m_pParticleData->m_vBoundaryObjTypes[p]<<endl;	
		if(m_pParticleData->m_vBoundaryObjTypes[p]!="inflow") continue;
		if(m_pParticleData->m_vBoundaryObj[p]->UpdateInflowBoundary(m_pParticleData,m_pEOS,m_fDt,m_fInitParticleSpacing))
		{
			std::cout<<"Error: too many inflow particles."<<std::endl;
			assert(0);
			return false;
		}
	}
	fluidEndIndex = m_pParticleData->m_iFluidStartIndex + m_pParticleData->m_iFluidNum;

	#pragma omp parallel  
	{
	
	int tid = omp_get_thread_num();
	#else
	vector<double> bX, bY, bZ, bPressure, bVx, bVy, bVz;
	vector<size_t> fIndex;
	#endif

	for(size_t p=0; p<m_pParticleData->m_vBoundaryObjTypes.size(); p++) {
//		cout<<p<<" "<<m_pParticleData->m_vBoundaryObjTypes[p]<<endl;	
		if(m_pParticleData->m_vBoundaryObjTypes[p]!="solid") continue;
//		cout<<"m_pParticleData->m_vBoundaryObjTypes["<<p<<"]="<<m_pParticleData->m_vBoundaryObjTypes[p]<<endl;
		if(m_iDimension==2) {
			#ifdef _OPENMP
			#pragma omp for
			#endif
			for(size_t index=fluidStartIndex; index<fluidEndIndex+m_pParticleData->m_iInflowNum; index++)  {
				#ifdef _OPENMP
				int num = m_pParticleData->m_vBoundaryObj[p]->operator()(
				m_pParticleData->m_vPositionX[index],
				m_pParticleData->m_vPositionY[index],
				0,
				m_pParticleData->m_vPressure[index],
				m_pParticleData->m_vVelocityU[index],
				m_pParticleData->m_vVelocityV[index],
				0,
				bX[tid],bY[tid],bZ[tid],bPressure[tid],bVx[tid],bVy[tid],bVz[tid]); 
				
				for(int k=0; k<num; k++) fIndex[tid].push_back(index);
				#else
				int num = m_pParticleData->m_vBoundaryObj[p]->operator()(
				m_pParticleData->m_vPositionX[index],
				m_pParticleData->m_vPositionY[index],
				0,
				m_pParticleData->m_vPressure[index],
				m_pParticleData->m_vVelocityU[index],
				m_pParticleData->m_vVelocityV[index],
				0,	
				bX,bY,bZ,bPressure,bVx,bVy,bVz); 
				for(int k=0; k<num; k++) fIndex.push_back(index);
				#endif
			}
		}
		else if(m_iDimension==3) {
			#ifdef _OPENMP
			#pragma omp for
			#endif
			for(size_t index=fluidStartIndex; index<fluidEndIndex+m_pParticleData->m_iInflowNum; index++)  {
				#ifdef _OPENMP
				int num = m_pParticleData->m_vBoundaryObj[p]->operator()(
				m_pParticleData->m_vPositionX[index],
				m_pParticleData->m_vPositionY[index],
				m_pParticleData->m_vPositionZ[index],
				m_pParticleData->m_vPressure[index],
				m_pParticleData->m_vVelocityU[index],
				m_pParticleData->m_vVelocityV[index],
				m_pParticleData->m_vVelocityW[index],
				bX[tid],bY[tid],bZ[tid],bPressure[tid],bVx[tid],bVy[tid],bVz[tid]);
				for(int k=0; k<num; k++) fIndex[tid].push_back(index);
				#else
				int num = m_pParticleData->m_vBoundaryObj[p]->operator()(
				m_pParticleData->m_vPositionX[index],
				m_pParticleData->m_vPositionY[index],
				m_pParticleData->m_vPositionZ[index],
				m_pParticleData->m_vPressure[index],
				m_pParticleData->m_vVelocityU[index],
				m_pParticleData->m_vVelocityV[index],
				m_pParticleData->m_vVelocityW[index],
				bX,bY,bZ,bPressure,bVx,bVy,bVz); 
				for(int k=0; k<num; k++) fIndex.push_back(index);
				#endif
			}
		}
	}

	#ifdef _OPENMP
	}
	#endif

//	cout<<"Start to put solid boundary particles into main arrays"<<endl;
	size_t boundaryIndex = m_pParticleData->getBoundaryStartIndex()+m_pParticleData->m_iInflowNum;
//	cout<<"boundaryIndex="<<boundaryIndex<<endl;
	

	#ifdef _OPENMP
	size_t sum = 0;
	for(size_t tid=0; tid<numThreads; tid++) 
		sum += bX[tid].size();
	if(boundaryIndex+sum > m_pParticleData->m_iCapacity) {
		cout<<m_pParticleData->m_iCapacity<<endl;
		cout<<fluidStartIndex<<" "<<fluidEndIndex<<endl;
		cout<<boundaryIndex<<" "<<sum<<endl;
		cout<<"Error: Not enough memory for solid boundary particles!!!"<<endl;
		assert(0);
		return false; // not enough space -> augment data array size!
	}	
	m_vMirrorIndex.resize(sum);

	if(m_iDimension==3) {
		size_t count = 0;
		for(size_t tid=0; tid<numThreads; tid++) {
			for(size_t j=0; j<bX[tid].size(); j++) {
				m_pParticleData->m_vPositionX[boundaryIndex] = bX[tid][j];
				m_pParticleData->m_vPositionY[boundaryIndex] = bY[tid][j];
				m_pParticleData->m_vPositionZ[boundaryIndex] = bZ[tid][j];
				m_pParticleData->m_vPressure[boundaryIndex] = bPressure[tid][j];
				m_pParticleData->m_vVelocityU[boundaryIndex] = bVx[tid][j];
				m_pParticleData->m_vVelocityV[boundaryIndex] = bVy[tid][j];
				m_pParticleData->m_vVelocityW[boundaryIndex] = bVz[tid][j];	

				m_pParticleData->m_vVolume[boundaryIndex] = m_pParticleData->m_vVolume[fIndex[tid][j]];
                                m_pParticleData->m_vVolumeOld[boundaryIndex] = m_pParticleData->m_vVolumeOld[fIndex[tid][j]];
                                m_pParticleData->m_vMass[boundaryIndex] = m_pParticleData->m_vMass[fIndex[tid][j]];

				boundaryIndex++;
				m_vMirrorIndex[count++] = fIndex[tid][j]; 
			}
		}
	}
	else if(m_iDimension==2) {
		size_t count = 0;
		for(size_t tid=0; tid<numThreads; tid++) {
			for(size_t j=0; j<bX[tid].size(); j++) {
				m_pParticleData->m_vPositionX[boundaryIndex] = bX[tid][j];
				m_pParticleData->m_vPositionY[boundaryIndex] = bY[tid][j];
				m_pParticleData->m_vPressure[boundaryIndex] = bPressure[tid][j];
				m_pParticleData->m_vVelocityU[boundaryIndex] = bVx[tid][j];
				m_pParticleData->m_vVelocityV[boundaryIndex] = bVy[tid][j];

                                m_pParticleData->m_vVolume[boundaryIndex] = m_pParticleData->m_vVolume[fIndex[tid][j]];
                                m_pParticleData->m_vVolumeOld[boundaryIndex] = m_pParticleData->m_vVolumeOld[fIndex[tid][j]];
                                m_pParticleData->m_vMass[boundaryIndex] = m_pParticleData->m_vMass[fIndex[tid][j]];

				boundaryIndex++;
				m_vMirrorIndex[count++] = fIndex[tid][j];
			}
		}
	}	
	#else	
	if(boundaryIndex+bX.size() > m_pParticleData->m_iCapacity) {
		cout<<"Not enough memory for solid boundary particles!!!"<<endl;
		return false; // not enough space -> augment data array size!
	}
	m_vMirrorIndex.resize(bX.size());
	assert(bX.size()==fIndex.size());

	if(m_iDimension==3) {
		
		for(size_t j=0; j<bX.size(); j++) {
			m_pParticleData->m_vPositionX[boundaryIndex] = bX[j];
			m_pParticleData->m_vPositionY[boundaryIndex] = bY[j];
			m_pParticleData->m_vPositionZ[boundaryIndex] = bZ[j];
			m_pParticleData->m_vPressure[boundaryIndex] = bPressure[j];
			m_pParticleData->m_vVelocityU[boundaryIndex] = bVx[j];
			m_pParticleData->m_vVelocityV[boundaryIndex] = bVy[j];
			m_pParticleData->m_vVelocityW[boundaryIndex] = bVz[j];

                        m_pParticleData->m_vVolume[boundaryIndex] = m_pParticleData->m_vVolume[fIndex[j]];
                        m_pParticleData->m_vVolumeOld[boundaryIndex] = m_pParticleData->m_vVolumeOld[fIndex[j]];
                        m_pParticleData->m_vMass[boundaryIndex] = m_pParticleData->m_vMass[fIndex[j]];

			boundaryIndex++;
			m_vMirrorIndex[j] = fIndex[j];
		}
		
	}
	else if(m_iDimension==2) {
		
		for(size_t j=0; j<bX.size(); j++) {
			m_pParticleData->m_vPositionX[boundaryIndex] = bX[j];
			m_pParticleData->m_vPositionY[boundaryIndex] = bY[j];	
			m_pParticleData->m_vPressure[boundaryIndex] = bPressure[j];
			m_pParticleData->m_vVelocityU[boundaryIndex] = bVx[j];
			m_pParticleData->m_vVelocityV[boundaryIndex] = bVy[j];
			
                        m_pParticleData->m_vVolume[boundaryIndex] = m_pParticleData->m_vVolume[fIndex[j]];
                        m_pParticleData->m_vVolumeOld[boundaryIndex] = m_pParticleData->m_vVolumeOld[fIndex[j]];
                        m_pParticleData->m_vMass[boundaryIndex] = m_pParticleData->m_vMass[fIndex[j]];

			boundaryIndex++;
			m_vMirrorIndex[j] = fIndex[j];
		}
		
	}
	#endif
	
	m_pParticleData->m_iBoundaryNum = boundaryIndex - m_pParticleData->m_iBoundaryStartIndex;
	m_pParticleData->m_iTotalNum = m_pParticleData->m_iFluidNum + 
								   m_pParticleData->m_iBoundaryNum;

	cout<<"Generated "<<m_pParticleData->m_iBoundaryNum<<" solid boundary particles"<<endl;
	m_pParticleData->setGhostStartIndex(m_pParticleData->m_iTotalNum);	

	cout<<"-------END HyperbolicLPSolver::generateSolidBoundaryByMirrorParticles()-------"<<endl;

	return true;
}


bool HyperbolicLPSolver::generatePeriodicBoundaryByMirrorParticles() {
	
	cout<<"-------HyperbolicLPSolver::generatePeriodicBoundaryByMirrorParticles()-------"<<endl;	

	size_t fluidStartIndex = m_pParticleData->m_iFluidStartIndex;
	size_t fluidEndIndex = m_pParticleData->m_iFluidStartIndex + m_pParticleData->m_iFluidNum;


	#ifdef _OPENMP
	size_t numThreads = min(omp_get_max_threads(), m_iNumThreads);
	vector<vector<double>> bX(numThreads);
	vector<vector<double>> bY(numThreads);
	vector<vector<double>> bZ(numThreads);
	vector<vector<double>> bPressure(numThreads);
	vector<vector<double>> bVx(numThreads);
	vector<vector<double>> bVy(numThreads);
	vector<vector<double>> bVz(numThreads);
	vector<vector<size_t>> fIndex(numThreads);// the corresponding fluid index of a mirror particle
/*        for(size_t p=0; p<m_pParticleData->m_vBoundaryObjTypes.size(); p++) {

                if(m_pParticleData->m_vBoundaryObjTypes[p]!="solid") continue;
                cout<<"m_pParticleData->m_vBoundaryObjTypes["<<p<<"]="<<m_pParticleData->m_vBoundaryObjTypes[p]<<endl;
	}*/
	#pragma omp parallel  
	{
	
	int tid = omp_get_thread_num();
	#else
	vector<double> bX, bY, bZ, bPressure, bVx, bVy, bVz;
	vector<size_t> fIndex;
	#endif

	for(size_t p=0; p<m_pParticleData->m_vBoundaryObjTypes.size(); p++) {
//		cout<<p<<" "<<m_pParticleData->m_vBoundaryObjTypes[p]<<endl;	
		if(m_pParticleData->m_vBoundaryObjTypes[p]!="periodic") continue;
//		cout<<"m_pParticleData->m_vBoundaryObjTypes["<<p<<"]="<<m_pParticleData->m_vBoundaryObjTypes[p]<<endl;
		if(m_iDimension==2) {
			#ifdef _OPENMP
			#pragma omp for
			#endif
			for(size_t index=fluidStartIndex; index<fluidEndIndex; index++)  {
				
				#ifdef _OPENMP
				int num = m_pParticleData->m_vBoundaryObj[p]->operator()(
				m_pParticleData->m_vPositionX[index],
				m_pParticleData->m_vPositionY[index],
				0,
				m_pParticleData->m_vPressure[index],
				m_pParticleData->m_vVelocityU[index],
				m_pParticleData->m_vVelocityV[index],
				0,
				bX[tid],bY[tid],bZ[tid],bPressure[tid],bVx[tid],bVy[tid],bVz[tid]); 

				if(num==-1)
				{
					m_pParticleData->m_vPositionX[index]=bX[tid].back();
					m_pParticleData->m_vPositionY[index]=bY[tid].back();
					bX[tid].pop_back();
					bY[tid].pop_back();
					num = m_pParticleData->m_vBoundaryObj[p]->operator()(
                                		m_pParticleData->m_vPositionX[index],
                                		m_pParticleData->m_vPositionY[index],
                                		0,
                                		m_pParticleData->m_vPressure[index],
                                		m_pParticleData->m_vVelocityU[index],
                                		m_pParticleData->m_vVelocityV[index],
                                		0,
                                		bX[tid],bY[tid],bZ[tid],bPressure[tid],bVx[tid],bVy[tid],bVz[tid]);
				}
				assert(num>=0);				
				for(int k=0; k<num; k++) fIndex[tid].push_back(index);
				#else
				int num = m_pParticleData->m_vBoundaryObj[p]->operator()(
				m_pParticleData->m_vPositionX[index],
				m_pParticleData->m_vPositionY[index],
				0,
				m_pParticleData->m_vPressure[index],
				m_pParticleData->m_vVelocityU[index],
				m_pParticleData->m_vVelocityV[index],
				0,	
				bX,bY,bZ,bPressure,bVx,bVy,bVz);

				if(num==-1)
				{
					m_pParticleData->m_vPositionX[index]=bX.back();
					m_pParticleData->m_vPositionY[index]=bY.back();
					bX.pop_back();
					bY.pop_back();
                                	num = m_pParticleData->m_vBoundaryObj[p]->operator()(
                                		m_pParticleData->m_vPositionX[index],
                                		m_pParticleData->m_vPositionY[index],
                                		0,
                                		m_pParticleData->m_vPressure[index],
                                		m_pParticleData->m_vVelocityU[index],
                                		m_pParticleData->m_vVelocityV[index],
                                		0,
                                		bX,bY,bZ,bPressure,bVx,bVy,bVz);
				}
				assert(num>=0);
				for(int k=0; k<num; k++) fIndex.push_back(index);
				#endif
			}
		}
		else if(m_iDimension==3) {
//TODO: implement 3D periodic boundary condition
			printf("3D periodic boundary condition has not beem implemented yet!\n");
			assert(false);
			#ifdef _OPENMP
			#pragma omp for
			#endif
			for(size_t index=fluidStartIndex; index<fluidEndIndex; index++)  {
				#ifdef _OPENMP
				int num = m_pParticleData->m_vBoundaryObj[p]->operator()(
				m_pParticleData->m_vPositionX[index],
				m_pParticleData->m_vPositionY[index],
				m_pParticleData->m_vPositionZ[index],
				m_pParticleData->m_vPressure[index],
				m_pParticleData->m_vVelocityU[index],
				m_pParticleData->m_vVelocityV[index],
				m_pParticleData->m_vVelocityW[index],
				bX[tid],bY[tid],bZ[tid],bPressure[tid],bVx[tid],bVy[tid],bVz[tid]);
				for(int k=0; k<num; k++) fIndex[tid].push_back(index);
				#else
				int num = m_pParticleData->m_vBoundaryObj[p]->operator()(
				m_pParticleData->m_vPositionX[index],
				m_pParticleData->m_vPositionY[index],
				m_pParticleData->m_vPositionZ[index],
				m_pParticleData->m_vPressure[index],
				m_pParticleData->m_vVelocityU[index],
				m_pParticleData->m_vVelocityV[index],
				m_pParticleData->m_vVelocityW[index],
				bX,bY,bZ,bPressure,bVx,bVy,bVz); 
				for(int k=0; k<num; k++) fIndex.push_back(index);
				#endif
			}
		}
	}

	#ifdef _OPENMP
	}
	#endif

//	cout<<"Start to put solid boundary particles into main arrays"<<endl;
	size_t boundaryIndex = m_pParticleData->getBoundaryStartIndex();
//	cout<<"boundaryIndex="<<boundaryIndex<<endl;
	

	#ifdef _OPENMP
	size_t sum = 0;
	for(size_t tid=0; tid<numThreads; tid++) 
		sum += bX[tid].size();
	if(boundaryIndex+sum > m_pParticleData->m_iCapacity) {
		cout<<m_pParticleData->m_iCapacity<<endl;
		cout<<fluidStartIndex<<" "<<fluidEndIndex<<endl;
		cout<<boundaryIndex<<" "<<sum<<endl;
		cout<<"Not enough memory for periodic boundary particles!!!"<<endl;
		return false; // not enough space -> augment data array size!
	}	
	m_vMirrorIndex.resize(sum);

	if(m_iDimension==3) {
		size_t count = 0;
		for(size_t tid=0; tid<numThreads; tid++) {
			for(size_t j=0; j<bX[tid].size(); j++) {
				m_pParticleData->m_vPositionX[boundaryIndex] = bX[tid][j];
				m_pParticleData->m_vPositionY[boundaryIndex] = bY[tid][j];
				m_pParticleData->m_vPositionZ[boundaryIndex] = bZ[tid][j];
				m_pParticleData->m_vPressure[boundaryIndex] = bPressure[tid][j];
				m_pParticleData->m_vVelocityU[boundaryIndex] = bVx[tid][j];
				m_pParticleData->m_vVelocityV[boundaryIndex] = bVy[tid][j];
				m_pParticleData->m_vVelocityW[boundaryIndex] = bVz[tid][j];	

				m_pParticleData->m_vVolume[boundaryIndex] = m_pParticleData->m_vVolume[fIndex[tid][j]];
                                m_pParticleData->m_vVolumeOld[boundaryIndex] = m_pParticleData->m_vVolumeOld[fIndex[tid][j]];
                                m_pParticleData->m_vMass[boundaryIndex] = m_pParticleData->m_vMass[fIndex[tid][j]];

				boundaryIndex++;
				m_vMirrorIndex[count++] = fIndex[tid][j]; 
			}
		}
	}
	else if(m_iDimension==2) {
		size_t count = 0;
		for(size_t tid=0; tid<numThreads; tid++) {
			for(size_t j=0; j<bX[tid].size(); j++) {
				m_pParticleData->m_vPositionX[boundaryIndex] = bX[tid][j];
				m_pParticleData->m_vPositionY[boundaryIndex] = bY[tid][j];
				m_pParticleData->m_vPressure[boundaryIndex] = bPressure[tid][j];
				m_pParticleData->m_vVelocityU[boundaryIndex] = bVx[tid][j];
				m_pParticleData->m_vVelocityV[boundaryIndex] = bVy[tid][j];

                                m_pParticleData->m_vVolume[boundaryIndex] = m_pParticleData->m_vVolume[fIndex[tid][j]];
                                m_pParticleData->m_vVolumeOld[boundaryIndex] = m_pParticleData->m_vVolumeOld[fIndex[tid][j]];
                                m_pParticleData->m_vMass[boundaryIndex] = m_pParticleData->m_vMass[fIndex[tid][j]];

				boundaryIndex++;
				m_vMirrorIndex[count++] = fIndex[tid][j];
			}
		}
	}	
	#else	
	if(boundaryIndex+bX.size() > m_pParticleData->m_iCapacity) {
		cout<<"Not enough memory for periodic boundary particles!!!"<<endl;
		return false; // not enough space -> augment data array size!
	}
	m_vMirrorIndex.resize(bX.size());
	assert(bX.size()==fIndex.size());

	if(m_iDimension==3) {
		
		for(size_t j=0; j<bX.size(); j++) {
			m_pParticleData->m_vPositionX[boundaryIndex] = bX[j];
			m_pParticleData->m_vPositionY[boundaryIndex] = bY[j];
			m_pParticleData->m_vPositionZ[boundaryIndex] = bZ[j];
			m_pParticleData->m_vPressure[boundaryIndex] = bPressure[j];
			m_pParticleData->m_vVelocityU[boundaryIndex] = bVx[j];
			m_pParticleData->m_vVelocityV[boundaryIndex] = bVy[j];
			m_pParticleData->m_vVelocityW[boundaryIndex] = bVz[j];

                        m_pParticleData->m_vVolume[boundaryIndex] = m_pParticleData->m_vVolume[fIndex[j]];
                        m_pParticleData->m_vVolumeOld[boundaryIndex] = m_pParticleData->m_vVolumeOld[fIndex[j]];
                        m_pParticleData->m_vMass[boundaryIndex] = m_pParticleData->m_vMass[fIndex[j]];

			boundaryIndex++;
			m_vMirrorIndex[j] = fIndex[j];
		}
		
	}
	else if(m_iDimension==2) {
		
		for(size_t j=0; j<bX.size(); j++) {
			m_pParticleData->m_vPositionX[boundaryIndex] = bX[j];
			m_pParticleData->m_vPositionY[boundaryIndex] = bY[j];	
			m_pParticleData->m_vPressure[boundaryIndex] = bPressure[j];
			m_pParticleData->m_vVelocityU[boundaryIndex] = bVx[j];
			m_pParticleData->m_vVelocityV[boundaryIndex] = bVy[j];
			
                        m_pParticleData->m_vVolume[boundaryIndex] = m_pParticleData->m_vVolume[fIndex[j]];
                        m_pParticleData->m_vVolumeOld[boundaryIndex] = m_pParticleData->m_vVolumeOld[fIndex[j]];
                        m_pParticleData->m_vMass[boundaryIndex] = m_pParticleData->m_vMass[fIndex[j]];

			boundaryIndex++;
			m_vMirrorIndex[j] = fIndex[j];
		}
		
	}
	#endif
	
	m_pParticleData->m_iBoundaryNum = boundaryIndex - m_pParticleData->m_iBoundaryStartIndex;
	m_pParticleData->m_iTotalNum = m_pParticleData->m_iFluidNum + 
								   m_pParticleData->m_iBoundaryNum;

	cout<<"Generated "<<m_pParticleData->m_iBoundaryNum<<" periodic boundary particles"<<endl;
	m_pParticleData->setGhostStartIndex(m_pParticleData->m_iTotalNum);	

	cout<<"-------END HyperbolicLPSolver::generatePeriodicBoundaryByMirrorParticles()-------"<<endl;

	return true;
}

bool HyperbolicLPSolver::generateGhostParticleByFillingVacancy() {
	
	cout<<"-------HyperbolicLPSolver::generateGhostParticleByFillingVacancy()-------"<<endl;

	double *x = m_pParticleData->m_vPositionX;
	double *y = m_pParticleData->m_vPositionY;
	double *z = m_pParticleData->m_vPositionZ;
	//int *neighbourList = m_pParticleData->m_vNeighbourList;
	int *neighbourListSize = m_pParticleData->m_vNeighbourListSize;
	size_t ghostIndex = m_pParticleData->getGhostStartIndex();
//	cout<<"ghostStartIndex="<<ghostIndex<<endl;
	size_t fluidStartIndex = m_pParticleData->m_iFluidStartIndex;
	size_t fluidEndIndex = m_pParticleData->m_iFluidStartIndex + m_pParticleData->m_iFluidNum;

	#ifdef _OPENMP
	size_t numThreads = min(omp_get_max_threads(), m_iNumThreads);
//	cout<<"numThreads="<<numThreads<<endl;
//	cout<<"omp_get_max_threads()="<<omp_get_max_threads()<<endl;
	vector<vector<size_t>> fIndex(numThreads);
	vector<vector<double>> gX(numThreads);
	vector<vector<double>> gY(numThreads);
	vector<vector<double>> gZ;
	if(m_iDimension == 3) gZ = vector<vector<double>>(numThreads);
	
	#pragma omp parallel  
	{
	
	int tid = omp_get_thread_num();
	
	#endif


//	#ifdef _OPENMP
//	#pragma omp parallel  
//	{	
//	int tid = omp_get_thread_num();		
//	#endif

	if(m_iDimension==2) {
	
		#ifdef _OPENMP
		#pragma omp for
		#endif
		for(size_t index=fluidStartIndex; index<fluidEndIndex; index++)  {
//			cout<<index<<endl;			
//			if(!m_vFillGhost[index]) continue;
			m_vFillGhost[index]=false;
			if(m_pParticleData->m_vObjectTag[index]<0) continue; // fluid particle in contact region

			double x0 = x[index], y0 = y[index];
			size_t neiListStartIndex = index*m_pParticleData->m_iMaxNeighbourNum;
			
			int count[8] = {0};
			
//			cout<<"index="<<index<<endl;
//			cout<<"neighbourListSize="<<neighbourListSize[index]<<endl;
			size_t other = 0;	
			for(int i=0; i<neighbourListSize[index]; i++) {
				
				size_t neiIndex = m_pParticleData->m_vNeighbourList[neiListStartIndex+i];
				
				double x1 = x[neiIndex], y1 = y[neiIndex];
				double dx = x1-x0, dy = y1-y0;

				if(dy>0 && dy<dx)       count[0]++;
				else if(dx>0 && dy>dx)  count[1]++;
				else if(dx<0 && dy>-dx) count[2]++;
				else if(dy>0 && dy<-dx) count[3]++;
				else if(dy<0 && dy>dx)  count[4]++;
				else if(dx<0 && dy<dx)  count[5]++;
				else if(dx>0 && dy<-dx) count[6]++;
				else if(dy<0 && dy>-dx) count[7]++;
				else other++;	
			}
			for(int i=0; i<8; i++) {
//				cout<<"count["<<i<<"]="<<count[i]<<endl;
				if(count[i]<8) {
					m_vFillGhost[index] = true;
					#ifdef _OPENMP
					fillGhostParticle2D(i,count,index,gX[tid],gY[tid],fIndex[tid]);
					#else
					bool b = fillGhostParticle2D(i,count,index,&ghostIndex);
					if(!b) return false;
					#endif
				}
			}
//			cout<<"other="<<other<<endl;

		}
		
		
	}
	else if(m_iDimension==3) {
		
		#ifdef _OPENMP
		#pragma omp for
		#endif
		for(size_t index=fluidStartIndex; index<fluidEndIndex; index++) {
//			if(!m_vFillGhost[index]) continue;
			m_vFillGhost[index]=false;
			if(m_pParticleData->m_vObjectTag[index]<0) continue; // fluid particle in contact region

			double x0 = x[index], y0 = y[index], z0 = z[index];
			size_t neiListStartIndex = index*m_pParticleData->m_iMaxNeighbourNum;
			
			int count[6] = {0};
			
//			cout<<"index="<<index<<endl;
//			cout<<"neighbourListSize="<<neighbourListSize[index]<<endl;
			size_t other = 0;	
			for(int i=0; i<neighbourListSize[index]; i++) {
				
				size_t neiIndex = m_pParticleData->m_vNeighbourList[neiListStartIndex+i];
				
				double x1 = x[neiIndex], y1 = y[neiIndex], z1 = z[neiIndex];
				double dx = x1-x0, dy = y1-y0, dz = z1-z0;
				if(dx>fabs(dy) && dx>fabs(dz)) count[0]++;
				else if(-dx>fabs(dy) && -dx>fabs(dz)) count[1]++;
				else if(dy>fabs(dx) && dy>fabs(dz)) count[2]++;
                                else if(-dy>fabs(dx) && -dy>fabs(dz)) count[3]++;
                                else if(dz>fabs(dx) && dz>fabs(dy)) count[4]++;
                                else if(-dz>fabs(dx) && -dz>fabs(dy)) count[5]++;
				else other++;
			}
			for(int i=0; i<6; i++) {
				//cout<<"count["<<i<<"]="<<count[i]<<endl;
				if(count[i]==0) {//=8 FOR POWDER TARGET
					m_vFillGhost[index] = true;
					#ifdef _OPENMP
					fillGhostParticle3D(i,count,index,gX[tid],gY[tid],gZ[tid],fIndex[tid]);
					#else
					bool b = fillGhostParticle3D(i,count,index,&ghostIndex);
					printf("Error: signle thread code for 3D ghost particles are old version and may produce inaccurate result.\n");assert(false);//TODO
					if(!b) return false;
					#endif
				}
			}
			//cout<<"other="<<other<<endl;

		}	
	
	
	}
	

	#ifdef _OPENMP
	}
	#endif
	
	cout<<"Start to fill ghost particles"<<endl;

	#ifdef _OPENMP
	size_t sum = 0;
	for(size_t tid=0; tid<numThreads; tid++) 
		sum += gX[tid].size();
	if(ghostIndex+sum > m_pParticleData->m_iCapacity) return false; // not enough space -> augment data array size!
	
	double *pressure = m_pParticleData->m_vPressure;
	double *velocityU = m_pParticleData->m_vVelocityU;
	double *velocityV = m_pParticleData->m_vVelocityV;
	double *velocityW;
	if(m_iDimension==3) velocityW = m_pParticleData->m_vVelocityW;
	double *localParSpacing = m_pParticleData->m_vLocalParSpacing;

	if(m_iDimension==3) {
		for(size_t tid=0; tid<numThreads; tid++) {
			for(size_t j=0; j<gX[tid].size(); j++) {
				x[ghostIndex] = gX[tid][j];
				y[ghostIndex] = gY[tid][j];
				z[ghostIndex] = gZ[tid][j];
				size_t index = fIndex[tid][j];

				pressure[ghostIndex] = 0;
				velocityU[ghostIndex] = velocityU[index];
				velocityV[ghostIndex] = velocityV[index];
				velocityW[ghostIndex] = velocityW[index];
				localParSpacing[ghostIndex] = localParSpacing[index];
				m_pParticleData->m_vMass[ghostIndex] = 0;
				m_pParticleData->m_vVolume[ghostIndex] = 1.0e5;
                                m_pParticleData->m_vVolumeOld[ghostIndex] = 1.0e5;
				size_t neiListStartIndex = index*m_pParticleData->m_iMaxNeighbourNum;
				size_t sz = m_pParticleData->m_vNeighbourListSize[index];
				if(sz<m_pParticleData->m_iMaxNeighbourNum){
					m_pParticleData->m_vNeighbourList[neiListStartIndex+sz]=ghostIndex;
					m_pParticleData->m_vNeighbourListSize[index]++;
					ghostIndex++;
				}
				else{
					std::cout<<"Too many ghost neighbours for particle ("<<x[index]<<" "<<y[index]<<" "<<z[index]<<")."<<std::endl;
				}
			}
		}
	}
	else if(m_iDimension==2) {
		for(size_t tid=0; tid<numThreads; tid++) {
			for(size_t j=0; j<gX[tid].size(); j++) {
				x[ghostIndex] = gX[tid][j];
				y[ghostIndex] = gY[tid][j];
				size_t index = fIndex[tid][j];
				
				pressure[ghostIndex] = 0;
				velocityU[ghostIndex] = velocityU[index];
				velocityV[ghostIndex] = velocityV[index];	
				localParSpacing[ghostIndex] = localParSpacing[index];
                                m_pParticleData->m_vMass[ghostIndex] = 0;
                                m_pParticleData->m_vVolume[ghostIndex] = 1.0e5;
                                m_pParticleData->m_vVolumeOld[ghostIndex] = 1.0e5;
				size_t neiListStartIndex = index*m_pParticleData->m_iMaxNeighbourNum;
				size_t sz = m_pParticleData->m_vNeighbourListSize[index];
				if(sz<m_pParticleData->m_iMaxNeighbourNum){
					m_pParticleData->m_vNeighbourList[neiListStartIndex+sz]=ghostIndex;
					m_pParticleData->m_vNeighbourListSize[index]++;
					ghostIndex++;
				}
				else{
					std::cout<<"Too many ghost neighbours for particle ("<<x[index]<<" "<<y[index]<<" "<<z[index]<<")."<<std::endl;
				}
			}
		}
	}	
	#endif
	
	m_pParticleData->m_iGhostNum = ghostIndex - m_pParticleData->m_iGhostStartIndex;
	m_pParticleData->m_iTotalNum = m_pParticleData->m_iFluidNum + 
								   m_pParticleData->m_iBoundaryNum + 
								   m_pParticleData->m_iGhostNum;

	cout<<"Generated "<<m_pParticleData->m_iGhostNum<<" ghost particles"<<endl;
        if(m_pParticleData->m_iTotalNum > m_pParticleData->m_iCapacity) {
                cout<<m_pParticleData->m_iTotalNum<<" "<<m_pParticleData->m_iCapacity<<endl;
                cout<<"Error: Not enough memory for vacuum particles!!!"<<endl;
		assert(0);
                return false; // not enough space -> augment data array size!
        }
	cout<<"-------END HyperbolicLPSolver::generateGhostParticleByFillingVacancy()-------"<<endl;
	return true;	
}

#ifdef _OPENMP
void HyperbolicLPSolver::fillGhostParticle2D(int dir, int count[], size_t index, 
vector<double>& gX, vector<double>& gY, vector<size_t>& fIndex) {
	
	double delta = m_pParticleData->m_vLocalParSpacing[index];
	double r = m_fTimesNeiSearchRadius * delta;
	int k = m_fTimesNeiSearchRadius+1;
	
	if(dir==0 || dir==3 || dir==4 || dir==7) {
		bool repeat = false;
		if(dir==0 && count[1]==0) repeat = true;
		else if(dir==3 && count[2]==0)repeat = true;
		else if(dir==4 && count[5]==0) repeat = true;
		else if(dir==7 && count[6]==0) repeat = true;
		for(int i=1; i<=k; i++) {
			int limit = (repeat==true)? i-1:i;
			for(int j=1; j<=limit; j++) {
				if(((i*i+j*j)*delta*delta <= r*r)&&((i*i+j*j)>0)) {
					double x=0, y=0;
					if(dir==0 || dir==7)
						x = m_pParticleData->m_vPositionX[index] + i*delta;
					else
						x = m_pParticleData->m_vPositionX[index] - i*delta;
					if(dir==0 || dir==3)
						y = m_pParticleData->m_vPositionY[index] + j*delta;
					else
						y = m_pParticleData->m_vPositionY[index] - j*delta;
					
					gX.push_back(x);
					gY.push_back(y);
					fIndex.push_back(index);	
				}
			}
		}
	}
	else {
		for(int j=1; j<=k; j++) {
			for(int i=1; i<=j; i++) {
				if((i*i+j*j)*delta*delta <= r*r) {
					double x=0, y=0;
					if(dir==1 || dir==6)
						x = m_pParticleData->m_vPositionX[index] + i*delta;
					else
						x = m_pParticleData->m_vPositionX[index] - i*delta;
					if(dir==1 || dir==2)
						y = m_pParticleData->m_vPositionY[index] + j*delta;
					else
						y = m_pParticleData->m_vPositionY[index] - j*delta;

					gX.push_back(x);
					gY.push_back(y);
					fIndex.push_back(index);	
				}
			}
		}		
	}
	
	//cout<<"index="<<index<<" num ghost="<<*ghostIndex-saveg<<endl;
	//cout<<"ghostIndex="<<*ghostIndex<<endl;

}
#else
bool HyperbolicLPSolver::fillGhostParticle2D(int dir, int count[], size_t index, size_t* ghostIndex) {

	double delta = m_pParticleData->m_vLocalParSpacing[index];
	double r = m_fTimesNeiSearchRadius * delta;
	int k = m_fTimesNeiSearchRadius;

	size_t neiListStartIndex = index*m_pParticleData->m_iMaxNeighbourNum;
	if(dir==0 || dir==3 || dir==4 || dir==7) {
		bool repeat = false;
		if(dir==0 && count[1]==0) repeat = true;
		else if(dir==3 && count[2]==0)repeat = true;
		else if(dir==4 && count[5]==0) repeat = true;
		else if(dir==7 && count[6]==0) repeat = true;
		for(int i=1; i<=k; i++) {
			int limit = (repeat==true)? i-1:i;
			for(int j=1; j<=limit; j++) {
				if((i*i+j*j)*delta*delta <= r*r) {
					double x=0, y=0;
					if(dir==0 || dir==7)
						x = m_pParticleData->m_vPositionX[index] + i*delta;
					else
						x = m_pParticleData->m_vPositionX[index] - i*delta;
					if(dir==0 || dir==3)
						y = m_pParticleData->m_vPositionY[index] + j*delta;
					else
						y = m_pParticleData->m_vPositionY[index] - j*delta;

					if(*ghostIndex>=m_pParticleData->m_iCapacity) return false;
					
					m_pParticleData->m_vPositionX[*ghostIndex] = x;
					m_pParticleData->m_vPositionY[*ghostIndex] = y;
					
					m_pParticleData->m_vPressure[*ghostIndex] = 0;
					m_pParticleData->m_vVelocityU[*ghostIndex] = m_pParticleData->m_vVelocityU[index];
					m_pParticleData->m_vVelocityV[*ghostIndex] = m_pParticleData->m_vVelocityV[index];	
					m_pParticleData->m_vLocalParSpacing[*ghostIndex] = m_pParticleData->m_vLocalParSpacing[index];

					int sz = m_pParticleData->m_vNeighbourListSize[index];
					m_pParticleData->m_vNeighbourList[neiListStartIndex+sz]=(*ghostIndex);
					m_pParticleData->m_vNeighbourListSize[index]++;
					(*ghostIndex)++;
					
				}
			}
		}
	}
	else {
		for(int j=1; j<=k; j++) {
			for(int i=1; i<=j; i++) {
				if((i*i+j*j)*delta*delta <= r*r) {
					double x=0, y=0;
					if(dir==1 || dir==6)
						x = m_pParticleData->m_vPositionX[index] + i*delta;
					else
						x = m_pParticleData->m_vPositionX[index] - i*delta;
					if(dir==1 || dir==2)
						y = m_pParticleData->m_vPositionY[index] + j*delta;
					else
						y = m_pParticleData->m_vPositionY[index] - j*delta;
					
					if(*ghostIndex>=m_pParticleData->m_iCapacity) return false;

					m_pParticleData->m_vPositionX[*ghostIndex] = x;
					m_pParticleData->m_vPositionY[*ghostIndex] = y;
					
					m_pParticleData->m_vPressure[*ghostIndex] = 0;
					m_pParticleData->m_vVelocityU[*ghostIndex] = m_pParticleData->m_vVelocityU[index];
					m_pParticleData->m_vVelocityV[*ghostIndex] = m_pParticleData->m_vVelocityV[index];	
					m_pParticleData->m_vLocalParSpacing[*ghostIndex] = m_pParticleData->m_vLocalParSpacing[index];

					int sz = m_pParticleData->m_vNeighbourListSize[index];
					m_pParticleData->m_vNeighbourList[neiListStartIndex+sz]=(*ghostIndex);
					m_pParticleData->m_vNeighbourListSize[index]++;
					(*ghostIndex)++;
					
				}
			}
		}		
	}
	
	return true;
	//cout<<"index="<<index<<" num ghost="<<*ghostIndex-saveg<<endl;
	//cout<<"ghostIndex="<<*ghostIndex<<endl;

}
#endif



#ifdef _OPENMP
void HyperbolicLPSolver::fillGhostParticle3D(int dir, int count[], size_t index, 
vector<double>& gX, vector<double>& gY, vector<double>& gZ, vector<size_t>& fIndex) {
	
	double delta = m_pParticleData->m_vLocalParSpacing[index];
	double r = (m_fTimesNeiSearchRadius+1) * delta;
//	int k = m_fTimesNeiSearchRadius+1;
	int k=3;
	int min=0;//=8 for powder target

	if(dir==0 || dir==1){
                for(int i=1; i<=k; i++) {
                        for(int j=-i+(count[3]>=min); j<=i-(count[2]>=min); j++) {
                                for(int l=-i+(count[5]>=min); l<=i-(count[4]>=min); l++) {
                                        if((i*i+j*j+l*l)*delta*delta <= r*r) {
                                                double x=0, y=0, z=0;
                                                if(dir==0)
                                                        x = m_pParticleData->m_vPositionX[index] + i*delta;
                                                else
                                                        x = m_pParticleData->m_vPositionX[index] - i*delta;
                                                y = m_pParticleData->m_vPositionY[index] + j*delta;
                                                z = m_pParticleData->m_vPositionZ[index] + l*delta;
                                                gX.push_back(x);
                                                gY.push_back(y);
                                                gZ.push_back(z);
                                                fIndex.push_back(index);
                                        }
                                }
                        }
                }
	}

        if(dir==2 || dir==3){
                for(int i=1; i<=k; i++) {
                        for(int j=-i+(count[1]>=min); j<=i-(count[0]>=min); j++) {
                                for(int l=-i+(count[5]>=min); l<=i-(count[4]>=min); l++) {
                                        if((i*i+j*j+l*l)*delta*delta <= r*r) {
                                                double x=0, y=0, z=0;
                                                if(dir==2)
                                                        y = m_pParticleData->m_vPositionY[index] + i*delta;
                                                else    
                                                        y = m_pParticleData->m_vPositionY[index] - i*delta;
                                                x = m_pParticleData->m_vPositionX[index] + j*delta;
                                                z = m_pParticleData->m_vPositionZ[index] + l*delta;
                                                gX.push_back(x);
                                                gY.push_back(y);
                                                gZ.push_back(z);
                                                fIndex.push_back(index);
                                        }
                                }
                        }
                }
        }

        if(dir==4 || dir==5){
                for(int i=1; i<=k; i++) {
                        for(int j=-i+(count[1]>=min); j<=i-(count[0]>=min); j++) {
                                for(int l=-i+(count[3]>=min); l<=i-(count[2]>=min); l++) {
                                        if((i*i+j*j+l*l)*delta*delta <= r*r) {
                                                double x=0, y=0, z=0;
                                                if(dir==4)
                                                        z = m_pParticleData->m_vPositionZ[index] + i*delta;
                                                else    
                                                        z = m_pParticleData->m_vPositionZ[index] - i*delta;
                                                x = m_pParticleData->m_vPositionX[index] + j*delta;
                                                y = m_pParticleData->m_vPositionY[index] + l*delta;
                                                gX.push_back(x);
                                                gY.push_back(y);
                                                gZ.push_back(z);
                                                fIndex.push_back(index);
                                        }
                                }
                        }
                }
        }

	//cout<<"index="<<index<<" num ghost="<<*ghostIndex-saveg<<endl;
	//cout<<"ghostIndex="<<*ghostIndex<<endl;

}
#else
bool HyperbolicLPSolver::fillGhostParticle3D(int dir, int count[], size_t index, size_t* ghostIndex) {
//	cout<<"ghostIndex="<<*ghostIndex<<endl;	

	double delta = m_pParticleData->m_vLocalParSpacing[index];
	double r = m_fTimesNeiSearchRadius * delta;
//	int k = m_fTimesNeiSearchRadius+1;
	int k = 2;

	size_t neiListStartIndex = index*m_pParticleData->m_iMaxNeighbourNum;
	if(dir==0 || dir==3 || dir==4 || dir==7 || dir==8 || dir==11 || dir==12 || dir==15) {
		bool repeat = false;
		if(dir==0 && count[1]==0) repeat = true;
		else if(dir==3 && count[2]==0)   repeat = true;
		else if(dir==4 && count[5]==0)   repeat = true;
		else if(dir==7 && count[6]==0)   repeat = true;
		else if(dir==8 && count[9]==0)   repeat = true;
		else if(dir==11 && count[10]==0) repeat = true;
		else if(dir==12 && count[13]==0) repeat = true;
		else if(dir==15 && count[14]==0) repeat = true;
		for(int i=1; i<=k; i++) {
			int limit = (repeat==true)? i-1:i;
			for(int j=1; j<=limit; j++) {
				for(int l=1; l<=k; l++) {
					if((i*i+j*j+l*l)*delta*delta <= r*r) {
						double x=0, y=0, z=0;
						if(dir==0 || dir==7 || dir==8 || dir==15)
							x = m_pParticleData->m_vPositionX[index] + i*delta;
						else
							x = m_pParticleData->m_vPositionX[index] - i*delta;
						if(dir==0 || dir==3 || dir==8 || dir==11)
							y = m_pParticleData->m_vPositionY[index] + j*delta;
						else
							y = m_pParticleData->m_vPositionY[index] - j*delta;
						if(dir==0 || dir==3 || dir==4 || dir==7)
							z = m_pParticleData->m_vPositionZ[index] + l*delta;
						else
							z = m_pParticleData->m_vPositionZ[index] - l*delta;

						if(*ghostIndex>=m_pParticleData->m_iCapacity) return false;
						
						m_pParticleData->m_vPositionX[*ghostIndex] = x;
						m_pParticleData->m_vPositionY[*ghostIndex] = y;
						m_pParticleData->m_vPositionZ[*ghostIndex] = z;

						m_pParticleData->m_vPressure[*ghostIndex] = m_pParticleData->m_vPressure[index];
						m_pParticleData->m_vVelocityU[*ghostIndex] = m_pParticleData->m_vVelocityU[index];
						m_pParticleData->m_vVelocityV[*ghostIndex] = m_pParticleData->m_vVelocityV[index];	
						m_pParticleData->m_vVelocityW[*ghostIndex] = m_pParticleData->m_vVelocityW[index];
						m_pParticleData->m_vLocalParSpacing[*ghostIndex] = m_pParticleData->m_vLocalParSpacing[index];

						int sz = m_pParticleData->m_vNeighbourListSize[index];
						m_pParticleData->m_vNeighbourList[neiListStartIndex+sz]=(*ghostIndex);
						m_pParticleData->m_vNeighbourListSize[index]++;
						(*ghostIndex)++;
					}	
				}
			}
		}
	}
	else {
		for(int j=1; j<=k; j++) {
			for(int i=1; i<=j; i++) {
				for(int l=1; l<=k; l++) {
					if((i*i+j*j+l*l)*delta*delta <= r*r) {
						double x=0, y=0, z=0;
						if(dir==1 || dir==6 || dir==9 || dir==14)
							x = m_pParticleData->m_vPositionX[index] + i*delta;
						else
							x = m_pParticleData->m_vPositionX[index] - i*delta;
						if(dir==1 || dir==2 || dir==9 || dir==10)
							y = m_pParticleData->m_vPositionY[index] + j*delta;
						else
							y = m_pParticleData->m_vPositionY[index] - j*delta;
						if(dir==1 || dir==2 || dir==5 || dir==6)
							z = m_pParticleData->m_vPositionZ[index] + l*delta;
						else
							z = m_pParticleData->m_vPositionZ[index] - l*delta;
					
						if(*ghostIndex>=m_pParticleData->m_iCapacity) return false;

						m_pParticleData->m_vPositionX[*ghostIndex] = x;
						m_pParticleData->m_vPositionY[*ghostIndex] = y;
						m_pParticleData->m_vPositionZ[*ghostIndex] = z;

						m_pParticleData->m_vPressure[*ghostIndex] = 0;
						m_pParticleData->m_vVelocityU[*ghostIndex] = m_pParticleData->m_vVelocityU[index];
						m_pParticleData->m_vVelocityV[*ghostIndex] = m_pParticleData->m_vVelocityV[index];	
						m_pParticleData->m_vVelocityW[*ghostIndex] = m_pParticleData->m_vVelocityW[index];
						m_pParticleData->m_vLocalParSpacing[*ghostIndex] = m_pParticleData->m_vLocalParSpacing[index];

						int sz = m_pParticleData->m_vNeighbourListSize[index];
						m_pParticleData->m_vNeighbourList[neiListStartIndex+sz]=(*ghostIndex);
						m_pParticleData->m_vNeighbourListSize[index]++;
						(*ghostIndex)++;
					}	
				}
			}
		}		
	}
	
	return true;
	//cout<<"index="<<index<<" num ghost="<<*ghostIndex-saveg<<endl;
	//cout<<"ghostIndex="<<*ghostIndex<<endl;

}
#endif


void setListInOneDir2D(size_t neiIndex, double a0, double a1, double b0, double b1,
                                           int* arr1, size_t& n1, int* arr2, size_t& n2,
                                           int* arr3, size_t& n3, int* arr4, size_t& n4) {

if((5*fabs(a1-a0))>=fabs(b1-b0)){
        if(a1 > a0) {
                //printf("a1 > a0\n");
                //printf("a1=%.16g\n",a1);
                //printf("a0=%.16g\n",a0);
                if(b1 >= b0) arr1[n1++]=neiIndex;
                else arr2[n2++]=neiIndex;
        }
        else if(a1 < a0) {
                //printf("a1 < a0\n");
                //printf("a1=%.16g\n",a1);
                //printf("a0=%.16g\n",a0);
                if(b1 >= b0) arr3[n3++]=neiIndex;
                else arr4[n4++]=neiIndex;
        }
}
}

// a1>a0  1 2 3 4
// a1<a0  5 6 7 8
// b1>=b0 1 2 5 6
// b1<b0  3 4 7 8
// c1>=c0 1 3 5 7
// c1<c0  2 4 6 8
void setListInOneDir3D(size_t neiIndex, double a0, double a1, double b0, double b1, double c0, double c1,
					   int* arr1, size_t& n1, int* arr2, size_t& n2,
					   int* arr3, size_t& n3, int* arr4, size_t& n4,
					   int* arr5, size_t& n5, int* arr6, size_t& n6,
					   int* arr7, size_t& n7, int* arr8, size_t& n8) {
if(((5*fabs(a1-a0))>=fabs(b1-b0))&&((5*fabs(a1-a0))>=fabs(c1-c0))){
	
	if(a1 > a0) { 
		if(b1 >= b0) {
			if(c1 >= c0) arr1[n1++]=neiIndex;
			else arr2[n2++]=neiIndex;
		}
		else {
			if(c1 >= c0) arr3[n3++]=neiIndex;
			else arr4[n4++]=neiIndex;
		}
	
	}
	else if(a1 < a0) { 
		if(b1 >= b0) {
			if(c1 >= c0) arr5[n5++]=neiIndex;
			else arr6[n6++]=neiIndex;
		}
		else {
			if(c1 >= c0) arr7[n7++]=neiIndex;
			else arr8[n8++]=neiIndex;
		}	
	}
}
}


void setListInOneDir2D(size_t index, size_t maxNeiNumInOneDir,
                                           const int* list1, size_t sz1, const int* list2, size_t sz2,
                                           int* upwindNeighbourList, int* upwindNeighbourListSize, const double* positionX, const double* positionY) { // output

        size_t neiListInOneDirStartIndex = index*maxNeiNumInOneDir;
        size_t numInOneDir = 0, n1 = 0, n2 = 0;

//        const double *positionX = m_pParticleData->m_vPositionX;
//        const double *positionY = m_pParticleData->m_vPositionY;
//        const double *positionZ = m_pParticleData->m_vPositionZ;
        int nei1,nei2;
        double d1,d2;
        while((numInOneDir < (maxNeiNumInOneDir-1))&&(n1<sz1)&&(n2<sz2)) {
                nei1=list1[n1];
                nei2=list2[n2];
                d1=(positionX[nei1]-positionX[index])*(positionX[nei1]-positionX[index])+(positionY[nei1]-positionY[index])*(positionY[nei1]-positionY[index]);
                d2=(positionX[nei2]-positionX[index])*(positionX[nei2]-positionX[index])+(positionY[nei2]-positionY[index])*(positionY[nei2]-positionY[index]);
                if (d1<d2) {
                        n1=n1+1;
                        n2=n2+1;
                        upwindNeighbourList[neiListInOneDirStartIndex+numInOneDir] = nei1;
                        numInOneDir++;
                        upwindNeighbourList[neiListInOneDirStartIndex+numInOneDir] = nei2;
                        numInOneDir++;
                }
                else {
                        n1=n1+1;
                        n2=n2+1;
                        upwindNeighbourList[neiListInOneDirStartIndex+numInOneDir] = nei2;
                        numInOneDir++;
                        upwindNeighbourList[neiListInOneDirStartIndex+numInOneDir] = nei1;
                        numInOneDir++;
                }
//	        if(n1==sz1 || n2==sz2) break;
        }

        while(numInOneDir < maxNeiNumInOneDir) {
                if(n1 < sz1) {
                        upwindNeighbourList[neiListInOneDirStartIndex+numInOneDir] = list1[n1++];
                        numInOneDir++;
                }
                if(numInOneDir >= maxNeiNumInOneDir) break;
                if(n2 < sz2) {
                        upwindNeighbourList[neiListInOneDirStartIndex+numInOneDir] = list2[n2++];
                        numInOneDir++;
                }
                if(n1==sz1 && n2==sz2) break;
        }
        upwindNeighbourListSize[index] = numInOneDir;
}

bool ascByDist(const pair<int,double>& l, const pair<int,double>& r ) {
        return l.second < r.second;
}

void setListInOneDir3D(size_t index, size_t maxNeiNumInOneDir,
                                           const int* list1, size_t sz1, const int* list2, size_t sz2,
                                           const int* list3, size_t sz3, const int* list4, size_t sz4,
                                           int* upwindNeighbourList, int* upwindNeighbourListSize, const double* positionX, const double* positionY, const double* positionZ) { // output

        size_t neiListInOneDirStartIndex = index*maxNeiNumInOneDir;
        size_t numInOneDir = 0, n1 = 0, n2 = 0, n3 = 0, n4 = 0;
	double dx, dy, dz, dis;
        vector<pair<int,double>> index_dis(4);
	while((numInOneDir < (maxNeiNumInOneDir-3)) && (n1<sz1) && (n2<sz2) && (n3<sz3) && (n4<sz4)) {
		dx=positionX[list1[n1]]-positionX[index];
                dy=positionY[list1[n1]]-positionY[index];
                dz=positionZ[list1[n1]]-positionZ[index];
		dis=dx*dx+dy*dy+dz*dz;
		index_dis[0]={list1[n1],dis};
                dx=positionX[list2[n2]]-positionX[index];
                dy=positionY[list2[n2]]-positionY[index];
                dz=positionZ[list2[n2]]-positionZ[index];
                dis=dx*dx+dy*dy+dz*dz;
                index_dis[1]={list2[n2],dis};
                dx=positionX[list3[n3]]-positionX[index];
                dy=positionY[list3[n3]]-positionY[index];
                dz=positionZ[list3[n3]]-positionZ[index];
                dis=dx*dx+dy*dy+dz*dz;
                index_dis[2]={list3[n3],dis};
                dx=positionX[list4[n4]]-positionX[index];
                dy=positionY[list4[n4]]-positionY[index];
                dz=positionZ[list4[n4]]-positionZ[index];
                dis=dx*dx+dy*dy+dz*dz;
                index_dis[3]={list4[n4],dis};

                std::sort(index_dis.begin(), index_dis.end(), ascByDist);
		for(int add=0; add<4; add++)
                        upwindNeighbourList[neiListInOneDirStartIndex+numInOneDir+add] = index_dis[add].first;
		n1++;
		n2++;
		n3++;
		n4++;
		numInOneDir=numInOneDir+4;
	}
        while(numInOneDir < maxNeiNumInOneDir) {
                if(n1 < sz1) {
                        upwindNeighbourList[neiListInOneDirStartIndex+numInOneDir] = list1[n1++];
                        numInOneDir++;
                }
                if(numInOneDir >= maxNeiNumInOneDir) break;
                if(n2 < sz2) {
                        upwindNeighbourList[neiListInOneDirStartIndex+numInOneDir] = list2[n2++];
                        numInOneDir++;
                }
                if(numInOneDir >= maxNeiNumInOneDir) break;
                if(n3 < sz3) {
                        upwindNeighbourList[neiListInOneDirStartIndex+numInOneDir] = list3[n3++];
                        numInOneDir++;
                }
                if(numInOneDir >= maxNeiNumInOneDir) break;
                if(n4 < sz4) {
                        upwindNeighbourList[neiListInOneDirStartIndex+numInOneDir] = list4[n4++];
                        numInOneDir++;
                }

                if(n1==sz1 && n2==sz2 && n3==sz3 && n4==sz4) break;
        }
        upwindNeighbourListSize[index] = numInOneDir;
}

void reorderNeighbour2D(size_t index, size_t neiListStartIndex, int* neighbourList, int* neighbourListSize, const double* positionX, const double* positionY) {
	vector<int> rnlist(neighbourListSize[index]),rslist(neighbourListSize[index]),lnlist(neighbourListSize[index]),lslist(neighbourListSize[index]);
        vector<pair<int,double>> index_dis(4);
	int rnsize=0,rssize=0,lnsize=0,lssize=0;
	for(size_t i=neiListStartIndex;i<neiListStartIndex+neighbourListSize[index];i++)
	{
		int k=neighbourList[i];
		double dx=positionX[k]-positionX[index];
		double dy=positionY[k]-positionY[index];
		if((dx+0.001*dy)>0&&(dy-0.001*dx>0))	rnlist[rnsize++]=k;
                if((dx+0.001*dy)<0&&(dy-0.001*dx>0))    lnlist[lnsize++]=k;
                if((dx+0.001*dy)>0&&(dy-0.001*dx<0))    rslist[rssize++]=k;
                if((dx+0.001*dy)<0&&(dy-0.001*dx<0))    lslist[lssize++]=k;
	}
	int rnindex=0,rsindex=0,lnindex=0,lsindex=0;
	int neighbourIndex=0;
	double dx,dy,dis;

        while(rnindex<rnsize && lnindex<lnsize && rsindex<rssize && lsindex<lssize) {
                dx=positionX[rnlist[rnindex]]-positionX[index];
                dy=positionY[rnlist[rnindex]]-positionY[index];
                dis=dx*dx+dy*dy;
                index_dis[0]={rnlist[rnindex++],dis};
                dx=positionX[rslist[rsindex]]-positionX[index];
                dy=positionY[rslist[rsindex]]-positionY[index];
                dis=dx*dx+dy*dy;
                index_dis[1]={rslist[rsindex++],dis};
                dx=positionX[lnlist[lnindex]]-positionX[index];
                dy=positionY[lnlist[lnindex]]-positionY[index];
                dis=dx*dx+dy*dy;
                index_dis[2]={lnlist[lnindex++],dis};
                dx=positionX[lslist[lsindex]]-positionX[index];
                dy=positionY[lslist[lsindex]]-positionY[index];
                dis=dx*dx+dy*dy;
                index_dis[3]={lslist[lsindex++],dis};

                std::sort(index_dis.begin(), index_dis.end(), ascByDist);
                for(int add=0; add<4; add++)
                        neighbourList[neiListStartIndex+neighbourIndex+add] = index_dis[add].first;
                neighbourIndex=neighbourIndex+4;
        }
        while(rnindex<rnsize || lsindex<lssize || lnindex<lnsize || rsindex<rssize) {
                if(rnindex < rnsize) {
                        neighbourList[neiListStartIndex+neighbourIndex] = rnlist[rnindex++];
                        neighbourIndex++;
                }
                if(lsindex < lssize) {
                        neighbourList[neiListStartIndex+neighbourIndex] = lslist[lsindex++];
                        neighbourIndex++;
                }
                if(lnindex < lnsize) {
                        neighbourList[neiListStartIndex+neighbourIndex] = lnlist[lnindex++];
                        neighbourIndex++;
                }
                if(rsindex < rssize) {
                        neighbourList[neiListStartIndex+neighbourIndex] = rslist[rsindex++];
                        neighbourIndex++;
                }
        }

}


// only fluid particles have upwind neighbour list
void HyperbolicLPSolver::setUpwindNeighbourList() {
	
	cout<<"-------HyperbolicLPSolver::setUpwindNeighbourList()-------"<<endl;
	const double *positionX = m_pParticleData->m_vPositionX;
	const double *positionY = m_pParticleData->m_vPositionY;		
	const double *positionZ = m_pParticleData->m_vPositionZ;
	if(m_iDimension==2) positionZ = nullptr;
 
	int *neighbourList = m_pParticleData->m_vNeighbourList;
	int *neighbourListRight = m_pParticleData->m_vNeighbourListRight;
	int *neighbourListLeft = m_pParticleData->m_vNeighbourListLeft;
	int *neighbourListNorth = m_pParticleData->m_vNeighbourListNorth;
	int *neighbourListSouth = m_pParticleData->m_vNeighbourListSouth;
	int *neighbourListUp, *neighbourListDown;
	if(m_iDimension==3) {
		neighbourListUp = m_pParticleData->m_vNeighbourListUp;
		neighbourListDown = m_pParticleData->m_vNeighbourListDown;
	}	
	
	int *neighbourListSize = m_pParticleData->m_vNeighbourListSize;
	int *neighbourListRightSize = m_pParticleData->m_vNeighbourListRightSize;
	int *neighbourListLeftSize = m_pParticleData->m_vNeighbourListLeftSize;
	int *neighbourListNorthSize = m_pParticleData->m_vNeighbourListNorthSize;
	int *neighbourListSouthSize = m_pParticleData->m_vNeighbourListSouthSize;
	int *neighbourListUpSize, *neighbourListDownSize;
	if(m_iDimension==3) {
		neighbourListUpSize = m_pParticleData->m_vNeighbourListUpSize;
		neighbourListDownSize = m_pParticleData->m_vNeighbourListDownSize;
	}
		
	// get the maximum size of neighbour lists	
	size_t maxNeiNum = m_pParticleData->m_iMaxNeighbourNum;	
	size_t maxNeiNumInOneDir = m_pParticleData->m_iMaxNeighbourNumInOneDir;

	size_t fluidStartIndex = m_pParticleData->m_iFluidStartIndex;
	size_t fluidEndIndex = m_pParticleData->m_iFluidStartIndex + m_pParticleData->m_iFluidNum;
	
	#ifdef _OPENMP
	#pragma omp parallel
	{
	#endif
	if(m_iDimension==2) {
		
		int rn[maxNeiNum]; int rs[maxNeiNum]; 
		int ln[maxNeiNum]; int ls[maxNeiNum]; 
		int nr[maxNeiNum]; int nl[maxNeiNum]; 
		int sr[maxNeiNum]; int sl[maxNeiNum];

		#ifdef _OPENMP
		#pragma omp for
		#endif
		for(size_t index=fluidStartIndex; index<fluidEndIndex; index++) {
		size_t rnSz = 0, rsSz = 0, lnSz = 0, lsSz = 0;
		size_t nrSz = 0, nlSz = 0, srSz = 0, slSz = 0;
		
		size_t neiListStartIndex = index*maxNeiNum;
		size_t neiListEndIndex = neiListStartIndex + neighbourListSize[index];

		double x0 = positionX[index], y0 = positionY[index];
//Sort the neighbours by distance for free boundary. In other cases the neighbours are already sorted.
		if(m_iFreeBoundary) {
			if(m_vFillGhost[index]) {
				vector<pair<int,double>> index_dis(neighbourListSize[index]);
				for(int i=0; i<neighbourListSize[index]; i++) {
					size_t  neiIndex = neighbourList[neiListStartIndex+i];
					double dx = positionX[neiIndex] - x0, dy = positionY[neiIndex] - y0;
					double dis = dx*dx+dy*dy;
					index_dis[i]={neiIndex,dis};
				}
				std::sort(index_dis.begin(), index_dis.end(), ascByDist);
				for(size_t i=0; i<index_dis.size(); i++) {
					neighbourList[neiListStartIndex+i] = index_dis[i].first;
				}	
			}
		}
//Put neighbours in each direction in a list
		for(size_t i=neiListStartIndex; i<neiListEndIndex; i++) {	
			size_t  neiIndex = neighbourList[i];
			double x1 = positionX[neiIndex], y1 = positionY[neiIndex];
			// right or left
			setListInOneDir2D(neiIndex, x0, x1, y0, y1,
							  rn, rnSz, rs, rsSz, ln, lnSz, ls, lsSz);
			
			// north or south
			setListInOneDir2D(neiIndex, y0, y1, x0, x1,
							  nr, nrSz, nl, nlSz, sr, srSz, sl, slSz);

		}

//Set up upwind stencil
                setListInOneDir2D(index, maxNeiNumInOneDir, rn, rnSz, rs, rsSz,
                                                  neighbourListRight, neighbourListRightSize, positionX, positionY);

                setListInOneDir2D(index, maxNeiNumInOneDir, ln, lnSz, ls, lsSz,
                                                  neighbourListLeft, neighbourListLeftSize, positionX, positionY);

                setListInOneDir2D(index, maxNeiNumInOneDir, nr, nrSz, nl, nlSz,
                                                  neighbourListNorth, neighbourListNorthSize, positionX, positionY);

                setListInOneDir2D(index, maxNeiNumInOneDir, sr, srSz, sl, slSz,
                                                  neighbourListSouth, neighbourListSouthSize, positionX, positionY);

//Set up central stencil, reorder neighbour list to balance neighbours from each quedrant begin
		reorderNeighbour2D(index, neiListStartIndex, neighbourList,neighbourListSize,positionX,positionY);

	}

}
else if(m_iDimension==3) {

	int rnu[maxNeiNum]; int rsu[maxNeiNum]; int rnd[maxNeiNum]; int rsd[maxNeiNum]; // right
	int lnu[maxNeiNum]; int lsu[maxNeiNum]; int lnd[maxNeiNum]; int lsd[maxNeiNum]; // left
	int nru[maxNeiNum]; int nlu[maxNeiNum]; int nrd[maxNeiNum]; int nld[maxNeiNum]; // north
	int sru[maxNeiNum]; int slu[maxNeiNum]; int srd[maxNeiNum]; int sld[maxNeiNum]; // south
	int urn[maxNeiNum]; int uln[maxNeiNum]; int urs[maxNeiNum]; int uls[maxNeiNum]; // up
	int drn[maxNeiNum]; int dln[maxNeiNum]; int drs[maxNeiNum]; int dls[maxNeiNum]; // down 
	
	#ifdef _OPENMP
	#pragma omp for
	#endif
	for(size_t index=fluidStartIndex; index<fluidEndIndex; index++) {
		size_t rnuSz = 0, rsuSz = 0, rndSz = 0, rsdSz = 0;
		size_t lnuSz = 0, lsuSz = 0, lndSz = 0, lsdSz = 0;
		size_t nruSz = 0, nluSz = 0, nrdSz = 0, nldSz = 0;
		size_t sruSz = 0, sluSz = 0, srdSz = 0, sldSz = 0;
		size_t urnSz = 0, ulnSz = 0, ursSz = 0, ulsSz = 0; 
		size_t drnSz = 0, dlnSz = 0, drsSz = 0, dlsSz = 0; 
		
		size_t neiListStartIndex = index*maxNeiNum;
		size_t neiListEndIndex = neiListStartIndex + neighbourListSize[index];
		
		double x0 = positionX[index], y0 = positionY[index], z0 = positionZ[index];
	
		if(m_iFreeBoundary) {
			if(m_vFillGhost[index]) {
				vector<pair<int,double>> index_dis(neighbourListSize[index]);
				for(size_t i=neiListStartIndex; i<neiListEndIndex; i++) {
					size_t  neiIndex = neighbourList[i];
					double dx = positionX[neiIndex] - x0, dy = positionY[neiIndex] - y0, dz = positionZ[neiIndex] - z0;
					double dis = dx*dx+dy*dy+dz*dz;
					index_dis[i-neiListStartIndex]={neiIndex,dis};
				}
				std::sort(index_dis.begin(), index_dis.end(),ascByDist);
				for(size_t i=neiListStartIndex; i<neiListEndIndex; i++) {
					neighbourList[i] = index_dis[i-neiListStartIndex].first;
				}
			}
		}	
		
		for(size_t i=neiListStartIndex; i<neiListEndIndex; i++) {	
			size_t  neiIndex = neighbourList[i];
			double x1 = positionX[neiIndex], y1 = positionY[neiIndex], z1 = positionZ[neiIndex];
			// right or left
			setListInOneDir3D(neiIndex, x0, x1, y0, y1, z0, z1,
							  rnu, rnuSz, rnd, rndSz, rsu, rsuSz, rsd, rsdSz,
							  lnu, lnuSz, lnd, lndSz, lsu, lsuSz, lsd, lsdSz);
			// north or south
			setListInOneDir3D(neiIndex, y0, y1, x0, x1, z0, z1,
							  nru, nruSz, nrd, nrdSz, nlu, nluSz, nld, nldSz,
							  sru, sruSz, srd, srdSz, slu, sluSz, sld, sldSz);
			// up or down
			setListInOneDir3D(neiIndex, z0, z1, x0, x1, y0, y1,
							  urn, urnSz, urs, ursSz, uln, ulnSz, uls, ulsSz,	
							  drn, drnSz, drs, drsSz, dln, dlnSz, dls, dlsSz);	
		}
	

		//DEBUG INFO
		//if(index%100==0 && index<=1000) {
		//	cout<<"-------HyperbolicLPSolver::setUpwindNeighbourList()-------"<<endl;
		//	cout<<"index="<<index<<endl;
		//	cout<<"rnuSz="<<rnuSz<<endl;
		//	cout<<"rsuSz="<<rsuSz<<endl;
		//	cout<<"rndSz="<<rndSz<<endl;
		//	cout<<"rsdSz="<<rsdSz<<endl;	
		//	cout<<"right size = "<<rnuSz+rsuSz+rndSz+rsdSz<<endl;
		//	cout<<"lnuSz="<<lnuSz<<endl;
		//	cout<<"lsuSz="<<lsuSz<<endl;
		//	cout<<"lndSz="<<lndSz<<endl;
		//	cout<<"lsdSz="<<lsdSz<<endl;
		//	cout<<"left size = "<<lnuSz+lsuSz+lndSz+lsdSz<<endl;
		//	cout<<"nruSz="<<nruSz<<endl;
		//	cout<<"nluSz="<<nluSz<<endl;
		//	cout<<"nrdSz="<<nrdSz<<endl;
		//	cout<<"nldSz="<<nldSz<<endl;
		//	cout<<"north size = "<<nruSz+nluSz+nrdSz+nldSz<<endl;
		//	cout<<"sruSz="<<sruSz<<endl;
		//	cout<<"sluSz="<<sluSz<<endl;
		//	cout<<"srdSz="<<srdSz<<endl;
		//	cout<<"sldSz="<<sldSz<<endl;
		//	cout<<"south size = "<<sruSz+sluSz+srdSz+sldSz<<endl;
		//	cout<<"urnSz="<<urnSz<<endl;
		//	cout<<"ulnSz="<<ulnSz<<endl;
		//	cout<<"ursSz="<<ursSz<<endl;
		//	cout<<"ulsSz="<<ulsSz<<endl;
		//	cout<<"up size = "<<urnSz+ulnSz+ursSz+ulsSz<<endl;
		//	cout<<"drnSz="<<drnSz<<endl;
		//	cout<<"dlnSz="<<dlnSz<<endl;
		//	cout<<"drsSz="<<drsSz<<endl;
		//	cout<<"dlsSz="<<dlsSz<<endl;
		//	cout<<"down size = "<<drnSz+dlnSz+drsSz+dlsSz<<endl;
		//	cout<<"----------------------------------------------------------"<<endl;
		//}

		// call helper function to set the lists
		//right
		setListInOneDir3D(index, maxNeiNumInOneDir, rnu, rnuSz, rsu, rsuSz, rnd, rndSz, rsd, rsdSz,
						  neighbourListRight, neighbourListRightSize, positionX, positionY, positionZ); // output
		// left
		setListInOneDir3D(index, maxNeiNumInOneDir, lnu, lnuSz, lsu, lsuSz, lnd, lndSz, lsd, lsdSz,
						  neighbourListLeft, neighbourListLeftSize, positionX, positionY, positionZ); // output
		//north
		setListInOneDir3D(index, maxNeiNumInOneDir, nru, nruSz, nlu, nluSz, nrd, nrdSz, nld, nldSz,
						  neighbourListNorth, neighbourListNorthSize, positionX, positionY, positionZ); // output
		//south
		setListInOneDir3D(index, maxNeiNumInOneDir, sru, sruSz, slu, sluSz, srd, srdSz, sld, sldSz,
						  neighbourListSouth, neighbourListSouthSize, positionX, positionY, positionZ); // output
		//up
		setListInOneDir3D(index, maxNeiNumInOneDir, urn, urnSz, uln, ulnSz, urs, ursSz, uls, ulsSz,
						  neighbourListUp, neighbourListUpSize, positionX, positionY, positionZ); // output
		//down
		setListInOneDir3D(index, maxNeiNumInOneDir, drn, drnSz, dln, dlnSz, drs, drsSz, dls, dlsSz,
						  neighbourListDown, neighbourListDownSize, positionX, positionY, positionZ); // output
//reordering neighbour list to balance neighbours from each quedrant begin

/*		double alpha=0.001,beta=0.0001,gamma=0.00001;
		double A[9];
		A[0]=cos(alpha)*cos(gamma)-sin(alpha)*sin(beta)*sin(gamma);
		A[1]=-sin(alpha)*cos(beta);
		A[2]=-cos(alpha)*sin(gamma)-sin(alpha)*sin(beta)*cos(gamma);
		A[3]=cos(alpha)*sin(beta)*sin(gamma)+sin(alpha)*cos(gamma);
		A[4]=cos(alpha)*cos(beta);
		A[5]=cos(alpha)*sin(beta)*cos(gamma)-sin(alpha)*sin(gamma);
		A[6]=cos(beta)*sin(gamma);
		A[7]=-sin(beta);
		A[8]=cos(beta)*cos(gamma);*/

		vector<pair<int,double>> index_dis(neighbourListSize[index]);
		for(int i=0; i<neighbourListSize[index]; i++) {
			size_t  neiIndex = neighbourList[neiListStartIndex+i];
			double dx = positionX[neiIndex] - x0, dy = positionY[neiIndex] - y0, dz = positionZ[neiIndex] - z0;
			double dis = dx*dx+dy*dy+dz*dz;
			index_dis[i]={neiIndex,dis};
		}
		std::sort(index_dis.begin(), index_dis.end(), ascByDist);
		for(size_t i=0; i<index_dis.size(); i++) {
			neighbourList[neiListStartIndex+i] = index_dis[i].first;
		}
/*		double penalty[8];
		for(int i=0;i<8;i++)
			penalty[i]=1.0;
		double penalty_weight=10000.0;
		for(int i=0; i<neighbourListSize[index]; i++) {
			size_t  neiIndex = neighbourList[neiListStartIndex+i];
			double dx = positionX[neiIndex] - x0, dy = positionY[neiIndex] - y0, dz = positionZ[neiIndex] - z0;
			double dis = dx*dx+dy*dy+dz*dz;
			int region=4*((A[0]*dx+A[1]*dy+A[2]*dz)>0)+2*((A[3]*dx+A[4]*dy+A[5]*dz)>0)+((A[6]*dx+A[7]*dy+A[8]*dz)>0);
			dis=dis*penalty[region];
			penalty[region]=penalty[region]*penalty_weight;

			index_dis[i]={neiIndex,dis};

		}*/

	        double penalty[6];
                for(int i=0;i<6;i++)
                        penalty[i]=1.0;
                double penalty_weight=10000.0;
                for(int i=0; i<neighbourListSize[index]; i++) {
                        size_t  neiIndex = neighbourList[neiListStartIndex+i];
                        double dx = positionX[neiIndex] - x0, dy = positionY[neiIndex] - y0, dz = positionZ[neiIndex] - z0;
                        double dis = dx*dx+dy*dy+dz*dz;
			int region;
			if ((fabs(dx)>=fabs(dy))&&(fabs(dx)>=fabs(dz)))
			{
				if(dx>0)
					region=0;
				else
					region=1;
			}
                        else if ((fabs(dy)>=fabs(dx))&&(fabs(dy)>=fabs(dz)))
                        {
                                if(dy>0)
                                        region=2;
                                else
                                        region=3;
                        }
                        else
                        {
                                if(dz>0)
                                        region=4;
                                else
                                        region=5;
                        }
                        dis=dis*penalty[region];
                        penalty[region]=penalty[region]*penalty_weight;
                        index_dis[i]={neiIndex,dis};

                }


		std::sort(index_dis.begin(), index_dis.end(), ascByDist);
//		cout<<endl;
//		cout<<index<<" "<<x0<<" "<<y0<<" "<<z0<<endl;
		for(size_t i=0; i<index_dis.size(); i++) {
			neighbourList[neiListStartIndex+i] = index_dis[i].first;
		}

//reordering neighbour list to balance neighbours from each quedrant end

	}

}

#ifdef _OPENMP
}
#endif

// DEBUG (check if the upwind neighbour list is correct)
//checkUpwindNeighbourList();

cout<<"-------END HyperbolicLPSolver::setUpwindNeighbourList()-------"<<endl;
}



void HyperbolicLPSolver::resetLPFOrder() {
size_t fluidStartIndex = m_pParticleData->m_iFluidStartIndex;	
size_t fluidEndIndex = m_pParticleData->m_iFluidStartIndex + m_pParticleData->m_iFluidNum;
if(m_iDimension==3) {
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for(size_t index=fluidStartIndex; index<fluidEndIndex; index++) {
		m_pParticleData->m_vLPFOrderRight[index] = m_iLPFOrder;
		m_pParticleData->m_vLPFOrderLeft[index] = m_iLPFOrder;
		m_pParticleData->m_vLPFOrderNorth[index] = m_iLPFOrder;
		m_pParticleData->m_vLPFOrderSouth[index] = m_iLPFOrder;
		m_pParticleData->m_vLPFOrderUp[index] = m_iLPFOrder;
		m_pParticleData->m_vLPFOrderDown[index] = m_iLPFOrder;	
	}
}
else if(m_iDimension==2) {
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif	
	for(size_t index=fluidStartIndex; index<fluidEndIndex; index++) {
		m_pParticleData->m_vLPFOrderRight[index] = m_iLPFOrder;
		m_pParticleData->m_vLPFOrderLeft[index] = m_iLPFOrder;
		m_pParticleData->m_vLPFOrderNorth[index] = m_iLPFOrder;
		m_pParticleData->m_vLPFOrderSouth[index] = m_iLPFOrder;
	}	
}
}

void HyperbolicLPSolver::computeMinParticleSpacing() {

// set particle position pointers (to save typing)
const double *positionX = m_pParticleData->m_vPositionX;
const double *positionY = m_pParticleData->m_vPositionY;
const double *positionZ = m_pParticleData->m_vPositionZ;

// whole neighbour list
const int *neighbourList = m_pParticleData->m_vNeighbourList;
const int *neighbourListSize = m_pParticleData->m_vNeighbourListSize;

size_t maxNeiNum = m_pParticleData->m_iMaxNeighbourNum;

// initial value
m_fMinParticleSpacing=numeric_limits<double>::max();	

size_t startIndex = m_pParticleData->getFluidStartIndex();
size_t endIndex = startIndex + m_pParticleData->getFluidNum();

#ifdef _OPENMP
size_t numThreads = min(omp_get_max_threads(), m_iNumThreads);
//	cout<<"numThreads="<<numThreads<<endl;
//	cout<<"omp_get_max_threads()="<<omp_get_max_threads()<<endl;
vector<double> minsp(numThreads,numeric_limits<double>::max());	
#pragma omp parallel  
{
int tid = omp_get_thread_num();
#endif

if(m_iDimension==3) {
	#ifdef _OPENMP
	#pragma omp for
	#endif
	for(size_t index=startIndex; index<endIndex; index++) { // for each fluid particle

		size_t totalNumNei = neighbourListSize[index];	
		for(size_t i=0; i<totalNumNei; i++) { // for all of its neighbours
			size_t neiIndex = neighbourList[index*maxNeiNum+i];
			if(neiIndex>=startIndex && neiIndex<endIndex) { // find one fluid neighbour, compare, and break
				double dx = positionX[neiIndex] - positionX[index];
				double dy = positionY[neiIndex] - positionY[index];
				double dz = positionZ[neiIndex] - positionZ[index];
				double dist  = sqrt(dx*dx + dy*dy + dz*dz);
				#ifdef _OPENMP
					minsp[tid] = min(minsp[tid],dist);
				#else
					m_fMinParticleSpacing = min(m_fMinParticleSpacing,dist);
				#endif
				
				break;
			}
		}
	}
}
else if(m_iDimension==2) {
	#ifdef _OPENMP
	#pragma omp for
	#endif
	for(size_t index=startIndex; index<endIndex; index++) { // for each fluid particle

		size_t totalNumNei = neighbourListSize[index];	
		for(size_t i=0; i<totalNumNei; i++) { // for all of its neighbours
			size_t neiIndex = neighbourList[index*maxNeiNum+i];
			if(neiIndex>=startIndex && neiIndex<endIndex) { // find one fluid neighbour, compare, and break
				double dx = positionX[neiIndex] - positionX[index];
				double dy = positionY[neiIndex] - positionY[index];
				double dist  = sqrt(dx*dx + dy*dy);
				#ifdef _OPENMP
					minsp[tid] = min(minsp[tid],dist);
				#else
					m_fMinParticleSpacing = min(m_fMinParticleSpacing,dist);
				#endif	
				break;
			}
		}
	}
	#ifdef _OPENMP
	for(size_t i=0; i<numThreads; i++) {
		m_fMinParticleSpacing = min(m_fMinParticleSpacing,minsp[i]);
	}
	#endif

}

#ifdef _OPENMP
	for(size_t i=0; i<numThreads; i++) {
		m_fMinParticleSpacing = min(m_fMinParticleSpacing,minsp[i]);	
	}
}
#endif

// the corner case when no fluid particle has a fluid neighbour
assert(m_fMinParticleSpacing != numeric_limits<double>::max());

//debug<<"-------HyperbolicLPSolver::computeMinParticleSpacing()-------"<<endl;
cout<<"m_fMinParticleSpacing="<<m_fMinParticleSpacing<<endl;
//debug<<"-------------------------------------------------------------"<<endl;
}


void HyperbolicLPSolver::computeAvgParticleSpacing() {

// set particle position pointers (to save typing)
const double *positionX = m_pParticleData->m_vPositionX;
const double *positionY = m_pParticleData->m_vPositionY;
const double *positionZ = m_pParticleData->m_vPositionZ;

// whole neighbour list
const int *neighbourList = m_pParticleData->m_vNeighbourList;
const int *neighbourListSize = m_pParticleData->m_vNeighbourListSize;

size_t maxNeiNum = m_pParticleData->m_iMaxNeighbourNum;
	
double sumDist = 0;
size_t num = 0;
size_t startIndex = m_pParticleData->getFluidStartIndex();
size_t endIndex = startIndex + m_pParticleData->getFluidNum();
if(m_iDimension==3) {
	for(size_t index=startIndex; index<endIndex; index++) { // for each fluid particle

		size_t totalNumNei = neighbourListSize[index];	
		for(size_t i=0; i<totalNumNei; i++) { // for all of its neighbours
			size_t neiIndex = neighbourList[index*maxNeiNum+i];
			if(neiIndex>=startIndex && neiIndex<endIndex) { // find one fluid neighbour, compare, and break
				double dx = positionX[neiIndex] - positionX[index];
				double dy = positionY[neiIndex] - positionY[index];
				double dz = positionZ[neiIndex] - positionZ[index];
				double dist  = sqrt(dx*dx + dy*dy + dz*dz);
				sumDist += dist;
				num++;
				break;
			}
		}
	}
}
else if(m_iDimension==2) {
	for(size_t index=startIndex; index<endIndex; index++) { // for each fluid particle

		size_t totalNumNei = neighbourListSize[index];	
		for(size_t i=0; i<totalNumNei; i++) { // for all of its neighbours
			size_t neiIndex = neighbourList[index*maxNeiNum+i];
			if(neiIndex>=startIndex && neiIndex<endIndex) { // find one fluid neighbour, compare, and break
				double dx = positionX[neiIndex] - positionX[index];
				double dy = positionY[neiIndex] - positionY[index];
				double dist  = sqrt(dx*dx + dy*dy);
				sumDist += dist;
				num++;
				break;
			}
		}
	}
}

assert(num!=0);	

//cout<<"-------HyperbolicLPSolver::computeAvgParticleSpacing()-------"<<endl;
cout<<"m_fAvgParticleSpacing changed from "<<m_fAvgParticleSpacing;
m_fAvgParticleSpacing = sumDist/(double)num;
cout<<" to "<<m_fAvgParticleSpacing<<endl;
//cout<<"-------------------------------------------------------------"<<endl;
}


void HyperbolicLPSolver::updateLocalParSpacingByVolume() {
	
const double *volume    = m_pParticleData->m_vVolume;
const double *volumeOld = m_pParticleData->m_vVolumeOld;
		
double *localParSpacing = m_pParticleData->m_vLocalParSpacing;

size_t startIndex = m_pParticleData->getFluidStartIndex();
size_t endIndex = startIndex + m_pParticleData->getFluidNum();

#ifdef _OPENMP
#pragma omp parallel for 
#endif
for(size_t index=startIndex; index<endIndex; index++) {
	if(volume[index]==0 || volumeOld[index]==0) {
		cout<<"Cannot update local_spacing because: volume["<<index<<"]="
		<<volume[index]<<"  volumeOld["<<index<<"]="<<volumeOld[index]<<endl;
		//continue;
	}
	else {
if(m_iDimension==3) 
		localParSpacing[index] *= std::cbrt(volume[index]/volumeOld[index]);
if(m_iDimension==2)
                localParSpacing[index] *= std::sqrt(volume[index]/volumeOld[index]);

	}
}	
}

void HyperbolicLPSolver::computeMaxSoundSpeed() {

const double* soundSpeed = m_pParticleData->m_vSoundSpeed;

//initial value
//m_fMaxSoundSpeed = numeric_limits<double>::min();
m_fMaxSoundSpeed = -1;

size_t startIndex = m_pParticleData->m_iFluidStartIndex;
size_t endIndex = m_pParticleData->m_iFluidStartIndex + m_pParticleData->m_iFluidNum;	

#ifdef _OPENMP
size_t numThreads = min(omp_get_max_threads(), m_iNumThreads);
//	cout<<"numThreads="<<numThreads<<endl;
//	cout<<"omp_get_max_threads()="<<omp_get_max_threads()<<endl;
vector<double> maxcs(numThreads,-1);	
#pragma omp parallel  
{
int tid = omp_get_thread_num();
#endif

#ifdef _OPENMP
#pragma omp for
#endif
for(size_t index=startIndex; index<endIndex; index++) {
	#ifdef _OPENMP
//	if(m_pParticleData->m_vVolume[index]<1e+4)
		maxcs[tid] = max(maxcs[tid],soundSpeed[index]);
	#else
//        if(m_pParticleData->m_vVolume[index]<1e+4)
		m_fMaxSoundSpeed = max(m_fMaxSoundSpeed,soundSpeed[index]);
	#endif
}

#ifdef _OPENMP
	for(size_t i=0; i<numThreads; i++) {
		m_fMaxSoundSpeed = max(m_fMaxSoundSpeed,maxcs[i]);	
	}
}
#endif

//assert(m_fMaxSoundSpeed != -1);

//cout<<"-------HyperbolicLPSolver::computeMaxSoundSpeed()-------"<<endl;
cout<<"m_fMaxSoundSpeed="<<m_fMaxSoundSpeed<<endl;
//cout<<"--------------------------------------------------------"<<endl;
assert(m_fMaxSoundSpeed != -1);

}


void HyperbolicLPSolver::computeMinCFL() {
const double* vU = m_pParticleData->m_vVelocityU;
const double* vV = m_pParticleData->m_vVelocityV;
const double* vW = (m_iDimension==3)? m_pParticleData->m_vVelocityW:nullptr;
double *localParSpacing = m_pParticleData->m_vLocalParSpacing;
const double* soundSpeed = m_pParticleData->m_vSoundSpeed;
m_fMinCFL = -1;

size_t startIndex = m_pParticleData->getFluidStartIndex();
size_t endIndex = startIndex + m_pParticleData->getFluidNum();
#ifdef _OPENMP
size_t numThreads = min(omp_get_max_threads(), m_iNumThreads);
vector<double> mincfl(numThreads,-1);
#pragma omp parallel  
{
int tid = omp_get_thread_num();
#endif

#ifdef _OPENMP
#pragma omp for
#endif
for(size_t index=startIndex; index<endIndex; index++) {
	double dx = localParSpacing[index];
	double sound = soundSpeed[index];
        double speed = vU[index]*vU[index]+vV[index]*vV[index];
	if(m_iDimension==3)
		speed+=vW[index]*vW[index];
	double cfl=dx*dx/max(sound*sound,speed);
                #ifdef _OPENMP
		if(cfl<mincfl[tid] || mincfl[tid]<0)
			mincfl[tid]=cfl;
		#else
		if(cfl<m_fMinCFL || m_fMinCFL<0)
			m_fMinCFL = cfl;
		#endif
}
#ifdef _OPENMP
m_fMinCFL = mincfl[0];
        for(size_t i=0; i<numThreads; i++) {
                m_fMinCFL = min(m_fMinCFL,mincfl[i]);
        }
}
#endif

assert(m_fMinCFL != -1);
m_fMinCFL =sqrt(m_fMinCFL);
cout<<"m_fMinCFL = "<<m_fMinCFL<<endl;
}

void HyperbolicLPSolver::computeMaxFluidVelocity() {

const double* vU = m_pParticleData->m_vVelocityU;
const double* vV = m_pParticleData->m_vVelocityV;
const double* vW = (m_iDimension==3)? m_pParticleData->m_vVelocityW:nullptr;

// initial value
m_fMaxFluidVelocity = -1;

size_t startIndex = m_pParticleData->getFluidStartIndex();
size_t endIndex = startIndex + m_pParticleData->getFluidNum();	
#ifdef _OPENMP
size_t numThreads = min(omp_get_max_threads(), m_iNumThreads);
//	cout<<"numThreads="<<numThreads<<endl;
//	cout<<"omp_get_max_threads()="<<omp_get_max_threads()<<endl;
vector<double> maxs(numThreads,-1);	
#pragma omp parallel  
{
int tid = omp_get_thread_num();
#endif
	
if(m_iDimension==3) {
	#ifdef _OPENMP
	#pragma omp for
	#endif
	for(size_t index=startIndex; index<endIndex; index++) { 
		double speed = vU[index]*vU[index]+vV[index]*vV[index]+vW[index]*vW[index];
		#ifdef _OPENMP
//	        if(m_pParticleData->m_vVolume[index]<1e+4)
			maxs[tid] = max(maxs[tid],speed);
		#else
//	        if(m_pParticleData->m_vVolume[index]<1e+4)
			m_fMaxFluidVelocity = max(m_fMaxFluidVelocity,speed);
		#endif
	}
}
else if(m_iDimension==2) {
	#ifdef _OPENMP
	#pragma omp for
	#endif
	for(size_t index=startIndex; index<endIndex; index++) {
		double speed = vU[index]*vU[index]+vV[index]*vV[index];
		#ifdef _OPENMP
//	        if(m_pParticleData->m_vVolume[index]<1e+4)
			maxs[tid] = max(maxs[tid],speed);
		#else
//	        if(m_pParticleData->m_vVolume[index]<1e+4)
			m_fMaxFluidVelocity = max(m_fMaxFluidVelocity,speed);	
		#endif
	}
}

#ifdef _OPENMP
	for(size_t i=0; i<numThreads; i++) {
		m_fMaxFluidVelocity = max(m_fMaxFluidVelocity,maxs[i]);	
	}
}
#endif

assert(m_fMaxFluidVelocity != -1);

m_fMaxFluidVelocity = sqrt(m_fMaxFluidVelocity);

//cout<<"-------HyperbolicLPSolver::computeMaxFluidVelocity()-------"<<endl;
cout<<"m_fMaxFluidVelocity="<<m_fMaxFluidVelocity<<endl;
//cout<<"--------------------------------------------------------"<<endl;

}

bool HyperbolicLPSolver::density_derivative() {
const int *neiList=m_pParticleData->m_vNeighbourList;
const int *neiListSize=m_pParticleData->m_vNeighbourListSize;
const double *inVolume=m_pParticleData->m_vVolumeOld;
double *Volume_x=m_pParticleData->m_vVolume_x;
double *Volume_y=m_pParticleData->m_vVolume_y;
double *Volume_z=m_pParticleData->m_vVolume_z;
double *inDensity=m_pParticleData->m_vDensity;
int *LPFOrder0=nullptr, *LPFOrder1=nullptr;
vector<int*> LPFOrderOther;    
setLPFOrderPointers(0,&LPFOrder0,&LPFOrder1,LPFOrderOther);

bool phaseSuccess = true;

void (HyperbolicLPSolver::*computeA) (size_t, const int *, const int*, size_t, size_t,double*, double*);
int offset;
if(m_iDimension==2) {computeA = &HyperbolicLPSolver::computeA2D; offset=2;}
else if(m_iDimension==3) {computeA = &HyperbolicLPSolver::computeA3D; offset=3;}
// iteration start index
size_t startIndex = m_pParticleData->m_iFluidStartIndex;
size_t endIndex = startIndex + m_pParticleData->m_iFluidNum;
for(size_t index=startIndex; index<(endIndex+m_pParticleData->m_iBoundaryNum+m_pParticleData->m_iGhostNum); index++)
	inDensity[index]=1.0/inVolume[index];

// iterate through fluid particles
#ifdef _OPENMP
#pragma omp parallel for 
#endif
for(size_t index=startIndex; index<endIndex; index++) {
                double volume_x, volume_y, volume_z; // output
                computeSpatialDer(index, offset, computeA,
                                                  inVolume, neiList, neiListSize,
                                                  LPFOrder0, &volume_x, &volume_y, &volume_z); // output
		Volume_x[index]=volume_x;
                Volume_y[index]=volume_y;
                Volume_z[index]=volume_z;
}
return phaseSuccess;
}

bool HyperbolicLPSolver::solve_upwind(int phase) {
//      cout<<"--------------HyperbolicLPSolver::solve_upwind()--------------"<<endl;
int numFluid = m_pParticleData->m_iFluidNum;
// determine dir: x(0), y(1), or z(2)
const int dir = m_vDirSplitTable[m_iDirSplitOrder][phase];
// set neighbour list pointers by dir (dir=0->right/left, dir=1->north/south, dir=2->up/down)
const int *neiList0=nullptr, *neiList1=nullptr;
const int *neiListSize0=nullptr, *neiListSize1=nullptr;
setNeighbourListPointers_gpu(dir, &neiList0, &neiList1, &neiListSize0, &neiListSize1);
// input data pointers
const double *inVelocity=nullptr, *inPressure=nullptr, *inVolume=nullptr, *inSoundSpeed=nullptr;
// output data pointers
double *outVelocity=nullptr, *outPressure=nullptr, *outVolume=nullptr, *outSoundSpeed=nullptr;
// set data pointers by phase (for temp1 or temp2) and dir(for velocity U or V) 
setInAndOutDataPointers_gpu(phase,dir,&inVelocity,&inPressure,&inVolume,&inSoundSpeed,
						&outVelocity,&outPressure,&outVolume,&outSoundSpeed);
// set local polynomail order pointers
// dir==0->right(0) & left(1), dir==1->north(0) & south(1), dir==2->up(0) & down(1)
int *LPFOrder0=nullptr, *LPFOrder1=nullptr;
vector<int*> LPFOrderOther;
setLPFOrderPointers_gpu(dir,&LPFOrder0,&LPFOrder1,LPFOrderOther);
// phase_success will be false if one particle LPF order is zero in one side
double *p_d_0 = nullptr, *p_dd_0 = nullptr, *vel_d_0 = nullptr , *vel_dd_0 = nullptr;
double *p_d_1 = nullptr, *p_dd_1 = nullptr, *vel_d_1 = nullptr , *vel_dd_1 = nullptr;
setDirOfPressureAndVelocityPointer_gpu(&p_d_0, &p_dd_0, &p_d_1, &p_dd_1, &vel_d_0, &vel_dd_0, &vel_d_1, &vel_dd_1);
double* deltaq = nullptr;
if(m_pParticleData->m_iNumberofPellet){
    *&deltaq = m_pParticleData->d_deltaq;  
    }




bool phaseSuccess = true;



cout<<"BOUARY NUMBLE!! "<<m_pParticleData->m_iBoundaryNum<<endl;
    cout<<"GOAST NUMBLE!! "<<m_pParticleData->m_iGhostNum<<endl;



// gravity is only on y(1) direction 
double gravity;
if(m_iDimension==2) gravity = dir==1? m_fGravity:0; // only in y direction
else if(m_iDimension==3) gravity = dir==1? m_fGravity:0; // only in z direction TODO: MODIFIED GRAVITY DIRECTION FOR POWDER TARGET SIMULATION

// set real dt:  
// 2D: phase:   0               1     2 
//                         dt/4   dt/2   dt/4
// 3D: phase:           0               1               2               3               4
//                                dt/6    dt/6     dt/3    dt/6    dt/6

//copy data from host and do the computation

double realDt;
if(m_iDimension==2) realDt = phase==1? m_fDt/2.:m_fDt/4.;
else if(m_iDimension==3) realDt = phase==2? m_fDt/3.:m_fDt/6.;
// set the function pointer to compute the A matrix for QR solver
void (HyperbolicLPSolver::*computeA) (const int *, const int*, int, int, int);
int offset; // offset is used to get results of computed spatial derivatives from QR solver
if(m_iDimension==2) {computeA = &HyperbolicLPSolver::computeA2D_cpu; offset=2;}
else if(m_iDimension==3) {computeA = &HyperbolicLPSolver::computeA3D_cpu; offset=3;}
// the coeff before first and second order term during time integration
double multiplier1st, multiplier2nd;
if(m_iDimension==2) {multiplier1st=2; multiplier2nd=m_fDt/2.;}
else if(m_iDimension==3) {multiplier1st=3; multiplier2nd=m_fDt*3./4.;}
// iteration start index
int additional=0;
//int numFluid = m_pParticleData->m_iFluidNum;
int blocks = 128;
int threads = 32;

initLPFOrder_upwind_gpu<<<blocks,threads>>>(LPFOrder0, LPFOrder1, numFluid);

computeSpatialDer_gpu(dir, offset, computeA, inPressure, inVelocity,
	neiList0, neiListSize0, additional,
	 LPFOrder0, vel_d_0, vel_dd_0,  p_d_0,  p_dd_0);

computeSpatialDer_gpu(dir, offset, computeA, inPressure, inVelocity,
	neiList1, neiListSize1, additional,
	 LPFOrder1, vel_d_1, vel_dd_1,  p_d_1,  p_dd_1);
cudaMemset(d_info_single,0,sizeof(int));
int info[1];
timeIntegration_gpu<<<blocks,threads>>>(
         realDt,  multiplier1st,  multiplier2nd,  numFluid,
         gravity, inVolume, inVelocity, inPressure, inSoundSpeed,
         vel_d_0, vel_dd_0, p_d_0, p_dd_0,
         vel_d_1, vel_dd_1, p_d_1, p_dd_1,
         outVolume, outVelocity, outPressure, outSoundSpeed, d_info_single);

if(m_pParticleData->m_iNumberofPellet){
   
   updateOutPressureForPellet_gpu<<<blocks,threads>>>(deltaq, outPressure, realDt, m_pGamma, numFluid, d_info_single);

    }


checkPressureAndDensity_gpu<<<blocks,threads>>>(outPressure, outVolume, outVelocity, outSoundSpeed, inPressure, inVelocity,
inVolume, inSoundSpeed, m_fInvalidPressure, m_fInvalidDensity, numFluid);

updateSoundSpeed_gpu<<<blocks,threads>>>(outPressure, outVolume, outSoundSpeed, m_pGamma, numFluid, d_info_single);

cudaMemcpy(info,d_info_single,sizeof(int),cudaMemcpyDeviceToHost);
if(info[0] == 1){
    printf("Wrong output data from timeintegration and updateSoundSpeed!\n");
    assert(false);
}
return phaseSuccess;
}







bool HyperbolicLPSolver::solve_laxwendroff() {
        cout<<"--------------HyperbolicLPSolver::solve_laxwendroff()--------------"<<endl;
        // set neighbour list pointers  const int *neiList=m_pParticleData->m_vNeighbourList;
        const int *neiList=m_pParticleData->m_vNeighbourList;
        const int *neiListSize=m_pParticleData->m_vNeighbourListSize;
        // input data pointers
        const double *inVelocityU, *inVelocityV, *inVelocityW, *inPressure=m_pParticleData->m_vPressure, *inVolume=m_pParticleData->m_vVolume, *inSoundSpeed=m_pParticleData->m_vSoundSpeed;
        inVelocityU = m_pParticleData->m_vVelocityU;
        inVelocityV = m_pParticleData->m_vVelocityV;
        if(m_iDimension==3)
                inVelocityW = m_pParticleData->m_vVelocityW;
        // output data pointers
        double *outVelocityU, *outVelocityV, *outVelocityW, *outPressure=m_pParticleData->m_vTemp1Pressure, *outVolume=m_pParticleData->m_vTemp1Volume, *outSoundSpeed=m_pParticleData->m_vTemp1SoundSpeed;
        outVelocityU = m_pParticleData->m_vTemp1VelocityU;
        outVelocityV = m_pParticleData->m_vTemp1VelocityV;
        if(m_iDimension==3)
                outVelocityW = m_pParticleData->m_vTemp1VelocityW;
// set local polynomail order pointers
        int *LPFOrder=m_pParticleData->m_vLPFOrderRight;
// phase_success
        bool phaseSuccess = true;
// gravity is only on the last direction 
        double gravity=m_fGravity;
// set the function pointer to compute the A matrix for QR solver

        void (HyperbolicLPSolver::*computeA) (size_t, const int *, const int*, size_t, size_t, double*, double*);
        int number_of_derivative;
        size_t offset;
        if(m_iDimension==2) {computeA = &HyperbolicLPSolver::computeA2D; number_of_derivative=5; offset = 2;}
        else if(m_iDimension==3) {computeA = &HyperbolicLPSolver::computeA3D; number_of_derivative=9; offset = 3;}
        // iteration start index
        size_t startIndex = m_pParticleData->m_iFluidStartIndex;
        size_t endIndex = startIndex + m_pParticleData->m_iFluidNum;

        // iterate through fluid particles
        #ifdef _OPENMP
        #pragma omp parallel for 
        #endif
        for(size_t index=startIndex; index<endIndex; index++) {
		if(m_iUseLimiter){
	                m_pParticleData->m_vPhi[index]=1;
        	        int count_p=0,count_u=0,count_v=0,count_rho=0;
                	int maxNeiNum = m_pParticleData->m_iMaxNeighbourNumInOneDir;
	                for(int i=0;i<8;i++)
        	        {
                	        if(inPressure[neiList[index*maxNeiNum+i]]>inPressure[index])
                        	        count_p++;
	                        if(inVelocityU[neiList[index*maxNeiNum+i]]>inVelocityU[index])
        	                        count_u++;
                	        if(inVelocityV[neiList[index*maxNeiNum+i]]>inVelocityV[index])
                        	        count_v++;
	                        if(inVolume[neiList[index*maxNeiNum+i]]>inVolume[index])
        	                        count_rho++;
                	}
	                if(((count_p==8)||(count_u==8)||(count_v==8)||(count_rho==8)||(count_p==0)||(count_u==0)||(count_v==0)||(count_rho==0)))//if local exterme, go to next index
        	                continue;
		}
//Test if the particle is a free boundary particle. If so, use upwind method instead.
		if(m_iFreeBoundary==true)
			if(m_vFillGhost[index]==true)
				continue;

                if(inSoundSpeed[index]==0 || inVolume[index]==0) {

                        cout<<"NOTICE!!!!!!! index="<<index<<", p="<<inPressure[index]
                        <<", vol="<<inVolume[index]<<", velU="<<inVelocityU[index]<<", velV="<<inVelocityV[index]
                        <<", cs="<<inSoundSpeed[index]<<endl;

                        outVolume[index]   = inVolume[index];
                        outPressure[index] = inPressure[index];
                        outVelocityU[index] = inVelocityU[index];
                        outVelocityV[index] = inVelocityV[index];
                        if(m_iDimension==3) outVelocityW[index] = inVelocityW[index];
                        outSoundSpeed[index] = inSoundSpeed[index];
                }
                else {
			LPFOrder[index]=2;
                        // spatial derivatives
                        double Ud[number_of_derivative],Vd[number_of_derivative],Wd[number_of_derivative],Pd[number_of_derivative],Volumed[number_of_derivative];
                        computeSpatialDer(index, offset, computeA, inPressure, inVelocityU, inVelocityV, inVelocityW, inVolume, neiList, neiListSize,
                                                          LPFOrder, Pd, Ud, Vd, Wd, Volumed, number_of_derivative);//output
			if(LPFOrder[index]<2)
			{
//				cout<<"Warning: first order used for particle "<<m_pParticleData->m_vPositionX[index]<<" "<<m_pParticleData->m_vPositionY[index]<<endl;
				continue;
			}
/*			if(1)//tvplot
			{
				m_pParticleData->m_vPError0[index]=sqrt(Pd[0]*Pd[0]+Pd[1]*Pd[1]);
				m_pParticleData->m_vPError1[index]=sqrt(Volumed[0]*Volumed[0]+Volumed[1]*Volumed[1])/inVolume[index]/inVolume[index];
				m_pParticleData->m_vVelError0[index]=sqrt(Ud[0]*Ud[0]+Ud[1]*Ud[1]+Vd[0]*Vd[0]+Vd[1]*Vd[1]);

			}*/
                        if(m_iDimension==3)     timeIntegration(index, m_fDt, gravity, inVolume[index], inVelocityU[index], inVelocityV[index], inVelocityW[index], inPressure[index], inSoundSpeed[index],
                                        Volumed, Ud, Vd, Wd, Pd,
                                                        &outVolume[index], &outVelocityU[index], &outVelocityV[index], &outVelocityW[index], &outPressure[index]); // output 
                        else
                        {
                                double temp1=0,temp2;
                                timeIntegration(index, m_fDt, gravity, inVolume[index], inVelocityU[index], inVelocityV[index], temp1, inPressure[index], inSoundSpeed[index],
                                        Volumed, Ud, Vd, Wd, Pd,
                                                        &outVolume[index], &outVelocityU[index], &outVelocityV[index], &temp2, &outPressure[index]); // output 
                        }
                        if(m_pParticleData->m_iNumberofPellet)
                        {
                                outPressure[index]+=m_fDt*m_pParticleData->m_vDeltaq[index]*(m_pGamma-1);
                        }
                        m_pParticleData->m_vPhi[index]=LPFOrder[index];

                        bool redo = false;
                        if(outPressure[index]<m_fInvalidPressure) redo=true;
                        else if(outVolume[index]!=0 && 1./outVolume[index]<m_fInvalidDensity) redo=true;
                        else if(std::isnan(outVolume[index]) || std::isnan(outVelocityU[index]) || std::isnan(outVelocityV[index]) || std::isnan(outPressure[index])) {
                                cout<<"nan value!!!"<<endl;
                                redo=true;
                        }
                        else if(std::isinf(outVolume[index]) || std::isinf(outVelocityU[index]) || std::isinf(outVelocityV[index]) || std::isinf(outPressure[index])) {
                                cout<<"inf value!!!"<<endl;
                                redo=true;
                        }
                        if(redo) {
                                outVolume[index]   = inVolume[index];
                                outPressure[index] = inPressure[index];
                                outVelocityU[index] = inVelocityU[index];
                                outVelocityV[index] = inVelocityV[index];
                                if(m_iDimension==3)     outVelocityW[index] = inVelocityW[index];
                                outSoundSpeed[index] = inSoundSpeed[index];
                        }
                        else {
                                if(outVolume[index]==0)
                                        outSoundSpeed[index] = 0;
                                else
                                        outSoundSpeed[index] = m_pEOS->getSoundSpeed(outPressure[index],1./outVolume[index]);
                        }
                }
        }
        cout<<"--------------END HyperbolicLPSolver::nodirectionalSplitting()--------------"<<endl;

        return phaseSuccess;
}


bool HyperbolicLPSolver::solve_laxwendroff_fix() {
        cout<<"--------------HyperbolicLPSolver::nodirectionalSplitting_fix()--------------"<<endl;
        // set neighbour list pointers  const int *neiList=m_pParticleData->m_vNeighbourList;
        const int *neiList=m_pParticleData->m_vNeighbourList;
        const int *neiListSize=m_pParticleData->m_vNeighbourListSize;
        // input data pointers
        const double *inVelocityU, *inVelocityV, *inVelocityW, *inPressure=m_pParticleData->m_vPressure, *inVolume=m_pParticleData->m_vVolume, *inSoundSpeed=m_pParticleData->m_vSoundSpeed;
        inVelocityU = m_pParticleData->m_vVelocityU;
        inVelocityV = m_pParticleData->m_vVelocityV;
        if(m_iDimension==3)
                inVelocityW = m_pParticleData->m_vVelocityW;
        // output data pointers
        double *outVelocityU, *outVelocityV, *outVelocityW, *outPressure=m_pParticleData->m_vTemp1Pressure, *outVolume=m_pParticleData->m_vTemp1Volume, *outSoundSpeed=m_pParticleData->m_vTemp1SoundSpeed;
        outVelocityU = m_pParticleData->m_vTemp1VelocityU;
        outVelocityV = m_pParticleData->m_vTemp1VelocityV;
        if(m_iDimension==3)
                outVelocityW = m_pParticleData->m_vTemp1VelocityW;
// set local polynomail order pointers
        int *LPFOrder=m_pParticleData->m_vLPFOrderRight;
// phase_success
        bool phaseSuccess = true;
// gravity is only on the last direction 
        double gravity=m_fGravity;
// set the function pointer to compute the A matrix for QR solver

        void (HyperbolicLPSolver::*computeA) (size_t, const int *, const int*, size_t, size_t, double*, double*);
        int number_of_derivative;
        size_t offset;
        if(m_iDimension==2) {computeA = &HyperbolicLPSolver::computeA2D; number_of_derivative=5; offset = 2;}
        else if(m_iDimension==3) {computeA = &HyperbolicLPSolver::computeA3D; number_of_derivative=9; offset = 3;}
        // iteration start index
        size_t startIndex = m_pParticleData->m_iFluidStartIndex;
        size_t endIndex = startIndex + m_pParticleData->m_iFluidNum;
	bool complain=true;

        // iterate through fluid particles
        #ifdef _OPENMP
        #pragma omp parallel for 
        #endif
        for(size_t index=startIndex; index<endIndex; index++) {
                if(m_iFreeBoundary==true)
                        if(complain && m_vFillGhost[index]==true){
				complain=false;
                                cout<<"Warning: Fixed particle algorithm should not be used with free surface!"<<endl;
		}

                if(inSoundSpeed[index]==0 || inVolume[index]==0) {

                        cout<<"NOTICE!!!!!!! index="<<index<<", p="<<inPressure[index]
                        <<", vol="<<inVolume[index]<<", velU="<<inVelocityU[index]<<", velV="<<inVelocityV[index]
                        <<", cs="<<inSoundSpeed[index]<<endl;

                        outVolume[index]   = inVolume[index];
                        outPressure[index] = inPressure[index];
                        outVelocityU[index] = inVelocityU[index];
                        outVelocityV[index] = inVelocityV[index];
                        if(m_iDimension==3) outVelocityW[index] = inVelocityW[index];
                        outSoundSpeed[index] = inSoundSpeed[index];
                }
                else {
			LPFOrder[index]=2;
                        // spatial derivatives
                        double Ud[number_of_derivative],Vd[number_of_derivative],Wd[number_of_derivative],Pd[number_of_derivative],Volumed[number_of_derivative];
                        computeSpatialDer(index, offset, computeA, inPressure, inVelocityU, inVelocityV, inVelocityW, inVolume, neiList, neiListSize,
                                                          LPFOrder, Pd, Ud, Vd, Wd, Volumed, number_of_derivative);//output
			if(LPFOrder[index]<2)
			{
                                outVolume[index]   = inVolume[index];
                                outPressure[index] = inPressure[index];
                                outVelocityU[index] = inVelocityU[index];
                                outVelocityV[index] = inVelocityV[index];
                                if(m_iDimension==3)     outVelocityW[index] = inVelocityW[index];
                                outSoundSpeed[index] = inSoundSpeed[index];
				
				cout<<"Warning: old states used for particle "<<m_pParticleData->m_vPositionX[index]<<" "<<m_pParticleData->m_vPositionY[index]<<endl;
				continue;
			}
                        if(m_iDimension==3)     timeIntegration_fix(index, m_fDt, gravity, inVolume[index], inVelocityU[index], inVelocityV[index], inVelocityW[index], inPressure[index], inSoundSpeed[index],
                                        Volumed, Ud, Vd, Wd, Pd,
                                                        &outVolume[index], &outVelocityU[index], &outVelocityV[index], &outVelocityW[index], &outPressure[index]); // output 
                        else
                        {
                                double temp1=0,temp2;
                                timeIntegration_fix(index, m_fDt, gravity, inVolume[index], inVelocityU[index], inVelocityV[index], temp1, inPressure[index], inSoundSpeed[index],
                                        Volumed, Ud, Vd, Wd, Pd,
                                                        &outVolume[index], &outVelocityU[index], &outVelocityV[index], &temp2, &outPressure[index]); // output 
                        }
                        m_pParticleData->m_vPhi[index]=LPFOrder[index];

                        bool redo = false;
                        if(outPressure[index]<m_fInvalidPressure) redo=true;
                        else if(outVolume[index]!=0 && 1./outVolume[index]<m_fInvalidDensity) redo=true;
                        else if(std::isnan(outVolume[index]) || std::isnan(outVelocityU[index]) || std::isnan(outVelocityV[index]) || std::isnan(outPressure[index])) {
                                cout<<"nan value!!!"<<endl;
                                redo=true;
                        }
                        else if(std::isinf(outVolume[index]) || std::isinf(outVelocityU[index]) || std::isinf(outVelocityV[index]) || std::isinf(outPressure[index])) {
                                cout<<"inf value!!!"<<endl;
                                redo=true;
                        }
                        if(redo) {
				cout<<"Warning: invalid states for particle "<<m_pParticleData->m_vPositionX[index]<<" "<<m_pParticleData->m_vPositionY[index]<<endl;
                                outVolume[index]   = inVolume[index];
                                outPressure[index] = inPressure[index];
                                outVelocityU[index] = inVelocityU[index];
                                outVelocityV[index] = inVelocityV[index];
                                if(m_iDimension==3)     outVelocityW[index] = inVelocityW[index];
                                outSoundSpeed[index] = inSoundSpeed[index];
                        }
                        else {
                                if(outVolume[index]==0)
                                        outSoundSpeed[index] = 0;
                                else
                                        outSoundSpeed[index] = m_pEOS->getSoundSpeed(outPressure[index],1./outVolume[index]);
                        }
                }
        }
        cout<<"--------------END HyperbolicLPSolver::nodirectionalSplitting_fix()--------------"<<endl;

        return phaseSuccess;
}


void HyperbolicLPSolver::setNeighbourListPointers(int dir, // input
	const int **neiList0, const int **neiList1, // output
	const int **neiListSize0, const int **neiListSize1) { 
	
	if(dir==0) { // x
		*neiList0 = m_pParticleData->m_vNeighbourListRight; 
		*neiList1 = m_pParticleData->m_vNeighbourListLeft;
		*neiListSize0 = m_pParticleData->m_vNeighbourListRightSize; 
		*neiListSize1 = m_pParticleData->m_vNeighbourListLeftSize;
	}
	else if(dir==1) { // y
		*neiList0 = m_pParticleData->m_vNeighbourListNorth; 
		*neiList1 = m_pParticleData->m_vNeighbourListSouth;
		*neiListSize0 = m_pParticleData->m_vNeighbourListNorthSize; 
		*neiListSize1 = m_pParticleData->m_vNeighbourListSouthSize;
	}
	else if(dir==2) { // z (if m_iDimension==2, dir != 2 for sure)
		*neiList0 = m_pParticleData->m_vNeighbourListUp; 
		*neiList1 = m_pParticleData->m_vNeighbourListDown;
		*neiListSize0 = m_pParticleData->m_vNeighbourListUpSize; 
		*neiListSize1 = m_pParticleData->m_vNeighbourListDownSize;	
	}
	else 
		assert(false);

}


void HyperbolicLPSolver::setInAndOutDataPointers(int phase, int dir,
	const double** inVelocity, const double** inPressure, const double** inVolume, const double** inSoundSpeed, 
	double** outVelocity, double** outPressure, double** outVolume, double** outSoundSpeed) {
	
	// assign pressure, volume, and sound_speed pointers
	if(phase==0) { // input: original output:temp1
		*inPressure   = m_pParticleData->m_vPressure;
		*inVolume     = m_pParticleData->m_vVolume;
		*inSoundSpeed = m_pParticleData->m_vSoundSpeed;
		
		*outPressure   = m_pParticleData->m_vTemp1Pressure;
		*outVolume     = m_pParticleData->m_vTemp1Volume;
		*outSoundSpeed = m_pParticleData->m_vTemp1SoundSpeed;	
	}
	else if(phase==1 || phase==3) { // input:temp1 output:temp2
		*inPressure   = m_pParticleData->m_vTemp1Pressure;
		*inVolume     = m_pParticleData->m_vTemp1Volume;
		*inSoundSpeed = m_pParticleData->m_vTemp1SoundSpeed;
		
		*outPressure   = m_pParticleData->m_vTemp2Pressure;
		*outVolume     = m_pParticleData->m_vTemp2Volume;
		*outSoundSpeed = m_pParticleData->m_vTemp2SoundSpeed;	
	}
	else if(phase==2 || phase==4){ // input:temp2 output: temp1
		*inPressure   = m_pParticleData->m_vTemp2Pressure;
		*inVolume     = m_pParticleData->m_vTemp2Volume;
		*inSoundSpeed = m_pParticleData->m_vTemp2SoundSpeed;
		
		*outPressure   = m_pParticleData->m_vTemp1Pressure;
		*outVolume	  = m_pParticleData->m_vTemp1Volume;
		*outSoundSpeed = m_pParticleData->m_vTemp1SoundSpeed;	
	}	
	else assert(false);
	
	// assign velocity pointers
	if(m_iDimension==2) {
		if(phase==0 || phase==1) { // input: original output:temp2
			if(dir==0) {
				*inVelocity = m_pParticleData->m_vVelocityU;
				*outVelocity = m_pParticleData->m_vTemp2VelocityU;
			}
			else if(dir==1) {
				*inVelocity = m_pParticleData->m_vVelocityV;
				*outVelocity = m_pParticleData->m_vTemp2VelocityV;	
			}	
		}	
		else if(phase==2){ // input:temp2 output: temp1
			if(dir==0) { // u v u
				*inVelocity = m_pParticleData->m_vTemp2VelocityU;
				*outVelocity = m_pParticleData->m_vTemp1VelocityU;
				// swap pointers so that temp1 will contain new info
				swap(m_pParticleData->m_vTemp1VelocityV, m_pParticleData->m_vTemp2VelocityV);	
			}
			else if(dir==1) { // v u v
				*inVelocity = m_pParticleData->m_vTemp2VelocityV;
				*outVelocity = m_pParticleData->m_vTemp1VelocityV;
				// swap pointers so that temp1 will contain new info
				swap(m_pParticleData->m_vTemp1VelocityU, m_pParticleData->m_vTemp2VelocityU);
			}		
		}	
		else assert(false);	
	}
	else if(m_iDimension==3) {
		if(phase==0 || phase==1 || phase==2) { // input: original output:temp1
			if(dir==0) {
				*inVelocity = m_pParticleData->m_vVelocityU;
				*outVelocity = m_pParticleData->m_vTemp1VelocityU;
			}
			else if(dir==1) {
				*inVelocity = m_pParticleData->m_vVelocityV;
				*outVelocity = m_pParticleData->m_vTemp1VelocityV;	
			}
			else if(dir==2) {
				*inVelocity = m_pParticleData->m_vVelocityW;
				*outVelocity = m_pParticleData->m_vTemp1VelocityW;	
			}
		}	
		else if(phase==3){ // input:temp1 output: temp2
			if(dir==0) { 
				*inVelocity = m_pParticleData->m_vTemp1VelocityU;
				*outVelocity = m_pParticleData->m_vTemp2VelocityU;
				// swap pointers so that temp2 will contain new info
				swap(m_pParticleData->m_vTemp1VelocityV, m_pParticleData->m_vTemp2VelocityV);
				swap(m_pParticleData->m_vTemp1VelocityW, m_pParticleData->m_vTemp2VelocityW);
			}
			else if(dir==1) { 
				*inVelocity = m_pParticleData->m_vTemp1VelocityV;
				*outVelocity = m_pParticleData->m_vTemp2VelocityV;
				// swap pointers so that temp2 will contain new info
				swap(m_pParticleData->m_vTemp1VelocityU, m_pParticleData->m_vTemp2VelocityU);
				swap(m_pParticleData->m_vTemp1VelocityW, m_pParticleData->m_vTemp2VelocityW);
			}
			else if(dir==2) { 
				*inVelocity = m_pParticleData->m_vTemp1VelocityW;
				*outVelocity = m_pParticleData->m_vTemp2VelocityW;
				// swap pointers so that temp2 will contain new info
				swap(m_pParticleData->m_vTemp1VelocityU, m_pParticleData->m_vTemp2VelocityU);
				swap(m_pParticleData->m_vTemp1VelocityV, m_pParticleData->m_vTemp2VelocityV);
			}
		}
		else if(phase==4){ // input:temp2 output: temp1
			if(dir==0) { 
				*inVelocity = m_pParticleData->m_vTemp2VelocityU;
				*outVelocity = m_pParticleData->m_vTemp1VelocityU;
				// swap pointers so that temp1 will contain new info
				swap(m_pParticleData->m_vTemp1VelocityV, m_pParticleData->m_vTemp2VelocityV);
				swap(m_pParticleData->m_vTemp1VelocityW, m_pParticleData->m_vTemp2VelocityW);
			}
			else if(dir==1) { 
				*inVelocity = m_pParticleData->m_vTemp2VelocityV;
				*outVelocity = m_pParticleData->m_vTemp1VelocityV;
				// swap pointers so that temp1 will contain new info
				swap(m_pParticleData->m_vTemp1VelocityU, m_pParticleData->m_vTemp2VelocityU);
				swap(m_pParticleData->m_vTemp1VelocityW, m_pParticleData->m_vTemp2VelocityW);
			}
			else if(dir==2) { 
				*inVelocity = m_pParticleData->m_vTemp2VelocityW;
				*outVelocity = m_pParticleData->m_vTemp1VelocityW;
				// swap pointers so that temp1 will contain new info
				swap(m_pParticleData->m_vTemp1VelocityU, m_pParticleData->m_vTemp2VelocityU);
				swap(m_pParticleData->m_vTemp1VelocityV, m_pParticleData->m_vTemp2VelocityV);
			}
		}
		else assert(false);	
	}

}


void HyperbolicLPSolver::setLPFOrderPointers(int dir, // input
	int** LPFOrder0, int** LPFOrder1, vector<int*>& LPFOrderOther) { // output
	
	if(dir==0) { // x
		*LPFOrder0 = m_pParticleData->m_vLPFOrderRight; // this direction
		*LPFOrder1 = m_pParticleData->m_vLPFOrderLeft;
		
		LPFOrderOther.push_back(m_pParticleData->m_vLPFOrderNorth); // other directions
		LPFOrderOther.push_back(m_pParticleData->m_vLPFOrderSouth);
		if(m_iDimension==3) {
			LPFOrderOther.push_back(m_pParticleData->m_vLPFOrderUp); 
			LPFOrderOther.push_back(m_pParticleData->m_vLPFOrderDown);
		}
	}
	else if(dir==1) { // y
		*LPFOrder0 = m_pParticleData->m_vLPFOrderNorth; // this direction
		*LPFOrder1 = m_pParticleData->m_vLPFOrderSouth;
		
		LPFOrderOther.push_back(m_pParticleData->m_vLPFOrderRight); // other directions
		LPFOrderOther.push_back(m_pParticleData->m_vLPFOrderLeft);
		if(m_iDimension==3) {
			LPFOrderOther.push_back(m_pParticleData->m_vLPFOrderUp); 
			LPFOrderOther.push_back(m_pParticleData->m_vLPFOrderDown);
		}
	}
	else if(dir==2) { // z
		*LPFOrder0 = m_pParticleData->m_vLPFOrderUp; // this direction
		*LPFOrder1 = m_pParticleData->m_vLPFOrderDown;
		
		LPFOrderOther.push_back(m_pParticleData->m_vLPFOrderRight); // other directions
		LPFOrderOther.push_back(m_pParticleData->m_vLPFOrderLeft);
		LPFOrderOther.push_back(m_pParticleData->m_vLPFOrderNorth); 
		LPFOrderOther.push_back(m_pParticleData->m_vLPFOrderSouth);
	}
	else
		assert(false);

}
void HyperbolicLPSolver::computeSpatialDer(int dir, size_t index, // input 
	int offset, void (HyperbolicLPSolver::*computeA) (size_t, const int *, const int*, size_t, size_t,double*, double*),
	const double* inPressure, const double* inVelocity,
	const int *neighbourList, const int *neighbourListSize,int additional,
	int* LPFOrder, double* vel_d, double* vel_dd, double* p_d, double* p_dd) { // output
	
	//bool sufficientRank = false;

	// the initial value of the try-error process of finding a good A matrix with sufficient rank
	// these two variables will be increased if necessary in the while loop
	size_t numRow2nd = m_iNumRow2ndOrder+additional;
	size_t numRow1st = m_iNumRow1stOrder;
	double distance;	
	//cout<<"-------HyperbolicLPSolver::computeSpatialDer()-------"<<endl;
	//cout<<"numRow2nd="<<numRow2nd<<endl;
	//cout<<"numRow1st="<<numRow1st<<endl;

	while(true) {
	//while(!sufficientRank) {
		
		// decide row and column number based on LPFOrder (and numRow2nd, numRow1st) and the total number of neighbours
		size_t numRow, numCol;
		computeNumRowAndNumColAndLPFOrder(index, neighbourList, neighbourListSize, numRow2nd, numRow1st, // input
										  LPFOrder, &numRow, &numCol); // output	

		if(LPFOrder[index] == 0) { 
			*vel_d  = 0; *vel_dd = 0; *p_d  = 0; *p_dd = 0;
			//return;
			break;
		}
		
		// compute A
		double A[numRow*numCol];
		(this->*computeA)(index, neighbourList, LPFOrder, numRow, numCol, // input
						  A, &distance); // output
		double b[numRow]; 
		computeB(index, neighbourList, numRow, inPressure, // input: pressure first
				 b); // output
        
    // printf("In %d compute A for number %d is %.5f,%.5f,%.5f,%.5f\n",m_iCount, index,A[0],A[1],A[2],A[3]);

       //  printf("In %d compute B for number %d is %.5f,%.5f,%.5f\n",m_iCount,index,b[0],b[1],b[2]);

		QRSolver qrSolver(numRow,numCol,A);
		
		double result[numCol];
		int info = qrSolver.solve(result,b);
// printf("In %d compute result for number %d is %.5f,%.5f\n",m_iCount,index,result[0],result[1]);
		if(info!=0) { // then need to recompute A
//                        cout<<"index="<<index<<", numRow="<<numRow<<", rank="<<info<<endl;
            if(LPFOrder[index]==2) numRow2nd++;
			else if(LPFOrder[index]==1) numRow1st++;	
		}	
		else {
			//sufficientRank = true;
			
			*p_d = result[dir]/distance; //dir=0 (x), dir=1(y), dir=2(z)
			*p_dd = LPFOrder[index]==2? result[dir+offset]/distance/distance:0;

			computeB(index, neighbourList, numRow, inVelocity, // input: velocity comes second 
					 b); // output (rewrite array b)	
			
			qrSolver.solve(result,b);

			*vel_d = result[dir]/distance; //dir=0 (x), dir=1(y), dir=2(z)
			*vel_dd = LPFOrder[index]==2? result[dir+offset]/distance/distance:0;
			
			if(std::isnan(*p_d) || std::isnan(*p_dd) || std::isnan(*vel_d) || std::isnan(*vel_dd) ||
			   std::isinf(*p_d) || std::isinf(*p_dd) || std::isinf(*vel_d) || std::isinf(*vel_dd)) {
				if(LPFOrder[index]==2) numRow2nd++;
				else if(LPFOrder[index]==1) numRow1st++;
			}
			else {

				{
					if(numRow>10)	printf("Notice: numRow=%zu for index = %zu\n",numRow,index);
				}

				break;
			}
		}

	}
	
}

void HyperbolicLPSolver::computeSpatialDer(size_t index,  size_t offset, void (HyperbolicLPSolver::*computeA) (size_t, const int *, const int*, size_t, size_t,double*, double*),  const double* inPressure, const double* inVelocityU, const double* inVelocityV, const double* inVelocityW, const double* inVolume, const int *neighbourList, const int *neighbourListSize,
                                                          int *LPFOrder, double* Pd, double* Ud, double* Vd, double* Wd, double* Volumed, int number_of_derivative)
{
//	double threshold=0.1;
	double dis,first,second;
	double max_dis=0;
        for(int i=0;i<number_of_derivative;i++)
        {
        	Pd[i]=0.0;
                Ud[i]=0.0;
                Vd[i]=0.0;
                Wd[i]=0.0;
		Volumed[i]=0.0;
        }
	double distance;
        size_t numRow2nd = m_iNumRow2ndOrder;
        size_t numRow1st = m_iNumRow1stOrder;
        while(true) {
                size_t numRow, numCol;
                computeNumRowAndNumColAndLPFOrder(index, neighbourList, neighbourListSize, numRow2nd, numRow1st, // input

  LPFOrder, &numRow, &numCol); // output  
                if(LPFOrder[index] == 0) {
                        break;
                }

                double A[numRow*numCol];
                (this->*computeA)(index, neighbourList, LPFOrder, numRow, numCol, // input
                                                  A, &distance); // output

                double b[numRow];
                computeB(index, neighbourList, numRow, inPressure, // input: pressure first
                                 b); // output

                QRSolver qrSolver(numRow,numCol,A);

                double result[numCol];
                int info = qrSolver.solve(result,b);
                if(info!=0) { // then need to recompute A
                        //cout<<"index="<<index<<", numRow="<<numRow<<", rank="<<info<<endl;
                        if(LPFOrder[index]==2) numRow2nd++;
                        else if(LPFOrder[index]==1) numRow1st++;
                }
                else {

			first=0;
			second=0;
                        for(size_t i=0;i<numCol;i++)
			{
				if (i<offset)
				{
                                	Pd[i]=result[i]/distance;
					first=first+result[i];
				}
				else
				{
					Pd[i]=result[i]/distance/distance;
					second=second+result[i];
				}
			}
			dis=second/first;
			if (dis>max_dis)	max_dis=dis;

                        computeB(index, neighbourList, numRow, inVelocityU, // input: velocity comes second 
                                         b); // output (rewrite array b)        

                        qrSolver.solve(result,b);
                        first=0;
                        second=0;
                        for(size_t i=0;i<numCol;i++)
                        {
                                if (i<offset)
                                {
                                        Ud[i]=result[i]/distance;
                                        first=first+result[i];
                                }
                                else
                                {
                                        Ud[i]=result[i]/distance/distance;
                                        second=second+result[i];
                                }
                        }
                        dis=second/first;
                        if (dis>max_dis)        max_dis=dis;
                        computeB(index, neighbourList, numRow, inVelocityV, // input: velocity comes second 
                                         b); // output (rewrite array b)        

                        qrSolver.solve(result,b);
                        first=0;
                        second=0;
                        for(size_t i=0;i<numCol;i++)
                        {
                                if (i<offset)
                                {
                                        Vd[i]=result[i]/distance;
                                        first=first+result[i];
                                }
                                else
                                {
                                        Vd[i]=result[i]/distance/distance;
                                        second=second+result[i];
                                }
                        }
                        dis=second/first;
                        if (dis>max_dis)        max_dis=dis;

			if(m_iDimension==3)
			{
	                        computeB(index, neighbourList, numRow, inVelocityW, // input: velocity comes second 
        	                                 b); // output (rewrite array b)        
	
        	                qrSolver.solve(result,b);
            	        	first=0;
               	        	second=0;
                        	for(size_t i=0;i<numCol;i++)
                        	{
                                	if (i<offset)
                                	{
                                        	Wd[i]=result[i]/distance;
                                        	first=first+result[i];
                                	}
                                	else
                                	{
                                        	Wd[i]=result[i]/distance/distance;
                                        	second=second+result[i];
                                	}
                        	}
                        	dis=second/first;
                        	if (dis>max_dis)        max_dis=dis;
			}
                        computeB(index, neighbourList, numRow, inVolume, // input: volume comes last 
                                         b); // output (rewrite array b)        

                        qrSolver.solve(result,b);
                        first=0;
                        second=0;
                        for(size_t i=0;i<numCol;i++)
                        {
                                if (i<offset)
                                {
                                        Volumed[i]=result[i]/distance;
                                        first=first+result[i];
                                }
                                else
                                {
                                        Volumed[i]=result[i]/distance/distance;
                                        second=second+result[i];
                                }
                        }
                        dis=second/first;
                        if (dis>max_dis)        max_dis=dis;
//			if (max_dis>threshold)	LPFOrder[index]=1;
			int fail=0;
			for(int i=0; i<number_of_derivative;i++)
	                        if(std::isnan(Pd[i]) || std::isnan(Ud[i]) || std::isnan(Vd[i]) || std::isnan(Wd[i]) || std::isnan(Volumed[i]) ||
                           std::isinf(Pd[i]) || std::isinf(Ud[i]) || std::isinf(Vd[i]) || std::isinf(Wd[i])|| std::isnan(Volumed[i])) {
				fail=fail+1;
				}
			if(fail)
			{
                                if(LPFOrder[index]==2) numRow2nd++;
                                else if(LPFOrder[index]==1) numRow1st++;
			}
			else
			{
				break;
			}
		}
	}
}
//support function for density_derivative
void HyperbolicLPSolver::computeSpatialDer(size_t index, size_t offset, void (HyperbolicLPSolver::*computeA) (size_t, const int *, const int*, size_t, size_t,double*, double*),
                                                  const double* inVolume, const int *neighbourList, const int *neighbourListSize,
                                                  int *LPFOrder, double* volume_x, double* volume_y, double* volume_z)
{
	*volume_x=0;
	*volume_y=0;
	*volume_z=0;
	int order=LPFOrder[index];
	LPFOrder[index]=1;
        double distance;
        size_t numRow2nd = 4*m_iNumRow2ndOrder;
        size_t numRow1st = 4*m_iNumRow1stOrder;
	if (neighbourListSize[index]>(int)numRow1st)
		numRow1st=neighbourListSize[index];
        while(true) {
                size_t numRow, numCol;
                computeNumRowAndNumColAndLPFOrder(index, neighbourList, neighbourListSize, numRow2nd, numRow1st, // input

  LPFOrder, &numRow, &numCol); // output 

		if(LPFOrder[index]==0)
		{
			LPFOrder[index]=order;
			printf("Warning: cannot calculate derivative of volume for %zu th particle. SPH density estimator may be inaccurate.\n",index);
			break;
		}
                double A[numRow*numCol];
                (this->*computeA)(index, neighbourList, LPFOrder, numRow, numCol, // input
                                                  A, &distance); // output
                double b[numRow];
                computeB(index, neighbourList, numRow, inVolume, // input
                                 b); // output

                QRSolver qrSolver(numRow,numCol,A);

                double result[numCol];
                int info = qrSolver.solve(result,b);
                if(info!=0) { // then need to recompute A
                        //cout<<"index="<<index<<", numRow="<<numRow<<", rank="<<info<<endl;
                        if(LPFOrder[index]==2) numRow2nd++;
                        else if(LPFOrder[index]==1) numRow1st++;
                }
                else {
			*volume_x=result[0];
			*volume_y=result[1];
			if(offset==3)
				*volume_z=result[2];
			break;
		}
	}
	LPFOrder[index]=order;
}
void HyperbolicLPSolver::computeNumRowAndNumColAndLPFOrder(size_t index, // input
    const int *neighbourList, const int *neighbourListSize, size_t numRow2nd, size_t numRow1st,
	int* LPFOrder, size_t *numRow, size_t *numCol) { // output
	
	// compute the numRow and numCol for matrix A
	size_t totalNeiNum = neighbourListSize[index];

	if(LPFOrder[index] == 2) {
		if(totalNeiNum >= numRow2nd) { // if the total number of neighbour >= current numRow2nd
			(*numRow) = numRow2nd;
			(*numCol) = m_iNumCol2ndOrder; 
		}
		else LPFOrder[index] = 1; // if no enough neighbour -> reduce LPFOrder to 1
	}

	if(LPFOrder[index] == 1) { 
		if(totalNeiNum >= numRow1st) { // if the total number of neighbour >= current numRow1st
			(*numRow) = numRow1st;
			(*numCol) = m_iNumCol1stOrder; 
		}
		else LPFOrder[index] = 0;
	}

	if(LPFOrder[index] == 0) {
		(*numRow) = 0;
		(*numCol) = 0;
	}

	//cout<<"-------HyperbolicLPSolver::computeNumRowAndNumColAndLPFOrder()-------"<<endl;

}


void HyperbolicLPSolver::computeA2D(size_t index, const int *neighbourList, 
							  	    const int* LPFOrder, size_t numRow, size_t numCol,
								    double *A, double *dis) { // output 	
	//cout<<"--------------HyperbolicLPSolver::computeA2D()--------------"<<endl;
	//cout<<"index="<<index<<endl;
	
	size_t maxNeiNum = m_pParticleData->m_iMaxNeighbourNumInOneDir;
        double distance = sqrt((m_pParticleData->m_vPositionX[neighbourList[index*maxNeiNum+numRow/4]] - m_pParticleData->m_vPositionX[index]) * (m_pParticleData->m_vPositionX[neighbourList[index*maxNeiNum+numRow/4]] - m_pParticleData->m_vPositionX[index]) + (m_pParticleData->m_vPositionY[neighbourList[index*maxNeiNum+numRow/4]] - m_pParticleData->m_vPositionY[index]) * (m_pParticleData->m_vPositionY[neighbourList[index*maxNeiNum+numRow/4]] - m_pParticleData->m_vPositionY[index]));
	if(LPFOrder[index] == 1) {
		for(size_t i=0; i<numRow; i++) { // Note that the neighbour list does not contain the particle itself 
			int neiIndex = neighbourList[index*maxNeiNum+i];	
				
			double h = (m_pParticleData->m_vPositionX[neiIndex] - m_pParticleData->m_vPositionX[index])/distance;
			double k = (m_pParticleData->m_vPositionY[neiIndex] - m_pParticleData->m_vPositionY[index])/distance;

			A[i]            = h;
			A[i + 1*numRow] = k;	

			//cout<<A[i]<<"	"<<A[i + 1*numRow]<<endl;
		}
	}
	else if(LPFOrder[index] == 2) {
		for(size_t i=0; i<numRow; i++) { // Note that the neighbour list does not contain the particle itself
			int neiIndex = neighbourList[index*maxNeiNum+i];	
				
			double h = (m_pParticleData->m_vPositionX[neiIndex] - m_pParticleData->m_vPositionX[index])/distance;
			double k = (m_pParticleData->m_vPositionY[neiIndex] - m_pParticleData->m_vPositionY[index])/distance;
			
			A[i]            = h;
			A[i + 1*numRow] = k;
			A[i + 2*numRow]	= 0.5*h*h;
			A[i + 3*numRow]	= 0.5*k*k;
			A[i + 4*numRow]	= h*k;
			

			//cout<<A[i]<<"	"<<A[i + 1*numRow]<<"	"<<A[i + 2*numRow]<<"	"
			//	<<A[i + 3*numRow]<<"	"<<A[i + 4*numRow]<<endl;
		}
	}	
	(*dis)=distance;	
	//cout<<"------------------------------------------------------------"<<endl;
}

void HyperbolicLPSolver::computeA3D(size_t index, const int *neighbourList, 
								    const int* LPFOrder, size_t numRow, size_t numCol,
								    double *A, double *dis) { // output 	
	size_t maxNeiNum = m_pParticleData->m_iMaxNeighbourNumInOneDir;
        double distance = sqrt((m_pParticleData->m_vPositionX[neighbourList[index*maxNeiNum]] - m_pParticleData->m_vPositionX[index]) * (m_pParticleData->m_vPositionX[neighbourList[index*maxNeiNum]] - m_pParticleData->m_vPositionX[index]) + (m_pParticleData->m_vPositionY[neighbourList[index*maxNeiNum]] - m_pParticleData->m_vPositionY[index]) * (m_pParticleData->m_vPositionY[neighbourList[index*maxNeiNum]] - m_pParticleData->m_vPositionY[index]) + (m_pParticleData->m_vPositionZ[neighbourList[index*maxNeiNum]] - m_pParticleData->m_vPositionZ[index]) * (m_pParticleData->m_vPositionZ[neighbourList[index*maxNeiNum]] - m_pParticleData->m_vPositionZ[index]));
	
	if(LPFOrder[index] == 1) {
		for(size_t i=0; i<numRow; i++) { // Note that the neighbour list does not contain the particle itself
			int neiIndex = neighbourList[index*maxNeiNum+i];	
				
			double h = (m_pParticleData->m_vPositionX[neiIndex] - m_pParticleData->m_vPositionX[index])/distance;
			double k = (m_pParticleData->m_vPositionY[neiIndex] - m_pParticleData->m_vPositionY[index])/distance;
			double l = (m_pParticleData->m_vPositionZ[neiIndex] - m_pParticleData->m_vPositionZ[index])/distance;

			A[i]            = h;
			A[i + 1*numRow] = k;
			A[i + 2*numRow] = l;
		}
	}
	else if(LPFOrder[index] == 2) {
		for(size_t i=0; i<numRow; i++) { // Note that the neighbour list does not contain the particle itself
			int neiIndex = neighbourList[index*maxNeiNum+i];	
				
			double h = (m_pParticleData->m_vPositionX[neiIndex] - m_pParticleData->m_vPositionX[index])/distance;
			double k = (m_pParticleData->m_vPositionY[neiIndex] - m_pParticleData->m_vPositionY[index])/distance;
			double l = (m_pParticleData->m_vPositionZ[neiIndex] - m_pParticleData->m_vPositionZ[index])/distance;

			A[i]            = h;
			A[i + 1*numRow] = k;
			A[i + 2*numRow] = l;
			A[i + 3*numRow] = 0.5*h*h;
			A[i + 4*numRow] = 0.5*k*k;
			A[i + 5*numRow] = 0.5*l*l;
			A[i + 6*numRow] = h*k;
			A[i + 7*numRow] = h*l;
			A[i + 8*numRow] = k*l;
		}
	}
	(*dis)=distance;
}


void HyperbolicLPSolver::computeB(size_t index, const int *neighbourList, size_t numRow, const double* inData, 
								  double *b) { // output 	
	
	//cout<<"--------------HyerbolicLPSolver::computeB()--------------"<<endl;
	//cout<<"index="<<index<<endl;

	size_t maxNeiNum = m_pParticleData->m_iMaxNeighbourNumInOneDir;
	for(size_t i=0; i<numRow; i++) { 
		int neiIndex = neighbourList[index*maxNeiNum+i];	
		b[i] = inData[neiIndex] - inData[index];

		//cout<<b[i]<<endl;
	}	

	//cout<<"----------------------------------------------------------"<<endl;

}

void HyperbolicLPSolver::timeIntegration(
	double realDt, double multiplier1st, double multiplier2nd, 
	double gravity, double inVolume, double inVelocity, double inPressure, double inSoundSpeed, 
	double vel_d_0, double vel_dd_0, double p_d_0, double p_dd_0,
	double vel_d_1, double vel_dd_1, double p_d_1, double p_dd_1,
	double* outVolume, double* outVelocity, double* outPressure) { // output
	
	// TODO Note that this coeff K only works for Poly and Spoly EOS!!!!!!!
	double K = inSoundSpeed*inSoundSpeed/inVolume/inVolume; 
	//double K;
	//if(inSoundSpeed==0 && inVloume==0) K = 1;
	//else K = inSoundSpeed*inSoundSpeed/inVolume/inVolume;
	
	// Pt
	double Pt1st = -0.5*inVolume*K*(vel_d_0+vel_d_1) + 0.5*inVolume*sqrt(K)*(p_d_0-p_d_1);
	double Pt2nd = -inVolume*inVolume*pow(K,1.5)*(vel_dd_0-vel_dd_1) + inVolume*inVolume*K*(p_dd_0+p_dd_1);
	double Pt = multiplier1st*Pt1st + multiplier2nd*Pt2nd;
	
	// Vt
	double Vt = -Pt/K;
	

	// VELt
	double VELt1st = 0.5*inVolume*sqrt(K)*(vel_d_0-vel_d_1) - 0.5*inVolume*(p_d_0+p_d_1);
	double VELt2nd = inVolume*inVolume*K*(vel_dd_0+vel_dd_1) - inVolume*inVolume*sqrt(K)*(p_dd_0-p_dd_1);
	double VELt = multiplier1st*VELt1st + multiplier2nd*VELt2nd;

	// Note that the data pointers of in and out are different!!!!!!!
	(*outVolume)   = inVolume   + realDt*Vt;
	(*outPressure) = inPressure + realDt*Pt;
	(*outVelocity) = inVelocity + realDt*(VELt+gravity);	
	
	if(std::isnan(*outVolume) || std::isinf(*outVolume) || 
	   std::isnan(*outPressure) || std::isinf(*outPressure) ||
	   std::isnan(*outVelocity) || std::isinf(*outVelocity)) {
		assert(false);   
	}

}
void HyperbolicLPSolver::timeIntegration(int index, double Dt, double gravity, double inVolume, double inVelocityU, double inVelocityV, double inVelocityW, double inPressure, double inSoundSpeed,
                                      double* Volumed, double* Ud, double* Vd, double *Wd, double *Pd,
                                                        double* outVolume, double* outVelocityU, double* outVelocityV, double* outVelocityW, double* outPressure){
//      double gamma=inSoundSpeed*inSoundSpeed/inVolume/inPressure;
        double gamma = m_pGamma;
        double Pinf=m_pPinf;

        // TODO Note that this functions only works for Poly EOS!!!!!!!
        if(m_iDimension==3)
        {

                double div=Ud[0]+Vd[1]+Wd[2];
                double cross=Ud[0]*Ud[0]+Vd[1]*Vd[1]+Wd[2]*Wd[2]+2*Ud[1]*Vd[0]+2*Ud[2]*Wd[0]+2*Vd[2]*Wd[1];
                double Volumet=inVolume*div;
                double VelocityUt=-inVolume*Pd[0];
                double VelocityVt=-inVolume*Pd[1];
                double VelocityWt=-inVolume*Pd[2];
                double Pt=-gamma*(inPressure+Pinf)*div;
                double Volumett=inVolume*(div*div-Volumed[0]*Pd[0]-Volumed[1]*Pd[1]-Volumed[2]*Pd[2]-inVolume*(Pd[3]+Pd[4]+Pd[5])-cross);
                double VelocityUtt=inVolume*((gamma-1)*Pd[0]*div+gamma*(inPressure+Pinf)*(Ud[3]+Vd[6]+Wd[7])+Ud[0]*Pd[0]+Vd[0]*Pd[1]+Wd[0]*Pd[2]);
                double VelocityVtt=inVolume*((gamma-1)*Pd[1]*div+gamma*(inPressure+Pinf)*(Ud[6]+Vd[4]+Wd[8])+Ud[1]*Pd[0]+Vd[1]*Pd[1]+Wd[1]*Pd[2]);
                double VelocityWtt=inVolume*((gamma-1)*Pd[2]*div+gamma*(inPressure+Pinf)*(Ud[7]+Vd[8]+Wd[5])+Ud[2]*Pd[0]+Vd[2]*Pd[1]+Wd[2]*Pd[2]);
                double Ptt=gamma*gamma*(inPressure+Pinf)*div*div+gamma*(inPressure+Pinf)*(Volumed[0]*Pd[0]+Volumed[1]*Pd[1]+Volumed[2]*Pd[2]+inVolume*(Pd[3]+Pd[4]+Pd[5])+cross);
                (*outVolume)=inVolume+Dt*Volumet+0.5*Dt*Dt*Volumett;
                (*outVelocityU)=inVelocityU+Dt*VelocityUt+0.5*Dt*Dt*VelocityUtt;
                (*outVelocityV)=inVelocityV+Dt*VelocityVt+0.5*Dt*Dt*VelocityVtt+Dt*gravity;//TODO: MODIFIED GRAVITY DIRECTION
                (*outVelocityW)=inVelocityW+Dt*VelocityWt+0.5*Dt*Dt*VelocityWtt;
                (*outPressure)=inPressure+Dt*Pt+0.5*Dt*Dt*Ptt;
                if(std::isnan(*outVolume) || std::isinf(*outVolume) ||
                   std::isnan(*outPressure) || std::isinf(*outPressure) ||
                 std::isnan(*outVelocityU) || std::isinf(*outVelocityU) ||std::isnan(*outVelocityV) || std::isinf(*outVelocityV) ||std::isnan(*outVelocityW) || std::isinf(*outVelocityW)) {
                        assert(false);
                }

        }
        if(m_iDimension==2)
        {

                double div=Ud[0]+Vd[1];
                double cross=Ud[0]*Ud[0]+Vd[1]*Vd[1]+2*Ud[1]*Vd[0];
                double Volumet=inVolume*div;
                double VelocityUt=-inVolume*Pd[0];
                double VelocityVt=-inVolume*Pd[1];
                double Pt=-gamma*(inPressure+Pinf)*div;
                double Volumett=inVolume*(div*div-Volumed[0]*Pd[0]-Volumed[1]*Pd[1]-inVolume*(Pd[2]+Pd[3])-cross);
                double VelocityUtt=inVolume*((gamma-1)*Pd[0]*div+gamma*(inPressure+Pinf)*(Ud[2]+Vd[4])+Ud[0]*Pd[0]+Vd[0]*Pd[1]);
                double VelocityVtt=inVolume*((gamma-1)*Pd[1]*div+gamma*(inPressure+Pinf)*(Ud[4]+Vd[3])+Ud[1]*Pd[0]+Vd[1]*Pd[1]);
                double Ptt=gamma*gamma*(inPressure+Pinf)*div*div+gamma*(inPressure+Pinf)*(Volumed[0]*Pd[0]+Volumed[1]*Pd[1]+inVolume*(Pd[2]+Pd[3])+cross);
                (*outVolume)=inVolume+Dt*Volumet+0.5*Dt*Dt*Volumett;
                (*outVelocityU)=inVelocityU+Dt*VelocityUt+0.5*Dt*Dt*VelocityUtt;
                (*outVelocityV)=inVelocityV+Dt*VelocityVt+0.5*Dt*Dt*VelocityVtt+Dt*gravity;

                (*outPressure)=inPressure+Dt*Pt+0.5*Dt*Dt*Ptt;
                if(std::isnan(*outVolume) || std::isinf(*outVolume) ||
                   std::isnan(*outPressure) || std::isinf(*outPressure) ||
                 std::isnan(*outVelocityU) || std::isinf(*outVelocityU) ||std::isnan(*outVelocityV) || std::isinf(*outVelocityV)) {
                        assert(false);
                }
                Pd[3]=gamma*(inPressure+Pinf)*(inVolume*(Pd[2]+Pd[3]));
                Pd[4]=gamma*gamma*(inPressure+Pinf)*div*div+gamma*(inPressure+Pinf)*(Volumed[0]*Pd[0]+Volumed[1]*Pd[1]+cross);
                Pd[2]=-gamma*(inPressure+Pinf)*div;

        }
}


void HyperbolicLPSolver::timeIntegration_fix(int index, double Dt, double gravity, double inVolume, double inVelocityU, double inVelocityV, double inVelocityW, double inPressure, double inSoundSpeed,
                                      double* Volumed, double* Ud, double* Vd, double *Wd, double *Pd,
                                                        double* outVolume, double* outVelocityU, double* outVelocityV, double* outVelocityW, double* outPressure){
//      double gamma=inSoundSpeed*inSoundSpeed/inVolume/inPressure;
        double gamma = m_pGamma;
        double Pinf=m_pPinf;

        // TODO Note that this functions only works for Poly EOS!!!!!!!
        double V=inVolume,u=inVelocityU,v=inVelocityV,w=inVelocityW,p=inPressure;

        if(m_iDimension==3)
        {
		double Vx=Volumed[0],Vy=Volumed[1],Vz=Volumed[2],Vxx=Volumed[3],Vyy=Volumed[4],Vzz=Volumed[5],Vxy=Volumed[6],Vxz=Volumed[7],Vyz=Volumed[8];
		double ux=Ud[0],uy=Ud[1],uz=Ud[2],uxx=Ud[3],uyy=Ud[4],uzz=Ud[5],uxy=Ud[6],uxz=Ud[7],uyz=Ud[8];
                double vx=Vd[0],vy=Vd[1],vz=Vd[2],vxx=Vd[3],vyy=Vd[4],vzz=Vd[5],vxy=Vd[6],vxz=Vd[7],vyz=Vd[8];
                double wx=Wd[0],wy=Wd[1],wz=Wd[2],wxx=Wd[3],wyy=Wd[4],wzz=Wd[5],wxy=Wd[6],wxz=Wd[7],wyz=Wd[8];
                double px=Pd[0],py=Pd[1],pz=Pd[2],pxx=Pd[3],pyy=Pd[4],pzz=Pd[5],pxy=Pd[6],pxz=Pd[7],pyz=Pd[8];
		double Vyx=Vxy,Vzx=Vxz,Vzy=Vyz;
                double uyx=uxy,uzx=uxz,uzy=uyz;
                double vyx=vxy,vzx=vxz,vzy=vyz;
                double wyx=wxy,wzx=wxz,wzy=wyz;
                double pyx=pxy,pzx=pxz,pzy=pyz;

		double div=ux+vy+wz;
		double Vt=V*div-(u*Vx+v*Vy+w*Vz);
		double ut=-V*px-(u*ux+v*uy+w*uz);
                double vt=-V*py-(u*vx+v*vy+w*vz);
                double wt=-V*pz-(u*wx+v*wy+w*wz);
		double pt=-gamma*p*div-(u*px+v*py+w*pz);

                double Vtx=Vx*div+V*(uxx+vxy+wxz)-(ux*Vx+u*Vxx+vx*Vy+v*Vxy+wx*Vz+w*Vxz);
                double utx=-Vx*px-V*pxx-(ux*ux+u*uxx+vx*uy+v*uxy+wx*uz+w*uxz);
                double vtx=-Vx*py-V*pxy-(ux*vx+u*vxx+vx*vy+v*vxy+wx*vz+w*vxz);
                double wtx=-Vx*pz-V*pxz-(ux*wx+u*wxx+vx*wy+v*wxy+wx*wz+w*wxz);
                double ptx=-gamma*px*div-gamma*p*(uxx+vxy+wxz)-(ux*px+u*pxx+vx*py+v*pxy+wx*pz+w*pxz);

                double Vty=Vy*div+V*(uyx+vyy+wyz)-(uy*Vx+u*Vyx+vy*Vy+v*Vyy+wy*Vz+w*Vyz);
                double uty=-Vy*px-V*pyx-(uy*ux+u*uyx+vy*uy+v*uyy+wy*uz+w*uyz);
                double vty=-Vy*py-V*pyy-(uy*vx+u*vyx+vy*vy+v*vyy+wy*vz+w*vyz);
                double wty=-Vy*pz-V*pyz-(uy*wx+u*wyx+vy*wy+v*wyy+wy*wz+w*wyz);
                double pty=-gamma*py*div-gamma*p*(uyx+vyy+wyz)-(uy*px+u*pyx+vy*py+v*pyy+wy*pz+w*pyz);

                double Vtz=Vz*div+V*(uzx+vzy+wzz)-(uz*Vx+u*Vzx+vz*Vy+v*Vzy+wz*Vz+w*Vzz);
                double utz=-Vz*px-V*pzx-(uz*ux+u*uzx+vz*uy+v*uzy+wz*uz+w*uzz);
                double vtz=-Vz*py-V*pzy-(uz*vx+u*vzx+vz*vy+v*vzy+wz*vz+w*vzz);
                double wtz=-Vz*pz-V*pzz-(uz*wx+u*wzx+vz*wy+v*wzy+wz*wz+w*wzz);
                double ptz=-gamma*pz*div-gamma*p*(uzx+vzy+wzz)-(uz*px+u*pzx+vz*py+v*pzy+wz*pz+w*pzz);

		double Vtt=Vt*div+V*(utx+vty+wtz)-(ut*Vx+u*Vtx+vt*Vy+v*Vty+wt*Vz+w*Vtz);
		double utt=-Vt*px-V*ptx-(ut*ux+u*utx+vt*uy+v*uty+wt*uz+w*utz);
                double vtt=-Vt*py-V*pty-(ut*vx+u*vtx+vt*vy+v*vty+wt*vz+w*vtz);
                double wtt=-Vt*pz-V*ptz-(ut*wx+u*wtx+vt*wy+v*wty+wt*wz+w*wtz);
		double ptt=-gamma*pt*div-gamma*p*(utx+vty+wtz)-(ut*px+u*ptx+vt*py+v*pty+wt*pz+w*ptz);

                (*outVolume)=V+Dt*Vt+0.5*Dt*Dt*Vtt;
                (*outVelocityU)=u+Dt*ut+0.5*Dt*Dt*utt;
                (*outVelocityV)=v+Dt*vt+0.5*Dt*Dt*vtt+Dt*gravity;//TODO: MODIFIED GRAVITY DIRECTION
                (*outVelocityW)=w+Dt*wt+0.5*Dt*Dt*wtt;
                (*outPressure)=p+Dt*pt+0.5*Dt*Dt*ptt;
                if(std::isnan(*outVolume) || std::isinf(*outVolume) ||
                   std::isnan(*outPressure) || std::isinf(*outPressure) ||
                 std::isnan(*outVelocityU) || std::isinf(*outVelocityU) ||std::isnan(*outVelocityV) || std::isinf(*outVelocityV) ||std::isnan(*outVelocityW) || std::isinf(*outVelocityW)) {
                        assert(false);
                }

        }
        if(m_iDimension==2)
        {

                double Vx=Volumed[0],Vy=Volumed[1],Vz=0,Vxx=Volumed[2],Vyy=Volumed[3],Vzz=0,Vxy=Volumed[4],Vxz=0,Vyz=0;
                double ux=Ud[0],uy=Ud[1],uz=0,uxx=Ud[2],uyy=Ud[3],uzz=0,uxy=Ud[4],uxz=0,uyz=0;
                double vx=Vd[0],vy=Vd[1],vz=0,vxx=Vd[2],vyy=Vd[3],vzz=0,vxy=Vd[4],vxz=0,vyz=0;
                double wx=0,wy=0,wz=0,wxx=0,wyy=0,wzz=0,wxy=0,wxz=0,wyz=0;
                double px=Pd[0],py=Pd[1],pz=0,pxx=Pd[2],pyy=Pd[3],pzz=0,pxy=Pd[4],pxz=0,pyz=0;
                double Vyx=Vxy,Vzx=Vxz,Vzy=Vyz;
                double uyx=uxy,uzx=uxz,uzy=uyz;
                double vyx=vxy,vzx=vxz,vzy=vyz;
                double wyx=wxy,wzx=wxz,wzy=wyz;
                double pyx=pxy,pzx=pxz,pzy=pyz;

                double div=ux+vy+wz;
                double Vt=V*div-(u*Vx+v*Vy+w*Vz);
                double ut=-V*px-(u*ux+v*uy+w*uz);
                double vt=-V*py-(u*vx+v*vy+w*vz);
                double wt=-V*pz-(u*wx+v*wy+w*wz);
                double pt=-gamma*p*div-(u*px+v*py+w*pz);

                double Vtx=Vx*div+V*(uxx+vxy+wxz)-(ux*Vx+u*Vxx+vx*Vy+v*Vxy+wx*Vz+w*Vxz);
                double utx=-Vx*px-V*pxx-(ux*ux+u*uxx+vx*uy+v*uxy+wx*uz+w*uxz);
                double vtx=-Vx*py-V*pxy-(ux*vx+u*vxx+vx*vy+v*vxy+wx*vz+w*vxz);
                double wtx=-Vx*pz-V*pxz-(ux*wx+u*wxx+vx*wy+v*wxy+wx*wz+w*wxz);
                double ptx=-gamma*px*div-gamma*p*(uxx+vxy+wxz)-(ux*px+u*pxx+vx*py+v*pxy+wx*pz+w*pxz);

                double Vty=Vy*div+V*(uyx+vyy+wyz)-(uy*Vx+u*Vyx+vy*Vy+v*Vyy+wy*Vz+w*Vyz);
                double uty=-Vy*px-V*pyx-(uy*ux+u*uyx+vy*uy+v*uyy+wy*uz+w*uyz);
                double vty=-Vy*py-V*pyy-(uy*vx+u*vyx+vy*vy+v*vyy+wy*vz+w*vyz);
                double wty=-Vy*pz-V*pyz-(uy*wx+u*wyx+vy*wy+v*wyy+wy*wz+w*wyz);
                double pty=-gamma*py*div-gamma*p*(uyx+vyy+wyz)-(uy*px+u*pyx+vy*py+v*pyy+wy*pz+w*pyz);

                double Vtz=Vz*div+V*(uzx+vzy+wzz)-(uz*Vx+u*Vzx+vz*Vy+v*Vzy+wz*Vz+w*Vzz);
                double utz=-Vz*px-V*pzx-(uz*ux+u*uzx+vz*uy+v*uzy+wz*uz+w*uzz);
                double vtz=-Vz*py-V*pzy-(uz*vx+u*vzx+vz*vy+v*vzy+wz*vz+w*vzz);
                double wtz=-Vz*pz-V*pzz-(uz*wx+u*wzx+vz*wy+v*wzy+wz*wz+w*wzz);
                double ptz=-gamma*pz*div-gamma*p*(uzx+vzy+wzz)-(uz*px+u*pzx+vz*py+v*pzy+wz*pz+w*pzz);

                double Vtt=Vt*div+V*(utx+vty+wtz)-(ut*Vx+u*Vtx+vt*Vy+v*Vty+wt*Vz+w*Vtz);
                double utt=-Vt*px-V*ptx-(ut*ux+u*utx+vt*uy+v*uty+wt*uz+w*utz);
                double vtt=-Vt*py-V*pty-(ut*vx+u*vtx+vt*vy+v*vty+wt*vz+w*vtz);
                double wtt=-Vt*pz-V*ptz-(ut*wx+u*wtx+vt*wy+v*wty+wt*wz+w*wtz);
                double ptt=-gamma*pt*div-gamma*p*(utx+vty+wtz)-(ut*px+u*ptx+vt*py+v*pty+wt*pz+w*ptz);

                (*outVolume)=V+Dt*Vt+0.5*Dt*Dt*Vtt;
                (*outVelocityU)=u+Dt*ut+0.5*Dt*Dt*utt;
                (*outVelocityV)=v+Dt*vt+0.5*Dt*Dt*vtt+Dt*gravity;//TODO: MODIFIED GRAVITY DIRECTION
//                (*outVelocityW)=w+Dt*wt+0.5*Dt*Dt*wtt;
                (*outPressure)=p+Dt*pt+0.5*Dt*Dt*ptt;

                if(std::isnan(*outVolume) || std::isinf(*outVolume) ||
                   std::isnan(*outPressure) || std::isinf(*outPressure) ||
                 std::isnan(*outVelocityU) || std::isinf(*outVelocityU) ||std::isnan(*outVelocityV) || std::isinf(*outVelocityV)) {
                        assert(false);
                }
//                Pd[3]=gamma*(inPressure+Pinf)*(inVolume*(Pd[2]+Pd[3]));
//                Pd[4]=gamma*gamma*(inPressure+Pinf)*div*div+gamma*(inPressure+Pinf)*(Volumed[0]*Pd[0]+Volumed[1]*Pd[1]+cross);
//                Pd[2]=-gamma*(inPressure+Pinf)*div;

        }
}

void HyperbolicLPSolver::setMirrorPressureAndVelocity(int phase) {

	if((m_iDimension==2 && phase==2) || (m_iDimension==3 && phase==4)) return;

	size_t fluidStartIndex = m_pParticleData->m_iFluidStartIndex;
	size_t fluidEndIndex = m_pParticleData->m_iFluidStartIndex + m_pParticleData->m_iFluidNum;
	size_t boundaryStartIndex = m_pParticleData->m_iBoundaryStartIndex + m_pParticleData->m_iInflowNum;
	size_t boundaryEndIndex = m_pParticleData->m_iBoundaryStartIndex + m_pParticleData->m_iBoundaryNum;
	

	if(phase==0)
	{
                        for(size_t index=fluidEndIndex; index<boundaryStartIndex; index++)//inflow
                        {
                                m_pParticleData->m_vTemp1Volume[index]=m_pParticleData->m_vVolume[index];
                                m_pParticleData->m_vTemp1Pressure[index]=m_pParticleData->m_vPressure[index];
                                m_pParticleData->m_vTemp1VelocityU[index]=m_pParticleData->m_vVelocityU[index];
                                m_pParticleData->m_vTemp1VelocityV[index]=m_pParticleData->m_vVelocityV[index];
                                if(m_iDimension==3) m_pParticleData->m_vTemp1VelocityW[index]=m_pParticleData->m_vVelocityW[index];
				m_pParticleData->m_vTemp1SoundSpeed[index]=m_pParticleData->m_vSoundSpeed[index];
                                m_pParticleData->m_vTemp2Volume[index]=m_pParticleData->m_vVolume[index];
                                m_pParticleData->m_vTemp2Pressure[index]=m_pParticleData->m_vPressure[index];
                                m_pParticleData->m_vTemp2VelocityU[index]=m_pParticleData->m_vVelocityU[index];
                                m_pParticleData->m_vTemp2VelocityV[index]=m_pParticleData->m_vVelocityV[index];
                                if(m_iDimension==3)m_pParticleData->m_vTemp2VelocityW[index]=m_pParticleData->m_vVelocityW[index];
                                m_pParticleData->m_vTemp2SoundSpeed[index]=m_pParticleData->m_vSoundSpeed[index];

                        }

	}
	if(m_iDimension==3) {
		// pressure and velocity
		if(phase==0 || phase==2) {
			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for(size_t index=boundaryStartIndex; index<boundaryEndIndex; index++) {	
				size_t fIndex = m_vMirrorIndex[index-boundaryStartIndex];
				assert(fIndex>=fluidStartIndex && fIndex<boundaryStartIndex);
				m_pParticleData->m_vTemp1Pressure[index] = m_pParticleData->m_vTemp1Pressure[fIndex];
//                                m_pParticleData->m_vTemp1Pressure[index] = m_pParticleData->m_vPressure[index];
                                m_pParticleData->m_vTemp1VelocityU[index] = m_pParticleData->m_vVelocityU[index];
                                m_pParticleData->m_vTemp1VelocityV[index] = m_pParticleData->m_vVelocityV[index];
                                m_pParticleData->m_vTemp1VelocityW[index] = m_pParticleData->m_vVelocityW[index];
//				if(phase==2) {
//					m_pParticleData->m_vTemp1VelocityU[index] = m_pParticleData->m_vTemp1VelocityU[fIndex];
//					m_pParticleData->m_vTemp1VelocityV[index] = m_pParticleData->m_vTemp1VelocityV[fIndex];
//					m_pParticleData->m_vTemp1VelocityW[index] = m_pParticleData->m_vTemp1VelocityW[fIndex];	
//				}
			}
		}
		else if(phase==1 || phase==3) {
			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for(size_t index=boundaryStartIndex; index<boundaryEndIndex; index++) {	
				size_t fIndex = m_vMirrorIndex[index-boundaryStartIndex];
				assert(fIndex>=fluidStartIndex && fIndex<boundaryStartIndex);	
//                                m_pParticleData->m_vTemp2Pressure[index] = m_pParticleData->m_vPressure[index];
				m_pParticleData->m_vTemp2Pressure[index] = m_pParticleData->m_vTemp2Pressure[fIndex];
                                m_pParticleData->m_vTemp2VelocityU[index] = m_pParticleData->m_vVelocityU[index];
                                m_pParticleData->m_vTemp2VelocityV[index] = m_pParticleData->m_vVelocityV[index];
                                m_pParticleData->m_vTemp2VelocityW[index] = m_pParticleData->m_vVelocityW[index];
//				if(phase==3) {
//					m_pParticleData->m_vTemp2VelocityU[index] = m_pParticleData->m_vTemp2VelocityU[fIndex];
//					m_pParticleData->m_vTemp2VelocityV[index] = m_pParticleData->m_vTemp2VelocityV[fIndex];
//					m_pParticleData->m_vTemp2VelocityW[index] = m_pParticleData->m_vTemp2VelocityW[fIndex];	
//				}	
			}	
		}
 
	}
	else if(m_iDimension==2) {	
		// pressure and velocity
		if(phase==0) {
			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for(size_t index=boundaryStartIndex; index<boundaryEndIndex; index++) {	
				size_t fIndex = m_vMirrorIndex[index-boundaryStartIndex];
				assert(fIndex>=fluidStartIndex && fIndex<boundaryStartIndex);
				m_pParticleData->m_vTemp1Pressure[index] = m_pParticleData->m_vTemp1Pressure[fIndex];
                                m_pParticleData->m_vTemp1VelocityU[index] = m_pParticleData->m_vVelocityU[index];
                                m_pParticleData->m_vTemp1VelocityV[index] = m_pParticleData->m_vVelocityV[index];
			}
		}
		else if(phase==1) {
			#ifdef _OPENMP
			#pragma omp parallel for
			#endif
			for(size_t index=boundaryStartIndex; index<boundaryEndIndex; index++) {	
				size_t fIndex = m_vMirrorIndex[index-boundaryStartIndex];
				assert(fIndex>=fluidStartIndex && fIndex<boundaryStartIndex);	
				m_pParticleData->m_vTemp2Pressure[index] = m_pParticleData->m_vTemp2Pressure[fIndex];
                                m_pParticleData->m_vTemp2VelocityU[index] = m_pParticleData->m_vVelocityU[index];
                                m_pParticleData->m_vTemp2VelocityV[index] = m_pParticleData->m_vVelocityV[index];
//				m_pParticleData->m_vTemp2VelocityU[index] = m_pParticleData->m_vTemp2VelocityU[fIndex];
//				m_pParticleData->m_vTemp2VelocityV[index] = m_pParticleData->m_vTemp2VelocityV[fIndex];
			}	
		
		}
		
	}


}


void HyperbolicLPSolver::setGhostVelocity(int phase) {
		
//	if((m_iDimension==2 && (phase==0 || phase==2)) || 
//	   (m_iDimension==3 && (phase==0 || phase==1 || phase==4))) return;
	size_t fluidStartIndex = m_pParticleData->m_iFluidStartIndex;
	size_t fluidEndIndex = m_pParticleData->m_iFluidStartIndex + m_pParticleData->m_iFluidNum;
	size_t ghostStartIndex = m_pParticleData->getGhostStartIndex();
	
	if(m_iDimension==3) {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for(size_t index=fluidStartIndex; index<fluidEndIndex; index++) {
			if(m_vFillGhost[index]) {
				size_t neiListStartIndex = index*m_pParticleData->m_iMaxNeighbourNum;
				
				for(int i=0; i<m_pParticleData->m_vNeighbourListSize[index]; i++) {	
							
					size_t neiIndex = m_pParticleData->m_vNeighbourList[neiListStartIndex+i];

					if(neiIndex>=ghostStartIndex) {// ghost
                                                m_pParticleData->m_vTemp1Pressure[neiIndex] = 0;//Need to set pressure as well!
                                                m_pParticleData->m_vTemp2Pressure[neiIndex] = 0;
						m_pParticleData->m_vTemp1VelocityU[neiIndex] = m_pParticleData->m_vTemp1VelocityU[index];
						m_pParticleData->m_vTemp2VelocityU[neiIndex] = m_pParticleData->m_vTemp2VelocityU[index];
						m_pParticleData->m_vTemp1VelocityV[neiIndex] = m_pParticleData->m_vTemp1VelocityV[index];
						m_pParticleData->m_vTemp2VelocityV[neiIndex] = m_pParticleData->m_vTemp2VelocityV[index];
						m_pParticleData->m_vTemp1VelocityW[neiIndex] = m_pParticleData->m_vTemp1VelocityW[index];
						m_pParticleData->m_vTemp2VelocityW[neiIndex] = m_pParticleData->m_vTemp2VelocityW[index];
					}
				}
			}
		}
	}
	else if(m_iDimension==2) {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for(size_t index=fluidStartIndex; index<fluidEndIndex; index++) {
			if(m_vFillGhost[index]) {
				size_t neiListStartIndex = index*m_pParticleData->m_iMaxNeighbourNum;
				for(int i=0; i<m_pParticleData->m_vNeighbourListSize[index]; i++) {
						
					size_t neiIndex = m_pParticleData->m_vNeighbourList[neiListStartIndex+i];

					if(neiIndex>=ghostStartIndex) {// ghost
						m_pParticleData->m_vTemp1Pressure[neiIndex] = 0;//Need to set pressure as well!
                                                m_pParticleData->m_vTemp2Pressure[neiIndex] = 0;
						m_pParticleData->m_vTemp1VelocityU[neiIndex] = m_pParticleData->m_vTemp1VelocityU[index];
						m_pParticleData->m_vTemp2VelocityU[neiIndex] = m_pParticleData->m_vTemp2VelocityU[index];
						m_pParticleData->m_vTemp1VelocityV[neiIndex] = m_pParticleData->m_vTemp1VelocityV[index];
						m_pParticleData->m_vTemp2VelocityV[neiIndex] = m_pParticleData->m_vTemp2VelocityV[index];	
					}
				}
			}
		}
	}


}


void HyperbolicLPSolver::updateFluidState() {
		
	swap(m_pParticleData->m_vTemp1Volume, m_pParticleData->m_vVolume);
	swap(m_pParticleData->m_vTemp1Pressure, m_pParticleData->m_vPressure);
	swap(m_pParticleData->m_vTemp1SoundSpeed, m_pParticleData->m_vSoundSpeed);	
	
}


void HyperbolicLPSolver::moveFluidParticle() {
		
	size_t fluidStartIndex = m_pParticleData->getFluidStartIndex();
	size_t fluidEndIndex = fluidStartIndex + m_pParticleData->getFluidNum();
	
	if(m_iDimension==2) {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif
		for(size_t index=fluidStartIndex; index<fluidEndIndex; index++) {
			m_pParticleData->m_vPositionX[index] += 0.5 * m_fDt * 
			(1*m_pParticleData->m_vVelocityU[index] + 1*m_pParticleData->m_vTemp1VelocityU[index]); // 0.5 (old + new)	
			
			m_pParticleData->m_vPositionY[index] += 0.5 * m_fDt * 
			(1*m_pParticleData->m_vVelocityV[index] + 1*m_pParticleData->m_vTemp1VelocityV[index]);
		}	
	}
	else if(m_iDimension==3) {
		#ifdef _OPENMP
		#pragma omp parallel for
		#endif	
		for(size_t index=fluidStartIndex; index<fluidEndIndex; index++) {
			m_pParticleData->m_vPositionX[index] += 0.5 * m_fDt * 
			(m_pParticleData->m_vVelocityU[index] + m_pParticleData->m_vTemp1VelocityU[index]); // 0.5 (old + new)	
			
			m_pParticleData->m_vPositionY[index] += 0.5 * m_fDt * 
			(m_pParticleData->m_vVelocityV[index] + m_pParticleData->m_vTemp1VelocityV[index]);
			
			m_pParticleData->m_vPositionZ[index] += 0.5 * m_fDt * 
			(m_pParticleData->m_vVelocityW[index] + m_pParticleData->m_vTemp1VelocityW[index]);
		}
	}

}


void HyperbolicLPSolver::updateFluidVelocity() {
	
	swap(m_pParticleData->m_vTemp1VelocityU, m_pParticleData->m_vVelocityU);
	swap(m_pParticleData->m_vTemp1VelocityV, m_pParticleData->m_vVelocityV);	
	if(m_iDimension==3)	swap(m_pParticleData->m_vTemp1VelocityW, m_pParticleData->m_vVelocityW);

}

void HyperbolicLPSolver::setNeighbourListPointers_gpu(int dir, // input
	const int **neiList0, const int **neiList1, // output
	const int **neiListSize0, const int **neiListSize1) { 
	
	if(dir==0) { // x
		*neiList0 = m_pParticleData->d_m_vNeighbourListRight; 
		*neiList1 = m_pParticleData->d_m_vNeighbourListLeft;
		*neiListSize0 = m_pParticleData->d_m_vNeighbourListRightSize; 
		*neiListSize1 = m_pParticleData->d_m_vNeighbourListLeftSize;
	}
	else if(dir==1) { // y
		*neiList0 = m_pParticleData->d_m_vNeighbourListNorth; 
		*neiList1 = m_pParticleData->d_m_vNeighbourListSouth;
		*neiListSize0 = m_pParticleData->d_m_vNeighbourListNorthSize; 
		*neiListSize1 = m_pParticleData->d_m_vNeighbourListSouthSize;
	}
	else if(dir==2) { // z (if m_iDimension==2, dir != 2 for sure)
		*neiList0 = m_pParticleData->d_m_vNeighbourListUp; 
		*neiList1 = m_pParticleData->d_m_vNeighbourListDown;
		*neiListSize0 = m_pParticleData->d_m_vNeighbourListUpSize; 
		*neiListSize1 = m_pParticleData->d_m_vNeighbourListDownSize;	
	}
	else 
		assert(false);

}


void HyperbolicLPSolver::setInAndOutDataPointers_gpu(int phase, int dir,
	const double** inVelocity, const double** inPressure, const double** inVolume, const double** inSoundSpeed, 
	double** outVelocity, double** outPressure, double** outVolume, double** outSoundSpeed) {
	
	// assign pressure, volume, and sound_speed pointers
	if(phase==0) { // input: original output:temp1
		*inPressure   = m_pParticleData->d_m_vPressure;
		*inVolume     = m_pParticleData->d_m_vVolume;
		*inSoundSpeed = m_pParticleData->d_m_vSoundSpeed;
		
		*outPressure   = m_pParticleData->d_m_vTemp1Pressure;
		*outVolume     = m_pParticleData->d_m_vTemp1Volume;
		*outSoundSpeed = m_pParticleData->d_m_vTemp1SoundSpeed;	
	}
	else if(phase==1 || phase==3) { // input:temp1 output:temp2
		*inPressure   = m_pParticleData->d_m_vTemp1Pressure;
		*inVolume     = m_pParticleData->d_m_vTemp1Volume;
		*inSoundSpeed = m_pParticleData->d_m_vTemp1SoundSpeed;
		
		*outPressure   = m_pParticleData->d_m_vTemp2Pressure;
		*outVolume     = m_pParticleData->d_m_vTemp2Volume;
		*outSoundSpeed = m_pParticleData->d_m_vTemp2SoundSpeed;	
	}
	else if(phase==2 || phase==4){ // input:temp2 output: temp1
		*inPressure   = m_pParticleData->d_m_vTemp2Pressure;
		*inVolume     = m_pParticleData->d_m_vTemp2Volume;
		*inSoundSpeed = m_pParticleData->d_m_vTemp2SoundSpeed;
		
		*outPressure   = m_pParticleData->d_m_vTemp1Pressure;
		*outVolume	  = m_pParticleData->d_m_vTemp1Volume;
		*outSoundSpeed = m_pParticleData->d_m_vTemp1SoundSpeed;	
	}	
	else assert(false);
	
	// assign velocity pointers
	if(m_iDimension==2) {
		if(phase==0 || phase==1) { // input: original output:temp2
			if(dir==0) {
				*inVelocity = m_pParticleData->d_m_vVelocityU;
				*outVelocity = m_pParticleData->d_m_vTemp2VelocityU;
			}
			else if(dir==1) {
				*inVelocity = m_pParticleData->d_m_vVelocityV;
				*outVelocity = m_pParticleData->d_m_vTemp2VelocityV;	
			}	
		}	
		else if(phase==2){ // input:temp2 output: temp1
			if(dir==0) { // u v u
				*inVelocity = m_pParticleData->d_m_vTemp2VelocityU;
				*outVelocity = m_pParticleData->d_m_vTemp1VelocityU;
				// swap pointers so that temp1 will contain new info
				swap(m_pParticleData->d_m_vTemp1VelocityV, m_pParticleData->d_m_vTemp2VelocityV);	
			}
			else if(dir==1) { // v u v
				*inVelocity = m_pParticleData->d_m_vTemp2VelocityV;
				*outVelocity = m_pParticleData->d_m_vTemp1VelocityV;
				// swap pointers so that temp1 will contain new info
				swap(m_pParticleData->d_m_vTemp1VelocityU, m_pParticleData->d_m_vTemp2VelocityU);
			}		
		}	
		else assert(false);	
	}
	else if(m_iDimension==3) {
		if(phase==0 || phase==1 || phase==2) { // input: original output:temp1
			if(dir==0) {
				*inVelocity = m_pParticleData->d_m_vVelocityU;
				*outVelocity = m_pParticleData->d_m_vTemp1VelocityU;
			}
			else if(dir==1) {
				*inVelocity = m_pParticleData->d_m_vVelocityV;
				*outVelocity = m_pParticleData->d_m_vTemp1VelocityV;	
			}
			else if(dir==2) {
				*inVelocity = m_pParticleData->d_m_vVelocityW;
				*outVelocity = m_pParticleData->d_m_vTemp1VelocityW;	
			}
		}	
		else if(phase==3){ // input:temp1 output: temp2
			if(dir==0) { 
				*inVelocity = m_pParticleData->d_m_vTemp1VelocityU;
				*outVelocity = m_pParticleData->d_m_vTemp2VelocityU;
				// swap pointers so that temp2 will contain new info
				swap(m_pParticleData->d_m_vTemp1VelocityV, m_pParticleData->d_m_vTemp2VelocityV);
				swap(m_pParticleData->d_m_vTemp1VelocityW, m_pParticleData->d_m_vTemp2VelocityW);
			}
			else if(dir==1) { 
				*inVelocity = m_pParticleData->d_m_vTemp1VelocityV;
				*outVelocity = m_pParticleData->d_m_vTemp2VelocityV;
				// swap pointers so that temp2 will contain new info
				swap(m_pParticleData->d_m_vTemp1VelocityU, m_pParticleData->d_m_vTemp2VelocityU);
				swap(m_pParticleData->d_m_vTemp1VelocityW, m_pParticleData->d_m_vTemp2VelocityW);
			}
			else if(dir==2) { 
				*inVelocity = m_pParticleData->d_m_vTemp1VelocityW;
				*outVelocity = m_pParticleData->d_m_vTemp2VelocityW;
				// swap pointers so that temp2 will contain new info
				swap(m_pParticleData->d_m_vTemp1VelocityU, m_pParticleData->d_m_vTemp2VelocityU);
				swap(m_pParticleData->d_m_vTemp1VelocityV, m_pParticleData->d_m_vTemp2VelocityV);
			}
		}
		else if(phase==4){ // input:temp2 output: temp1
			if(dir==0) { 
				*inVelocity = m_pParticleData->d_m_vTemp2VelocityU;
				*outVelocity = m_pParticleData->d_m_vTemp1VelocityU;
				// swap pointers so that temp1 will contain new info
				swap(m_pParticleData->d_m_vTemp1VelocityV, m_pParticleData->d_m_vTemp2VelocityV);
				swap(m_pParticleData->d_m_vTemp1VelocityW, m_pParticleData->d_m_vTemp2VelocityW);
			}
			else if(dir==1) { 
				*inVelocity = m_pParticleData->d_m_vTemp2VelocityV;
				*outVelocity = m_pParticleData->d_m_vTemp1VelocityV;
				// swap pointers so that temp1 will contain new info
				swap(m_pParticleData->d_m_vTemp1VelocityU, m_pParticleData->d_m_vTemp2VelocityU);
				swap(m_pParticleData->d_m_vTemp1VelocityW, m_pParticleData->d_m_vTemp2VelocityW);
			}
			else if(dir==2) { 
				*inVelocity = m_pParticleData->d_m_vTemp2VelocityW;
				*outVelocity = m_pParticleData->d_m_vTemp1VelocityW;
				// swap pointers so that temp1 will contain new info
				swap(m_pParticleData->d_m_vTemp1VelocityU, m_pParticleData->d_m_vTemp2VelocityU);
				swap(m_pParticleData->d_m_vTemp1VelocityV, m_pParticleData->d_m_vTemp2VelocityV);
			}
		}
		else assert(false);	
	}

}


void HyperbolicLPSolver::setLPFOrderPointers_gpu(int dir, // input
	int** LPFOrder0, int** LPFOrder1, vector<int*>& LPFOrderOther) { // output
	
	if(dir==0) { // x
		*LPFOrder0 = m_pParticleData->d_m_vLPFOrderRight; // this direction
		*LPFOrder1 = m_pParticleData->d_m_vLPFOrderLeft;
		
		LPFOrderOther.push_back(m_pParticleData->d_m_vLPFOrderNorth); // other directions
		LPFOrderOther.push_back(m_pParticleData->d_m_vLPFOrderSouth);
		if(m_iDimension==3) {
			LPFOrderOther.push_back(m_pParticleData->d_m_vLPFOrderUp); 
			LPFOrderOther.push_back(m_pParticleData->d_m_vLPFOrderDown);
		}
	}
	else if(dir==1) { // y
		*LPFOrder0 = m_pParticleData->d_m_vLPFOrderNorth; // this direction
		*LPFOrder1 = m_pParticleData->d_m_vLPFOrderSouth;
		
		LPFOrderOther.push_back(m_pParticleData->d_m_vLPFOrderRight); // other directions
		LPFOrderOther.push_back(m_pParticleData->d_m_vLPFOrderLeft);
		if(m_iDimension==3) {
			LPFOrderOther.push_back(m_pParticleData->d_m_vLPFOrderUp); 
			LPFOrderOther.push_back(m_pParticleData->d_m_vLPFOrderDown);
		}
	}
	else if(dir==2) { // z
		*LPFOrder0 = m_pParticleData->d_m_vLPFOrderUp; // this direction
		*LPFOrder1 = m_pParticleData->d_m_vLPFOrderDown;
		
		LPFOrderOther.push_back(m_pParticleData->d_m_vLPFOrderRight); // other directions
		LPFOrderOther.push_back(m_pParticleData->d_m_vLPFOrderLeft);
		LPFOrderOther.push_back(m_pParticleData->d_m_vLPFOrderNorth); 
		LPFOrderOther.push_back(m_pParticleData->d_m_vLPFOrderSouth);
	}
	else
		assert(false);

}

void HyperbolicLPSolver::computeA2D_cpu(const int *neighbourList,const int*  LPFOrder,  int numRow,   int startIndex, int numComputingParticle) 
{  	
    int maxNeighbourInOne = m_pParticleData->m_iMaxNeighbourNumInOneDir;
    const double*x = m_pParticleData->d_m_vPositionX;
    const double*y = m_pParticleData->d_m_vPositionY;
    dim3 blocks(128,1);
    dim3 threads(128,1);
    computeA2D_gpu<<<blocks,threads>>>(neighbourList, LPFOrder, numRow, x, y, startIndex, numComputingParticle, maxNeighbourInOne,
       d_A_LS,d_distance); 

          
}

void HyperbolicLPSolver::computeA3D_cpu( const int *neighbourList, 
								    const int* LPFOrder, int numRow, int startIndex, int numComputingPrticle)
{
	size_t maxNeiNum = m_pParticleData->m_iMaxNeighbourNumInOneDir;
    const double* x = m_pParticleData->d_m_vPositionX;
    const double* y = m_pParticleData->d_m_vPositionY;
    const double* z = m_pParticleData->d_m_vPositionZ;
    
    dim3 blocks(128,1);
    dim3 threads(128,1);

    computeA3D_gpu<<<blocks,threads>>>(neighbourList, LPFOrder,
        numRow, x, y, z, startIndex, numComputingPrticle, maxNeiNum, d_A_LS, d_distance);


}


void HyperbolicLPSolver::computeSpatialDer_gpu(int dir, int offset,  void (HyperbolicLPSolver::*computeA) ( const int *,
const int*, int, int, int),
	const double* inPressure, const double* inVelocity,
	const int *neighbourList, const int *neighbourListSize,int additional,
	int* LPFOrder, double* vel_d, double* vel_dd, double* p_d, double* p_dd){
    int numRow;
    int numCol;
    int numFluid = m_pParticleData->m_iFluidNum;
    int numComputing = capacity;
    int warningCount[1];  //warning count copy from device
    
    int startIndex = 0;
    
    
    int numNeighbourInOneDir = m_pParticleData->m_iMaxNeighbourNumInOneDir;
    if(m_iLPFOrder == 1){ 
        
        numRow = m_iNumRow1stOrder;
        numCol = m_iNumCol1stOrder;
}
    else if(m_iLPFOrder == 2){
        numRow = m_iNumRow2ndOrder + additional;
        numCol = m_iNumCol2ndOrder;
   }
    dim3 blocks(96,1);
    dim3 threads(64,1);
   
 
    int valueInfo[numFluid];
   cudaMemset(d_valueAssigned, 0,sizeof(int)*numFluid );
   cudaMemset(d_warningCount,0,sizeof(int));
 /*  cudaMemcpy(valueInfo, d_valueAssigned, sizeof(int)*numFluid, cudaMemcpyDeviceToHost);
   for(int i=0; i<numFluid; i++ ){
        cout<<valueInfo[0]<<endl;
       }
       */
  checkLPFOrder_gpu<<<blocks,threads>>>(neighbourListSize ,LPFOrder, vel_d, vel_dd, p_d, p_dd, d_valueAssigned,d_warningCount,numFluid, numRow); 
   
   while(true){
        

       if(numComputing > numFluid-startIndex)
           numComputing = numFluid-startIndex;

        

      (this->*computeA)(neighbourList, LPFOrder, numRow, startIndex, numComputing);
/*
      for(int i=0;i<numComputing;i++){
           cout<<"A of number: "<<i<<endl;
            magma_dprint_gpu(6,1,A_temp[i],6,queue_qr);
       }
*/


        computeB_gpu<<<blocks,threads>>>(neighbourList, numRow, inPressure, startIndex, numComputing, numNeighbourInOneDir,d_B_LS);//output
/*       for(int i=0;i<numComputing;i++){
           cout<<"pre B of number: "<<i<<endl;
            magma_dprint_gpu(3,1,B_temp[i],3,queue_qr);
       }
   
  */    int info_gpu[numComputing]; 
    
        cublasHandle_t handle;
        cublasCreate(&handle);
        
  //  magma_dgeqrf_batched(numRow ,numCol ,A ,numRow ,Tau ,info , numFluid, queue_qr);
        cublasDgeqrfBatched(handle, numRow, numCol, d_A_LS, numRow, d_Tau, info_gpu, numComputing); 
      
     //   magma_dgeqrf_batched(numRow, numCol, d_A_LS, numRow, d_Tau, d_info, numComputing,queue_qr);
        computeLS_gpu<<<blocks,threads>>>(d_A_LS, d_B_LS, d_Tau, numRow, numCol, numComputing, d_result);
  /*  for(int i=0;i<numComputing;i++){
          cout<<"result pre  number: "<<i+startIndex<<endl;
              magma_dprint_gpu(2,1,result_temp[i],2,queue_qr);
    }   
*/ 
        assignValue_gpu<<<blocks,threads>>>(LPFOrder,d_valueAssigned, d_result, d_distance, numComputing, startIndex, dir,
        offset, p_d, p_dd);


        computeB_gpu<<<blocks,threads>>>(neighbourList,numRow, inVelocity, startIndex, numComputing, numNeighbourInOneDir,
        d_B_LS);
   /*   for(int i=0;i<numComputing;i++){
               cout<<"vel B of number: "<<i<<endl;
                magma_dprint_gpu(3,1,B_temp[i],3,queue_qr);
       }
   */

        computeLS_gpu<<<blocks,threads>>>(d_A_LS, d_B_LS, d_Tau, numRow, numCol, numComputing, d_result);
   

   /* for(int i=0;i<numComputing;i++){
           cout<<"result number: "<<i+startIndex<<endl;
              magma_dprint_gpu(2,1,result_temp[i],2,queue_qr);
    }   
*/
      
        assignValue_gpu<<<blocks,threads>>>(LPFOrder, d_valueAssigned, d_result, d_distance, numComputing, startIndex, dir,
        offset, vel_d, vel_dd);
     

        checkInvalid_gpu<<<blocks,threads>>>(d_valueAssigned, d_info, p_d, p_dd, vel_d, vel_dd, startIndex,
        numComputing, d_warningCount);
 
        startIndex += numComputing;

       if(startIndex == numFluid){
           cudaMemcpy(warningCount,d_warningCount,sizeof(int),cudaMemcpyDeviceToHost);
/*              cudaMemcpy(valueInfo, d_valueAssigned, sizeof(int)*numFluid, cudaMemcpyDeviceToHost);
cout<<"assignvalue"<<endl;
        for(int i=0; i<numFluid; i++ ){
            cout<<valueInfo[i]<<endl;
             }
  */ 
           if(warningCount[0] == numFluid){
                break;}
           else{
                printf("There are %d particles not been assigned value!\n", numFluid-warningCount[0]);
                startIndex = 0;
                numComputing = capacity;
                numRow += 1;
                checkLPFOrder_gpu<<<blocks,threads>>>(neighbourListSize ,LPFOrder, vel_d, vel_dd, p_d, p_dd, d_valueAssigned,d_warningCount,numFluid, numRow); 
                printf("The number of rows has been increased to %d.\n", numRow);
            }
        }
    }
   
       
    /*   cout<<"----------------------p_d---------------------------"<<endl;
        magma_dprint_gpu(numFluid,1,p_d,numFluid,queue_qr);
        cout<<"----------------------p_dd--------------------------"<<endl;
        magma_dprint_gpu(numFluid,1,p_dd,numFluid,queue_qr);
       cout<<"----------------------vel_d-------------------------"<<endl;
         magma_dprint_gpu(numFluid,1,vel_d,numFluid,queue_qr);
 
        cout<<"----------------------vel_dd-------------------------"<<endl;
      magma_dprint_gpu(numFluid,1,vel_dd,numFluid,queue_qr);
*/


}


void HyperbolicLPSolver::setDirOfPressureAndVelocityPointer_gpu(double**p_d_0,
double**p_dd_0,double**p_d_1,double**p_dd_1,double**vel_d_0,double**vel_dd_0,
double**vel_d_1,double**vel_dd_1){
    *p_d_0 = m_pParticleData->d_p_d_0;
    *p_dd_0 = m_pParticleData->d_p_dd_0;
    *vel_d_0 = m_pParticleData->d_vel_d_0;
    *vel_dd_0 = m_pParticleData->d_vel_dd_0;

    *p_d_1 = m_pParticleData->d_p_d_1;
    *p_dd_1 = m_pParticleData->d_p_dd_1;
    *vel_d_1 = m_pParticleData->d_vel_d_1;
    *vel_dd_1 = m_pParticleData->d_vel_dd_1;
    
    }




////////////////////////////////////////////////////////////////////////////////////////
// End of HyperbolicLPSolver
////////////////////////////////////////////////////////////////////////////////////////
