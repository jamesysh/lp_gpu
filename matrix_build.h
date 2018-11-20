#ifndef MATRIX_BUILD
#define MATRIX_BUILD
#include <cuda.h>
#include <cuda_runtime_api.h>
        


__global__ void computeA2D_gpu(const int*neighbourList,const int*LPFOrder, int numRow,const double*x,const
double*y,int startIndex, int numComputing, int maxNeighbourOneDir,
        double**A,double*dis);//output
  
 __global__ void computeA3D_gpu(const int*neighbourList,const int*LPFOrder, int numRow, const double*x, const double*y,
const double*z, int startIndex, int numComputing, int maxNeiNumInOneDir,
        double**A,double*dis);//output
   


__global__ void computeB_gpu(const int* neighbourList, int numRow, const double* inData, int startIndex, int numComputing, const int maxNumNeighbourOne,
                        double** b);//output vector b


__global__ void computeLS_gpu(double**A,double**B,double**Tau, int numRow, int numCol, int numComputing, 
                        double**Result);//output result


__global__ void assignValue_gpu(const int* LPFOrder, int* valueAssigned, double** Result, const double* distance, int numComputing, int startIndex,
int dir, int offset, double* d1st, double* d2nd);//output



__global__ void checkSoundspeedAndVolume(double* inSoundSpeed, double* outSoundSpeed, double* inVolume, double* outVolume, double* inPressure,
double* outPressure, double* inVelocity, double* outVelocity, int numFluid);

__global__ void checkLPFOrder_gpu(const int* neighboursize, int* LPFOrder, double* vel_d, double* vel_dd, double* p_d, double* p_dd, int* valueAssigned, int* warningCount,  int numFluid
,int numRow);
 

__global__ void checkInvalid_gpu(int* valueAssigned, int* info, double* p_d, double* p_dd, double* vel_d, double* vel_dd, int startIndex ,int numComputing, int* warningCount);
 
__global__ void initLPFOrder_upwind_gpu(int* LPFOrder0, int* LPFOrder1,  int numFluid);

__global__ void timeIntegration_gpu( 
        double realDt, double multiplier1st, double multiplier2nd, int numFluid,
        double gravity,const double* inVolume,const double* inVelocity,const double* inPressure,const double* inSoundSpeed,
        double* vel_d_0, double* vel_dd_0, double* p_d_0, double* p_dd_0,
        double* vel_d_1, double* vel_dd_1, double* p_d_1, double* p_dd_1,
        double* outVolume, double* outVelocity, double* outPressure, double* outSoundSpeed, int* info );

__global__ void updateOutPressureForPellet_gpu(const double* Deltaq, double* outPressure, double realDt, int m_pGamma, int numFluid, int* info);
__global__ void checkPressureAndDensity_gpu(double* outPressure, double* outVolume, double* outVelocity,
double* outSoundSpeed, const double* inPressure, const double* inVelocity, const double* inVolume, const double*
inSoundSpeed, int m_fInvalidPressure, int m_fInvalidDensity, int numFluid);
 __global__ void updateSoundSpeed_gpu(const double* outPressure,const double* outVolume, double* outSoundSpeed, double
m_fGamma, int numFluid, int* info);
 
#endif
