#include <iostream>
#include "matrix_build.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>




__global__ void computeA2D_gpu(const int*neighbourList,const int*LPFOrder, int numRow,const double*x,const
double*y,int startIndex, int numComputing, int maxNeighbourOneDir,
        double**A,double*dis)//output
    {
        
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
    //printf("runing form %d\n",tid);
    while(tid<numComputing){
        int index = startIndex+tid;
        int numOfRow = numRow;
        double distance = sqrt((x[neighbourList[index*maxNeighbourOneDir+numOfRow/4]]-x[index])*(x[neighbourList[index*maxNeighbourOneDir+numOfRow/4]]-x[index])+(y[neighbourList[index*maxNeighbourOneDir+numOfRow/4]]-y[index])*(y[neighbourList[index*maxNeighbourOneDir+numOfRow/4]]-y[index]));
       if(LPFOrder[index] == 1){
            
            for(int i=0;i<numOfRow;i++){
            
                int neiIndex = neighbourList[index*maxNeighbourOneDir+i];
                    
                double h = (x[neiIndex]-x[index])/distance;
                double k = (y[neiIndex]-y[index])/distance;
                A[tid][i] = h;//Notice it should be A[tid] because A are assigned by order
                A[tid][i+numOfRow] = k;
            }   
    
        }
        else if(LPFOrder[index] == 2){
            for(int i=0;i<numOfRow;i++){
                int neiIndex = neighbourList[index*maxNeighbourOneDir+i];
                double h = (x[neiIndex]-x[index])/distance;
                double k = (y[neiIndex]-y[index])/distance;
                A[tid][i] = h;
                A[tid][i + numOfRow] = k;
                A[tid][i + 2*numOfRow] = 0.5*h*h;
                A[tid][i + 3*numOfRow] = 0.5*k*k;
                A[tid][i + 4*numOfRow] = h*k;
            }
        
        } 
    dis[tid] = distance;
    tid = tid + offset;
    }
}

__global__ void computeA3D_gpu(const int*neighbourList,const int*LPFOrder, int numRow, const double*x, const double*y,
const double*z, int startIndex, int numComputing, int maxNeighbourOneDir,
        double**A,double*dis)//output
   {
        
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
    //printf("runing form %d\n",tid);
    while(tid < numComputing){
        int index = tid + startIndex;
        int numOfRow = numRow;
        double distance = sqrt((x[neighbourList[index*maxNeighbourOneDir+numOfRow/4]]-x[index])*(x[neighbourList[index*maxNeighbourOneDir+numOfRow/4]]-x[index])+(y[neighbourList[index*maxNeighbourOneDir+numOfRow/4]]-y[index])*(y[neighbourList[index*maxNeighbourOneDir+numOfRow/4]]-y[index])+(z[neighbourList[index*maxNeighbourOneDir+numOfRow/4]]-z[index])*(z[neighbourList[index*maxNeighbourOneDir+numOfRow/4]]-z[index]));
        if(LPFOrder[index] == 1){
            
            for(int i=0;i<numOfRow;i++){
            
                int neiIndex = neighbourList[index*maxNeighbourOneDir+i];
                    
                double h = (x[neiIndex]-x[index])/distance;
                
                double k = (y[neiIndex]-y[index])/distance;
                
                double l = (z[neiIndex] - z[index])/distance;
                A[tid][i] = h;
                A[tid][i + numOfRow] = k;
                A[tid][i + 2*numOfRow] = l;
            }   

    
        }
        else if(LPFOrder[index] == 2){
            for(int i=0;i<numOfRow;i++){
                int neiIndex = neighbourList[index*maxNeighbourOneDir+i];
                double h = (x[neiIndex]-x[index])/distance;
                double k = (y[neiIndex]-y[index])/distance;
                double l = (z[neiIndex] - z[index])/distance;
              
                A[tid][i] = h;
                A[tid][i + numOfRow] = k;
                A[tid][i + 2*numOfRow] = l;
                A[tid][i + 3*numOfRow] = 0.5*h*h;
                A[tid][i + 4*numOfRow] = 0.5*k*k;
                A[tid][i + 5*numOfRow] = 0.5*l*l;
                A[tid][i + 6*numOfRow] = h*k;
                A[tid][i + 7*numOfRow] = h*l;
                A[tid][i + 8*numOfRow] = k*l;

            }
        
        } 
    dis[tid] = distance;
    tid = tid + offset;
    }
}


__global__ void computeB_gpu(const int* neighbourList, int numRow, const double* inData, int startIndex, int numComputing, const int maxNumNeighbourOne,
                        double** b)//output vector b

{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
    while(tid<numComputing){
        int index = tid + startIndex;
        for(int i=0;i<numRow;i++){
            int neiIndex = neighbourList[index*maxNumNeighbourOne + i];
            b[tid][i] = inData[neiIndex] - inData[index];
        }
    
        tid = tid + offset;
    }
}


__global__ void computeLS_gpu(double**A,double**B,double**Tau, int numRow, int numCol, int numComputing, 
                        double**Result)//output result
{

    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
    while(tid < numComputing){
        int nrow = numRow;
        int ncol = numCol;
        for(int i=0;i<ncol;i++){
            double v_times_b = 0.;
            for(int j=0;j<nrow;j++){
                if(j < i) continue;
                if(j == i) v_times_b += 1*B[tid][j];
               
                else v_times_b += A[tid][j+i*nrow]*B[tid][j];
            }
            v_times_b *= Tau[tid][i];

            for(int j=0;j<nrow;j++){
                if(j < i) continue;
                if(j == i) B[tid][j] -= v_times_b;
                else
                B[tid][j] -= v_times_b*A[tid][j+i*nrow];
           }

        }


//compute QTB complete

//Backsubstitution
        for(int i=ncol-1;i>=0;i--){
          Result[tid][i] = B[tid][i]/(A[tid][i*nrow+i]);
               for(int j=0;j<i;j++){
                
                   B[tid][j] -= A[tid][j+i*nrow]*Result[tid][i];
           }
        
        }
    tid += offset;
    
    }
}

__global__ void assignValue_gpu(const int* LPFOrder, int* valueAssigned,  double** Result, const double* distance, int numComputing, int startIndex,
int dir, int offset, double* d1st, double* d2nd)//output
{         
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset_in = blockDim.x*gridDim.x;
    while(tid<numComputing){
        int index = tid + startIndex;
        if(LPFOrder[index] == 1 && valueAssigned[index] == 0 ){
                 d1st[index] = Result[tid][dir]/distance[tid];
                 d2nd[index] = 0;

        }
         else if(LPFOrder[index] == 2 && valueAssigned[index] == 0 ){
                 d1st[index] = Result[tid][dir]/distance[tid];
                 d2nd[index] = Result[tid][dir+offset]/distance[tid]/distance[tid];
             }
              
        tid += offset_in;
        
    }

}

    





__global__ void timeIntegration(
        double realDt, double multiplier1st, double multiplier2nd, int numFluid,
        double gravity, double* inVolume, double* inVelocity, double* inPressure, double* inSoundSpeed,
        double* vel_d_0, double* vel_dd_0, double* p_d_0, double* p_dd_0,
        double* vel_d_1, double* vel_dd_1, double* p_d_1, double* p_dd_1,
        double* outVolume, double* outVelocity, double* outPressure, double* info   //output
        )
{
    
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
    while(tid < numFluid){
        double K = inSoundSpeed[tid]*inSoundSpeed[tid]/inVolume[tid]/inVolume[tid];
        //this K only works for poly and spoly eos
        //Pt
        double Pt1st = -0.5*inVolume[tid]*K*(vel_d_0[tid]+vel_d_1[tid]) +
            0.5*inVolume[tid]*sqrt(K)*(p_d_0[tid]-p_d_1[tid]);
        double Pt2nd = -inVolume[tid]*inVolume[tid]*pow(K,1.5)*(vel_dd_0[tid]-vel_dd_1[tid]) + 
            inVolume[tid]*inVolume[tid]*K*(p_dd_0[tid]+p_dd_1[tid]);
        double Pt = multiplier1st*Pt1st + multiplier2nd*Pt2nd;

        //vt
        double Vt = -Pt/K;

        //VELt
        double VELt1st = 0.5*inVolume[tid]*sqrt(K)*(vel_d_0[tid]-vel_d_1[tid]) - 
            0.5*inVolume[tid]*(p_d_0[tid]-p_d_1[tid]);
        double VELt2nd = inVolume[tid]*inVolume[tid]*(vel_dd_0[tid]+vel_dd_1[tid]) - 
            inVolume[tid]*inVolume[tid]*sqrt(K)*(p_dd_0[tid]-p_dd_1[tid]);
        double VELt = multiplier1st*VELt1st + multiplier2nd*VELt2nd;


        outVolume[tid] = inVolume[tid] + realDt*Vt;
        outPressure[tid] = inPressure[tid] + realDt*Pt;
        outVelocity[tid] = inVelocity[tid] + realDt*(VELt+gravity);

        if( isnan(outVolume[tid]) || isinf(outVolume[tid]) ||
            isnan(outPressure[tid]) || isinf(outPressure[tid]) ||
            isnan(outVelocity[tid]) || isinf(outVelocity[tid])
          )
            *info = 1;
        
        tid += offset;
    }



}

__global__ void checkSoundspeedAndVolume(double* inSoundSpeed, double* outSoundSpeed, double* inVolume, double* outVolume, double* inPressure,
double* outPressure, double* inVelocity, double* outVelocity, int numFluid){
    
    
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
      
    while(tid < numFluid){
        
        int index = tid;
        if(inSoundSpeed[tid] == 0 || inVolume[tid] == 0){
        printf("The %d particle has 0 soundspeed or involume", tid); 
       	outVolume[index]   = inVolume[index];
		outPressure[index] = inPressure[index];
		outVelocity[index] = inVelocity[index];
		outSoundSpeed[index] = inSoundSpeed[index];
            
        }

    tid += offset;
 }
    
}

__global__ void checkLPFOrder_gpu(const int* neighboursize, int* LPFOrder, double* vel_d, double* vel_dd, double* p_d,
double* p_dd, int* valueAssigned, int* warningCount,  int numFluid,int numRow){
    
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
   
    while(tid < numFluid){
        int numNeisize =  neighboursize[tid];
        if(valueAssigned[tid] == 0 ){
            if(LPFOrder[tid]==2){
                if(numNeisize < numRow){
               
                      LPFOrder[tid] = 1;
                     }    
            } 
           if(LPFOrder[tid]==1){
               if(numNeisize < numRow){
               
                 LPFOrder[tid] = 0;

              }
         }
            if(LPFOrder[tid] == 0){
            vel_d[tid] = 0;
            vel_dd[tid] = 0;
            p_d[tid] = 0;
            p_dd[tid] = 0;

            valueAssigned[tid] = 1;
            atomicAdd(warningCount,1);
            printf("The particle of Index %d has 0 order!!!!! neighbourSize is %d\n",tid,numNeisize);
            }

        }
        tid += offset;
        
        }
    }

__global__ void checkInvalid_gpu(int* valueAssigned, int* info, double* p_d, double* p_dd,  double* vel_d, double*
vel_dd,int startIndex ,int numComputing, int* warningCount ){
    
    
        int tid = threadIdx.x + blockIdx.x*blockDim.x;
        int offset = blockDim.x*gridDim.x;

        while(tid < numComputing){
            
           int index = tid + startIndex;
           if((!(isnan(p_d[index]) || isnan(p_dd[index]) || isnan(vel_d[index]) || isnan(vel_dd[index]) ||
			  isinf(p_d[index]) || isinf(p_dd[index]) || isinf(vel_d[index]) || isinf(vel_dd[index]) ||
              info[tid] != 0)) && valueAssigned[index] == 0)
              {  
                valueAssigned[index] = 1;
                 
                atomicAdd(warningCount,1);

           }
   /*             
           else{
               if(valueAssigned[index] == 0){
               valueAssigned[index] = 1;
               atomicAdd(warningCount,1);
            }
                         }*/
            tid += offset; 

            }
 
}





