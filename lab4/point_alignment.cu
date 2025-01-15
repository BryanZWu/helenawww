/**
 * CUDA Point Alignment
 * George Stathopoulos, Jenny Lee, Mary Giambrone, 2019*/ 

#include <cstdio>
#include <stdio.h>
#include <fstream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "helper_cuda.h"
#include <string>
#include <fstream>

#include "obj_structures.h"

// helper_cuda.h contains the error checking macros. note that they're called
// CUDA_CALL, CUBLAS_CALL, and CUSOLVER_CALL instead of the previous names

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

    const char* cusolverGetErrorString(cusolverStatus_t status) {
    switch (status) {
        case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_MAPPING_ERROR: return "CUSOLVER_STATUS_MAPPING_ERROR";
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSOLVER_STATUS_NOT_SUPPORTED: return "CUSOLVER_STATUS_NOT_SUPPORTED";
        case CUSOLVER_STATUS_ZERO_PIVOT: return "CUSOLVER_STATUS_ZERO_PIVOT";
        case CUSOLVER_STATUS_INVALID_LICENSE: return "CUSOLVER_STATUS_INVALID_LICENSE";
        default: return "UNKNOWN CUSOLVER ERROR";
    }
}
int main(int argc, char *argv[]) {
    if (argc != 4)
    {
        printf("Usage: ./point_alignment [file1.obj] [file2.obj] [output.obj]\n");
        return 1;
    }

    std::string filename, filename2, output_filename;
    filename = argv[1];
    filename2 = argv[2];
    output_filename = argv[3];

    std::cout << "Aligning " << filename << " with " << filename2 <<  std::endl;
    Object obj1 = read_obj_file(filename);
    std::cout << "Reading " << filename << ", which has " << obj1.vertices.size() << " vertices" << std::endl;
    Object obj2 = read_obj_file(filename2);

    std::cout << "Reading " << filename2 << ", which has " << obj2.vertices.size() << " vertices" << std::endl;
    if (obj1.vertices.size() != obj2.vertices.size())
    {
        printf("Error: number of vertices in the obj files do not match.\n");
        return 1;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Loading in obj into vertex Array
    ///////////////////////////////////////////////////////////////////////////

    int point_dim = 4; // 3 spatial + 1 homogeneous
    int num_points = obj1.vertices.size();

    // in col-major
    float * x1mat = vertex_array_from_obj(obj1);
    float * x2mat = vertex_array_from_obj(obj2);

    ///////////////////////////////////////////////////////////////////////////
    // Point Alignment
    ///////////////////////////////////////////////////////////////////////////

    // TODO: Initialize cublas handle
    cublasHandle_t handle;
    cublasStatus_t status;
    status = cublasCreate(&handle);
    // if (status == ???):
    //     print("lol help")

    float * dev_x1mat;
    float * dev_x2mat;
    float * dev_xx4x4;
    float * dev_x1Tx2;

    // TODO: Allocate device memory and copy over the data onto the device
    // Hint: Use cublasSetMatrix() for copying
    printf("allocating memory\n");

    cudaMalloc(&dev_x1mat, num_points * 4 * sizeof(float));
    cudaMalloc(&dev_x2mat, num_points * 4 * sizeof(float));
    cudaMalloc(&dev_xx4x4, 4 * 4 * sizeof(float));
    cudaMalloc(&dev_x1Tx2, num_points * 4 * sizeof(float));

    // for (int i = 0; i < num_points; i++) {
    //     printf("init matrix %.3f %.3f %.3f %.3f\n", 
    //            x1mat[i], 
    //            x1mat[num_points + i], 
    //            x1mat[2 * num_points + i], 
    //            x1mat[3 * num_points + i]);
    // }
    status = cublasSetMatrix(num_points, 4, sizeof(float), x1mat, num_points, dev_x1mat, num_points);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        printf("cublasSetMatrix failed: %s\n", "HELP1");
    }
    status = cublasSetMatrix(num_points, 4, sizeof(float), x2mat, num_points, dev_x2mat, num_points);
     if (status != CUSOLVER_STATUS_SUCCESS) {
    printf("cublasSetMatrix failed: %s\n", "HELP1");
    }
    // Now, proceed with the computations necessary to solve for the linear
    // transformation.

    // ???
    float one = 1.0;
    float zero = 0.0;

    // TODO: First calculate xx4x4 and x1Tx2
    // Following two calls should correspond to:
    //   xx4x4 = Transpose[x1mat] . x1mat
    printf("starting to calculate xx4x4\n");
    status = cublasSgemm_v2(
        handle=handle, CUBLAS_OP_T, CUBLAS_OP_N, 4, 4, num_points, &one, dev_x1mat, num_points, dev_x1mat, num_points, &zero, dev_xx4x4, 4
    );
    if (status != CUSOLVER_STATUS_SUCCESS) {
    printf("cublasSgemm_v2 failed: %s\n", "HELP1");
    } else {
        printf("cublasSgemm_v2 succeeded.\n");
        // Print the values
        float *debug = (float *)malloc(4 * 4 * sizeof(float));
        cudaMemcpy(debug, dev_xx4x4, 4 * 4 * sizeof(float), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < 4; i++) {
        //     for (int j = 0; j < 4; j++) {
        //         printf("debug1 %f ", debug[i * 4 + j]);
        //     }
        // }
    }
    //   x1Tx2 = Transpose[x1mat] . x2mat
    status = cublasSgemm_v2(
        handle=handle, CUBLAS_OP_T, CUBLAS_OP_N, 4, 4, num_points, &one, dev_x1mat, num_points, dev_x2mat, num_points, &zero, dev_x1Tx2, 4
    );
        if (status != CUSOLVER_STATUS_SUCCESS) {
    printf("cublasSgemm_v2 failed: %s\n", "HELP2"); // cudaGetErrorString(status));
    } else {
        printf("cublasSgemm_v2 succeeded.\n");
    }


    // TODO: Finally, solve the system using LU-factorization! We're solving
    //         xx4x4 . m4x4mat.T = x1Tx2   i.e.   m4x4mat.T = Inverse[xx4x4] . x1Tx2
    //
    //       Factorize xx4x4 into an L and U matrix, ie.  xx4x4 = LU
    //
    //       Then, solve the following two systems at once using cusolver's getrs
    //           L . temp  =  P . x1Tx2
    //       And then then,
    //           U . m4x4mat = temp
    //
    //       Generally, pre-factoring a matrix is a very good strategy when
    //       it is needed for repeated solves.


    // TODO: Make handle for cuSolver
    cusolverStatus_t status_cusolver;
    cusolverDnHandle_t solver_handle;
    status_cusolver = cusolverDnCreate(&solver_handle);


    // TODO: Initialize work buffer using cusolverDnSgetrf_bufferSize
    float * work;
    int Lwork;



    printf("compute buffer size\n");
    // TODO: compute buffer size and prepare memory
    status_cusolver = cusolverDnSgetrf_bufferSize(solver_handle, 4, 4, dev_xx4x4, 4, &Lwork);

    float *workspace;
    cudaMalloc(&workspace, Lwork * sizeof(float));
    


    // TODO: Initialize memory for pivot array, with a size of point_dim
    int * pivots;
    cudaMalloc(&pivots, point_dim * sizeof(float));
    int *info;
    cudaMalloc(&info, sizeof(int));    


    // TODO: Now, call the factorizer cusolverDnSgetrf, using the above initialized data
    status_cusolver = cusolverDnSgetrf(solver_handle, 4, 4, dev_xx4x4, 4, workspace, pivots, info);
    if (status_cusolver != CUSOLVER_STATUS_SUCCESS) {
        printf("buffer HELPPPP");
    } else {
        printf("buffer does NOT need HELP");
    }

    // DEBUG: copy the factorized matrix to the output
    cudaMemcpy(dev_x1Tx2, dev_xx4x4, 4 * 4 * sizeof(float), cudaMemcpyDeviceToDevice);


    // TODO: Finally, solve the factorized version using a direct call to cusolverDnSgetrs. This gets written inplace to... B??? dev_x1Tx2?
    // status_cusolver = cusolverDnSgetrs(solver_handle, CUBLAS_OP_N, 4, 4, dev_xx4x4, 4, pivots, dev_x1Tx2, 4, info);
    // if (status_cusolver != CUSOLVER_STATUS_SUCCESS) {
    //     printf("solver HELPPPP");
    // } else {
    //     printf("solver does NOT need HELP");
    // }

    // TODO: Destroy the cuSolver handle
    status_cusolver = cusolverDnDestroy(solver_handle);

    // TODO: Copy final transformation back to host. Note that at this point
    // the transformation matrix is transposed
    float * out_transformation;
    out_transformation = (float *)malloc(16 * sizeof(float));
    cudaMemcpy(out_transformation, dev_x1Tx2, 16 * sizeof(float), cudaMemcpyDeviceToHost);



    // TODO Helena: Don't forget to set the bottom row of the final transformation
    //       to [0,0,0,1] (right-most columns of the transposed matrix)
    out_transformation[3] = 0;
    out_transformation[7] = 0;
    out_transformation[11] = 0;
    out_transformation[15] = 1;

    // Print transformation in row order.
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << out_transformation[i * point_dim + j] << " ";
        }
        std::cout << "\n";
    }

    ///////////////////////////////////////////////////////////////////////////
    // Transform point and print output object file
    ///////////////////////////////////////////////////////////////////////////

    // TODO Allocate and Initialize data matrix
    float * dev_pt;
    cudaMalloc(&dev_pt, num_points * sizeof(float) * point_dim);
    cudaMemcpy(dev_pt, out_transformation, point_dim * num_points * sizeof(float), cudaMemcpyHostToDevice);

    // TODO Allocate and Initialize transformation matrix
    float * dev_trans_mat;
    cudaMalloc(&dev_trans_mat, sizeof(float) * 16);
    cudaMemcpy(dev_trans_mat, out_transformation, 16 * sizeof(float), cudaMemcpyHostToDevice);


    // TODO Allocate and Initialize transformed points
    float * dev_trans_pt;
    cudaMalloc(&dev_trans_pt, num_points * sizeof(float) * point_dim);

    float one_d = 1;
    float zero_d = 0;

    // TODO Transform point matrix
    //          (4x4 trans_mat) . (nx4 pointzx matrix)^T = (4xn transformed points)
    printf("compute point transformation\n");
    status = cublasSgemm_v2(
        handle=handle, CUBLAS_OP_N, CUBLAS_OP_T, 4, num_points, 4,
        &one_d, dev_trans_mat, 4,
        dev_pt, num_points, &one_d,
        dev_trans_pt, num_points
    );
    

    // So now dev_trans_pt has shape (4 x n)
    float * trans_pt; 
    trans_pt = (float *)malloc(num_points * 4 * sizeof(float));
    cudaMemcpy(trans_pt, dev_trans_pt, num_points * 4 * sizeof(float), cudaMemcpyDeviceToHost);


    // get Object from transformed vertex matrix
    Object trans_obj = obj_from_vertex_array(trans_pt, num_points, point_dim, obj1);

    // print Object to output file
    std::ofstream obj_file (output_filename);
    print_obj_data(trans_obj, obj_file);

    // free CPU memory
    free(trans_pt);

    ///////////////////////////////////////////////////////////////////////////
    // Free Memory
    ///////////////////////////////////////////////////////////////////////////

    // TODO: Free GPU memory
    free(dev_pt);
    free(dev_trans_mat);
    free(dev_trans_mat);
    free(pivots);
    free(info);
    free(workspace);
    free(dev_x1mat);
    free(dev_x2mat);
    free(dev_xx4x4);
    free(dev_x1Tx2);


    // TODO: Free CPU memory
    free(out_transformation);
    free(x1mat);
    free(x2mat);


}

