/*
 * Copyright 2010 PPI FUDAN University
 * CUDA-SURF v0.5
 * Author: Max Lv
 * Revision: 25
 */

// includes, common headers
#include "common.h"
#include "lfsr.h"
#include "zmq.hpp"
/*#include <omp.h>*/

// includes, kernels
#include <integral_kernel.cu>
#include <det_kernel.cu>
#include <surf_kernel.cu>

using namespace std;

uint m_det_size;

float sumtime = 0.f;
float tsum = 0.f;

//-------------------------------------------------------
//! Main functions list
int mainCUDAImage(int args, char** argv);

//-------------------------------------------------------
//! Entry function
int main(int args, char** argv)
{
    return mainCUDAImage(args, argv);
}

//-------------------------------------------------------
//! DetDes prepare for detdes_kernel
float4 *DetDes(IplImage *img, vector<float4> &ipts, float *orts)
{
    float *tmp;
    float4 *d_ipts;
    float *d_det;
    float *d_int;

    float time;

    // GPU Timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int height = img->height;
    int width = img->width;

    std::cout << width << " " << height << std::endl;

    cudaEventRecord(start, 0);

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(i_width, &width, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(i_height, &height, sizeof(int)));

    m_det_size = OCTAVES*INTERVALS*width*height;

    CUDA_SAFE_CALL(cudaMalloc((void **) &d_det, m_det_size * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_int, width * height * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &tmp, width * height * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_ipts, IPTSNUM * sizeof(float4)));
    zeroDetKernel<<< (m_det_size + 255) / 256, 256 >>>(d_det, m_det_size);

    //-------------------------------------------------------
    //! Integral on CUDA

    Integral(img, d_int, tmp);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sumtime += time;

    //-------------------------------------------------------
    //! BuildDet on CUDA

    cudaEventRecord(start, 0);

    buildDet( d_int, d_det, width, height );

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sumtime += time;

    //-------------------------------------------------------
    //! Get Interesting Points on CUDA

    float4 *h_ipts;
    int h_counter = 0;

    cudaEventRecord(start, 0);

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(counter, &h_counter, sizeof(int)));

    getIpoints( d_det, d_ipts, width, height );
    cutilCheckMsg("Kernel execution failed");

    CUDA_SAFE_CALL(cudaThreadSynchronize());

    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&h_counter, counter, sizeof(int)));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sumtime += time;

    int size = h_counter < IPTSNUM ? h_counter : IPTSNUM;

    if (!size)
    {
        //-------------------------------------------------------
        //! Free allocated memory

        // Device free
        CUDA_SAFE_CALL(cudaFree(d_det));
        CUDA_SAFE_CALL(cudaFree(d_ipts));
        CUDA_SAFE_CALL(cudaFree(d_int));

        // Host free
        return NULL;
    }

    cudaEventRecord(start, 0);

    //-------------------------------------------------------
    //! Describe interesting points on CUDA

    //Des results
    float4 *d_des;
    float4 *h_des;
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_des, 16 * size * sizeof(float4)));
    h_des = (float4 *)malloc(16 * size * sizeof(float4));
    h_ipts = (float4 *)malloc(size * sizeof(float4));

    //Orientation
    float *d_ort;
    float *h_ort;
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_ort, size * sizeof(float)));
    h_ort = (float *)malloc(size * sizeof(float));

    //Resource
    float3 *d_res;
    CUDA_SAFE_CALL(cudaMalloc((void **) &d_res, size * 121 * sizeof(float3)));

    //-------------------------------------------------------
    //! Run the Des kernel

    dim3 threads1(11, 11);
    getOrientationStep1<<< size, threads1 >>>( d_ipts, d_res, d_int );
    cutilCheckMsg("Kernel execution failed");

    getOrientationStep2<<< size, 42 >>>( d_ort, d_res );
    cutilCheckMsg("Kernel execution failed");

    dim3 threads2(4, 4);
    getDescriptor<<< size, threads2 >>>( d_ipts, d_des, d_ort, d_int );
    cutilCheckMsg("Kernel execution failed");

    CUDA_SAFE_CALL(cudaThreadSynchronize());
    //Copy back the result
    CUDA_SAFE_CALL(cudaMemcpy(h_des, d_des,
                16 * size * sizeof(float4), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_ort, d_ort,
                size * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaMemcpy(h_ipts, d_ipts, size * sizeof(float4), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    sumtime += time;


    for (int i = 0; i < size; i++)
    {
        ipts.push_back(h_ipts[i]);
    }
    //  cout << ipts.size() << endl;

    for (int i = 0; i < size; i++)
        orts[i] = h_ort[i];

    //-------------------------------------------------------
    //! Free allocated memory

    // Device free
    CUDA_SAFE_CALL(cudaFree(d_res));
    CUDA_SAFE_CALL(cudaFree(d_des));
    CUDA_SAFE_CALL(cudaFree(d_ort));
    CUDA_SAFE_CALL(cudaFree(d_det));
    CUDA_SAFE_CALL(cudaFree(d_ipts));
    CUDA_SAFE_CALL(cudaFree(d_int));

    // Host free
    free(h_ipts);
    free(h_ort);
    return h_des;
}


//-------------------------------------------------------
//! SURF on one image
int mainCUDAImage(int args, char** argv)
{

    if (args < 2) return -1;

    //  Prepare our context and socket
    /*char url[128];*/
    /*sprintf(url, "tcp://127.0.0.1:%s", argv[1]);*/
    /*zmq::context_t context (1);*/
    /*zmq::socket_t socket (context, ZMQ_REP);*/
    /*socket.bind (url);*/

    /*while (true)*/
    {
        /*zmq::message_t request;*/
        /*socket.recv (&request);*/
        /*string request_string((char*) request.data());*/
        /*char image_path[PATH_MAX];*/
        /*char result_path[PATH_MAX];*/
        /*sscanf(request_string.c_str(), "%s\n%s", image_path, result_path);*/
        char* image_path = argv[1];
        char* result_path = argv[2];

        vector<float4> ipts;
        struct timeval  bd_tick_x, bd_tick_e, bd_tick_d;
        float *orts;
        orts = (float *)malloc(IPTSNUM * sizeof(float));

        IplImage *img = cvLoadImage(image_path);

        // convert the image to single channel 32f
        IplImage *gray_img = getGray(img);

        gettimeofday(&bd_tick_x, 0);

        float4 *des = DetDes(gray_img, ipts, orts);

        gettimeofday(&bd_tick_e, 0);
        GET_TIME(bd_tick_x, bd_tick_e, bd_tick_d);
        std::cout<< "gpu took: " << sumtime << " ms" << std::endl;
        std::cout<< "surf took: " << bd_tick_d.tv_sec*1000 + bd_tick_d.tv_usec/1000 << " ms" << std::endl;

        gettimeofday(&bd_tick_x, 0);

        vector<struct feature>& features = lfsr(img, ipts);

        ofstream file;
        file.open(result_path);
        file << image_path << endl;
        file << 64 << " " << features.size() << endl;
        for (int i = 0; i < features.size(); i++) {
            file << features[i].x << " " << features[i].y 
                << " " << features[i].scl << " " << orts[i] << endl;
            for (int j = 0; j < 16; j++) {
                float4 d = des[features[i].index*16 + j];
                file << fRound(d.x * 10000) << " "
                    << fRound(d.y * 10000) << " "
                    << fRound(d.z * 10000) << " "
                    << fRound(d.w * 10000);
            }
            file << endl;
        }
        file.close();

        gettimeofday(&bd_tick_e, 0);
        GET_TIME(bd_tick_x, bd_tick_e, bd_tick_d);

        std::cout<< "lfsr took: " << bd_tick_d.tv_sec*1000 + bd_tick_d.tv_usec/1000 << " ms" << std::endl;

        // Deallocate the integral image

        // Draw the detected points
        /*drawIpoints(img, ipts, ipts.size(), orts);*/

        // Display the result
        /*showImage(img);*/

        cvReleaseImage(&img);
        cvReleaseImage(&gray_img);
        free(orts);
        free(des);
        delete (&features);

        /*zmq::message_t reply (3);*/
        /*memcpy ((void *) reply.data (), "OK", 3);*/
        /*socket.send (reply);*/
    }

    return 0;
}

