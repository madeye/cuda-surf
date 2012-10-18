/*
 * Copyright 2010 PPI FUDAN University
 * CUDA-SURF v0.5
 * Author: Max Lv
 * Revision: 25
 */

// includes, common headers
#include "common.h"
#include <omp.h>

// includes, kernels
#include <integral_kernel.cu>
#include <det_kernel.cu>
#include <surf_kernel.cu>
#include <match_kernel.cu>

#define PROCEDURE 1

using namespace std;

uint m_det_size;

float sumtime = 0.f;
float tsum = 0.f;

float *d_det;
float *Src;
float *Tmp;
float4 *d_ipts;
bool init=false;

//-------------------------------------------------------
//! Main functions list
int mainCUDAImage(int args, char** argv);
int mainCUDAImages(void);
int mainCUDAMatch(void);

//-------------------------------------------------------
//! Entry function
int main(int args, char** argv)
{
  if (PROCEDURE == 1) return mainCUDAImage(args, argv);
  if (PROCEDURE == 2) return mainCUDAImages();
  if (PROCEDURE == 3) return mainCUDAMatch();
}

//-------------------------------------------------------
//! Match prepare for match_kernel
int Match(float4 *des1, float4 *des2, int size1, int size2)
{

  int2 *d_matches;
  int2 *h_matches;
  float4 *d_des1;
  float4 *d_des2;
  int h_matchnum = 0;
  int matches_size = size1 >= size2 ? size2 : size1;
  matchnumGold = h_matchnum;

  CUDA_SAFE_CALL(cudaMalloc((void **) &d_matches, matches_size * sizeof(int2)));
  CUDA_SAFE_CALL(cudaMalloc((void **) &d_des1, size1 * 16 * sizeof(float4)));
  CUDA_SAFE_CALL(cudaMalloc((void **) &d_des2, size2 * 16 * sizeof(float4)));

  CUDA_SAFE_CALL(cudaMemcpy(d_des1, des1,
                            size1 * 16 * sizeof(float4), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_des2, des2,
                            size2 * 16 * sizeof(float4), cudaMemcpyHostToDevice));

  h_matches = (int2 *)malloc(matches_size * sizeof(int2));

  CUDA_SAFE_CALL(cudaMemcpyToSymbol(matchnum, &h_matchnum, sizeof(int)));

  if (size1 >= size2)
    {
      CUDA_SAFE_CALL(cudaBindTexture(0, TexDes2, d_des2,
                                     size2 * 16 * sizeof(float4)));
      getMatchesGold(des1, des2, size1, size2, h_matches);
      /*getMatches<<< (size1 + TNUMM - 1) / TNUMM, TNUMM >>>*/
      /*( d_des1, size1, size2, d_matches);*/
    }
  else
    {
      CUDA_SAFE_CALL(cudaBindTexture(0, TexDes2, d_des1,
                                     size1 * 16 * sizeof(float4)));
      getMatchesGold(des2, des1, size2, size1, h_matches);
      /*getMatches<<< (size2 + TNUMM - 1) / TNUMM, TNUMM >>>*/
      /*( d_des2, size2, size1, d_matches);*/
    }

  cutilCheckMsg("Kernel execution failed");

  /*CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&h_matchnum, matchnum, sizeof(int)));*/

  /*CUDA_SAFE_CALL(cudaMemcpy(h_matches, d_matches,*/
                            /*h_matchnum * sizeof(int2), cudaMemcpyDeviceToHost));*/

  h_matchnum = matchnumGold;

  /*for(int i = 0; i < h_matchnum; i++)*/
    /*cout << h_matches[i].x << " " << h_matches[i].y << endl;*/

  CUDA_SAFE_CALL(cudaFree(d_des1));
  CUDA_SAFE_CALL(cudaFree(d_des2));
  CUDA_SAFE_CALL(cudaFree(d_matches));
  free(h_matches);

  return h_matchnum;

}

//-------------------------------------------------------
//! DetDes prepare for detdes_kernel
float4 *DetDes(IplImage *img, vector<float4> &ipts, float *orts)
{
  // GPU Timer
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int height = img->height;
  int width = img->width;

  CUDA_SAFE_CALL(cudaMemcpyToSymbol(i_width, &width, sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(i_height, &height, sizeof(int)));

  m_det_size = octaves*intervals*width*height;

  if (!init)
    {
      CUDA_SAFE_CALL(cudaMalloc((void **) &d_det, m_det_size * sizeof(float)));
      CUDA_SAFE_CALL(cudaMalloc((void **) &Src, width * height * sizeof(float)));
      CUDA_SAFE_CALL(cudaMalloc((void **) &Tmp, width * height * sizeof(float)));
      CUDA_SAFE_CALL(cudaMalloc((void **) &d_ipts, IPTSNUM * sizeof(float4)));
      zeroDetKernel<<< (m_det_size + 255) / 256, 256 >>>(d_det, m_det_size);
      init = true;
    }


  //-------------------------------------------------------
  //! Integral on CUDA


  cudaEventRecord(start, 0);

  Integral(img, Src, Tmp);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float intime;
  cudaEventElapsedTime(&intime, start, stop);
  tsum += intime;

  //-------------------------------------------------------
  //! BuildDet on CUDA

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();


  CUDA_SAFE_CALL(cudaBindTexture(0, TexInt,
                                 Src, width * height * sizeof(float)));

  cudaEventRecord(start, 0);

  buildDet( d_det, width, height );

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  float time;
  cudaEventElapsedTime(&time, start, stop);
  sumtime += time;

  //-------------------------------------------------------
  //! Get Interesting Points on CUDA

  float4 *h_ipts;
  int h_counter = 0;


  // Bind d_det to texture
  CUDA_SAFE_CALL(cudaBindTexture(0, TexDet,
                                 d_det, m_det_size * sizeof(float)));

  CUDA_SAFE_CALL(cudaMemcpyToSymbol(counter, &h_counter, sizeof(int)));

  cudaEventRecord(start, 0);

  getIpoints( d_ipts, width, height );
  cutilCheckMsg("Kernel execution failed");

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time, start, stop);
  sumtime += time;


  CUDA_SAFE_CALL(cudaThreadSynchronize());

  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&h_counter, counter, sizeof(int)));

  int size = h_counter < IPTSNUM ? h_counter : IPTSNUM;

  if (!size)
    {
      //-------------------------------------------------------
      //! Free allocated memory

      // Unbind texture
      CUDA_SAFE_CALL(cudaUnbindTexture(TexDet));
      CUDA_SAFE_CALL(cudaUnbindTexture(TexInt));

      // Device free
      //CUDA_SAFE_CALL(cudaFree(d_det));
      //CUDA_SAFE_CALL(cudaFree(d_ipts));
      //CUDA_SAFE_CALL(cudaFree(Src));
      //CUDA_SAFE_CALL(cudaFree(z_ipts));

      // Host free
      //free(h_det);
      return NULL;
    }

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

  cudaEventRecord(start, 0);

  dim3 threads1(11, 11);
  getOrientationStep1<<< size, threads1 >>>( d_ipts, d_res );
  cutilCheckMsg("Kernel execution failed");

  getOrientationStep2<<< size, 42 >>>( d_ort, d_res );
  cutilCheckMsg("Kernel execution failed");

  dim3 threads2(4, 4);
  getDescriptor<<< size, threads2 >>>( d_ipts, d_des, d_ort );
  cutilCheckMsg("Kernel execution failed");

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&time, start, stop);
  sumtime += time;

  CUDA_SAFE_CALL(cudaThreadSynchronize());
  //Copy back the result
  CUDA_SAFE_CALL(cudaMemcpy(h_des, d_des,
                            16 * size * sizeof(float4), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(h_ort, d_ort,
                            size * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_SAFE_CALL(cudaMemcpy(h_ipts, d_ipts, size * sizeof(float4), cudaMemcpyDeviceToHost));

  for (int i = 0; i < size; i++)
    {
      ipts.push_back(h_ipts[i]);
    }
  //  cout << ipts.size() << endl;

#if PROCEDURE != 2
  for (int i = 0; i < size; i++)
    orts[i] = h_ort[i];
#endif

  //-------------------------------------------------------
  //! Free allocated memory

  // Unbind texture
  CUDA_SAFE_CALL(cudaUnbindTexture(TexDet));
  CUDA_SAFE_CALL(cudaUnbindTexture(TexInt));

  // Device free
  CUDA_SAFE_CALL(cudaFree(d_res));
  CUDA_SAFE_CALL(cudaFree(d_des));
  CUDA_SAFE_CALL(cudaFree(d_ort));
//CUDA_SAFE_CALL(cudaFree(d_det));
//CUDA_SAFE_CALL(cudaFree(d_ipts));
//CUDA_SAFE_CALL(cudaFree(Src));
  //CUDA_SAFE_CALL(cudaFree(z_ipts));

  // Host free
  //free(h_det);
  free(h_ipts);
  free(h_ort);
#if PROCEDURE == 3
  return h_des;
#else
  free(h_des);
  return NULL;
#endif
}


//-------------------------------------------------------
//! SURF on one image
int mainCUDAImage(int args, char** argv)
{

  vector<float4> ipts;
  struct timeval  bd_tick_x, bd_tick_e, bd_tick_d;
  float *orts;
  orts = (float *)malloc(IPTSNUM * sizeof(float));

  if (args < 2) return -1;

  IplImage *img = cvLoadImage(argv[1]);

  // convert the image to single channel 32f
  IplImage *gray_img = getGray(img);

  gettimeofday(&bd_tick_x, 0);

  DetDes(gray_img, ipts, orts);

  gettimeofday(&bd_tick_e, 0);
  GET_TIME(bd_tick_x, bd_tick_e, bd_tick_d);

  for (int i = 0; i < ipts.size(); i++)
    cout << ipts[i].z << " " << orts[i] << " "<< fRound(ipts[i].w) << " " <<
         fRound(ipts[i].y) << " " << fRound(ipts[i].x) << endl;
  std::cout<< "OpenSURF took: " << bd_tick_d.tv_sec*1000 + bd_tick_d.tv_usec/1000 << " ms" << std::endl;
  cout << "Ipoint Num: " << ipts.size() << endl;

  /*
      Ipoint *ipt;
      for(unsigned int i = 0; i < ipts.size(); i++)
      {
      ipt = &ipts.at(i);
      std::cout << ipt->scale;
      std::cout << " " << ipt->orientation << " " << ipt->laplacian;
      std::cout << " " << fRound(ipt->y) << " " << fRound(ipt->x) << std::endl;
      }
   */

  // Deallocate the integral image

  // Draw the detected points
  //drawIpoints(img, ipts, ipts.size(), orts);

  // Display the result
  //showImage(img);

  cvReleaseImage(&img);
  cvReleaseImage(&gray_img);
  free(orts);
  return 0;
}

//-------------------------------------------------------
//! SURF on images with OpenMP to achieve best performance
int mainCUDAImages(void)
{
  int start = 1; //start index of the image
  int n = 100; //number of imgs
  struct timeval  bd_tick_x, bd_tick_e, bd_tick_d;
  vector<IplImage*> imgs;
  vector<float4> ipts;
  int m = 0;
  float lastsumtime = 0;
  gettimeofday(&bd_tick_x, 0);
  #pragma omp parallel sections
  {
    #pragma omp section
    {
      for ( int i = start; i < n+start; ++i )
        {
          string fileName = "Images/ImageDB/img";
          char buf[5];
          sprintf(buf,"%d",i);
          fileName = fileName + buf;
          fileName = fileName + ".jpg";
          IplImage *img = cvLoadImage(fileName.c_str());
          // convert the image to single channel 32f
          IplImage *gray_img = getGray(img);
          imgs.push_back(gray_img);
          cvReleaseImage(&img);
        }
    }
    /*gettimeofday(&bd_tick_e, 0);*/
    /*GET_TIME(bd_tick_x, bd_tick_e, bd_tick_d);*/
    /*std::cout<< "Load took: " << bd_tick_d.tv_sec*1000 + bd_tick_d.tv_usec/1000 << " ms" << std::endl;*/
    /*gettimeofday(&bd_tick_x, 0);*/
    #pragma omp section
    {
      for ( int i = 0; i < n;)
        {
          if (imgs.size() > i)
            {
              ipts.clear();
              DetDes(imgs[i], ipts, NULL);
              cvReleaseImage(&imgs[i]);
              m += ipts.size();
              i++;
              cout << sumtime - lastsumtime << ",";
              lastsumtime = sumtime;
              //cout << i << endl;
              if (i == n) break;
            }
        }
    }
  }
  gettimeofday(&bd_tick_e, 0);
  GET_TIME(bd_tick_x, bd_tick_e, bd_tick_d);
  cout << endl;
  std::cout<< "CUDA-SURF took: " << bd_tick_d.tv_sec*1000 + bd_tick_d.tv_usec/1000 << " ms" << std::endl;
  cout << "GPU took: " << sumtime << endl;
  cout << "Integral took: " << tsum << endl;
  cout << "Ipoint Num: " << m << endl;

  return 0;
}

//-------------------------------------------------------
//! SURF Match for two images
int mainCUDAMatch()
{

  for (int block = 3; block < 4; block++)
    for (int idx = 2; idx <= 2; idx++)
      {
        struct timeval  bd_tick_x, bd_tick_e, bd_tick_d;
        vector<float4> ipts1;
        float *orts1;
        float4 *des1;
        orts1 = (float *)malloc(IPTSNUM * sizeof(float));

        vector<float4> ipts2;
        float *orts2;
        float4 *des2;
        orts2 = (float *)malloc(IPTSNUM * sizeof(float));

        string fileName1 = "Images/tests/img";
        char buf[5];
        sprintf(buf,"%d",block*6 + 1);
        fileName1 = fileName1 + buf;
        fileName1 = fileName1 + ".ppm";

        string fileName2 = "Images/tests/img";
        sprintf(buf,"%d",block*6 + idx);
        fileName2 = fileName2 + buf;
        fileName2 = fileName2 + ".ppm";

        IplImage *img1 = cvLoadImage(fileName1.c_str());
        IplImage *img2 = cvLoadImage(fileName2.c_str());
        // convert the image to single channel 32f
        IplImage *gray_img1 = getGray(img1);
        // convert the image to single channel 32f
        IplImage *gray_img2 = getGray(img2);

        des1 = DetDes(gray_img1, ipts1, orts1);
        des2 = DetDes(gray_img2, ipts2, orts2);

        gettimeofday(&bd_tick_x, 0);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);

        if (block == 3)
          {
            vector<int2> src1_corners;
            src1_corners.push_back(make_int2(0, 0));
            src1_corners.push_back(make_int2(img1->width - 1, 0));
            src1_corners.push_back(make_int2(img1->width - 1, img1->height - 1));
            src1_corners.push_back(make_int2(0, img1->height - 1));
            vector<int2> src2_corners;
            src2_corners.push_back(make_int2(0, 0));
            src2_corners.push_back(make_int2(img1->width - 1, 0));
            src2_corners.push_back(make_int2(img1->width - 1, img1->height - 1));
            src2_corners.push_back(make_int2(0, img1->height - 1));
            vector<int2> img1_corners;
            vector<int2> img2_corners;
            /*vector<int2> overlap1_corners;*/
            vector<int2> overlap2_corners;
            float H_inv[3][3];
            invert(H3[idx - 2], H_inv);

            /*cout<< H3[idx - 2][0][0] << " ";*/
            /*cout<< H3[idx - 2][0][1] << " ";*/
            /*cout<< H3[idx - 2][0][2] << endl;*/
            /*cout<< H3[idx - 2][1][0] << " ";*/
            /*cout<< H3[idx - 2][1][1] << " ";*/
            /*cout<< H3[idx - 2][1][2] << endl;*/
            /*cout<< H3[idx - 2][2][0] << " ";*/
            /*cout<< H3[idx - 2][2][1] << " ";*/
            /*cout<< H3[idx - 2][2][2] << endl;*/

            /*cout << endl;*/

            /*cout<< H_inv[0][0] << " ";*/
            /*cout<< H_inv[0][1] << " ";*/
            /*cout<< H_inv[0][2] << endl;*/
            /*cout<< H_inv[1][0] << " ";*/
            /*cout<< H_inv[1][1] << " ";*/
            /*cout<< H_inv[1][2] << endl;*/
            /*cout<< H_inv[2][0] << " ";*/
            /*cout<< H_inv[2][1] << " ";*/
            /*cout<< H_inv[2][2] << endl;*/

            /*cout << endl;*/

            //  cout << src_corners[2].x << " "  << src_corners[2].y << endl;
            project_region(src1_corners, img2_corners, H3[idx - 2]);
            Overlap(src2_corners, img2_corners, overlap2_corners);
            /*project_region(src_corners, img1_corners, H3[idx - 2]);*/
            project_region(overlap2_corners, img1_corners, H_inv);
            //Overlap(src1_corners, img1_corners, overlap1_corners);

            for (int i = 0; i < img1_corners.size(); i++)
              cout << img1_corners[i].x << " " << img1_corners[i].y << endl;

            cout << endl;

            /*for(int i = 0; i < overlap1_corners.size(); i++)*/
            /*cout << overlap1_corners[i].x << " " << overlap1_corners[i].y << endl;*/

            cout << endl;

            cout << img2_corners[0].x << " " << img2_corners[0].y << endl;
            cout << img2_corners[1].x << " " << img2_corners[1].y << endl;
            cout << img2_corners[2].x << " " << img2_corners[2].y << endl;
            cout << img2_corners[3].x << " " << img2_corners[3].y << endl;

            cout << endl;

            vector<float4> region_ipts1;
            vector<float4> region_ipts2;
            float4 *region_des1 = (float4 *)malloc(ipts1.size() * 16 * sizeof(float4));
            float4 *region_des2 = (float4 *)malloc(ipts2.size() * 16 * sizeof(float4));

            int region_count = 0;
            for (int i = 0; i < ipts1.size(); i++)
              {
                int2 point = make_int2(ipts1[i].x, ipts1[i].y);
                if (InPolygon(img1_corners, point))
                  {
                    region_ipts1.push_back(ipts1[i]);
                    for (int n = 0; n < 16; n++)
                      region_des1[region_count * 16 + n] = des1[i * 16 + n];
                    region_count++;
                  }
              }

            region_count = 0;
            for (int i = 0; i < ipts2.size(); i++)
              {
                int2 point = make_int2(ipts2[i].x, ipts2[i].y);
                if (InPolygon(overlap2_corners, point))
                  {
                    region_ipts2.push_back(ipts2[i]);
                    for (int n = 0; n < 16; n++)
                      region_des2[region_count * 16 + n] = des2[i * 16 + n];
                    region_count++;
                  }
              }

            free(des1);
            free(des2);

            des1 = region_des1;
            des2 = region_des2;
            ipts1 = region_ipts1;
            ipts2 = region_ipts2;

            // Draw box around object
            for (int i = 0; i < img2_corners.size(); i++ )
              {
                int2 r1 = img2_corners[i % img2_corners.size()];
                int2 r2 = img2_corners[(i+1) % img2_corners.size()];
                cvLine( img2,
                        cvPoint(r1.x,r1.y),
                        cvPoint(r2.x,r2.y),
                        cvScalar(255,255,255),
                        3 );
              }

            // Draw box around object
            for (int i = 0; i < img1_corners.size(); i++ )
              {
                int2 r1 = img1_corners[i % img1_corners.size()];
                int2 r2 = img1_corners[(i+1) % img1_corners.size()];
                cvLine( img1,
                        cvPoint(r1.x,r1.y),
                        cvPoint(r2.x,r2.y),
                        cvScalar(255,255,255),
                        3 );
              }
          }

        if (ipts1.size() == 0 || ipts2.size() == 0)
          continue;
        int matchnum = Match(des1, des2, ipts1.size(), ipts2.size());

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float time;

        cudaEventElapsedTime(&time, start, stop);

        gettimeofday(&bd_tick_e, 0);
        GET_TIME(bd_tick_x, bd_tick_e, bd_tick_d);
        cout << "Test Pic: " << block << "-" << idx << endl;
        cout << "Match Num: " << matchnum << endl;
        cout << "Ipoint Num1: " << ipts1.size() << " Ipoint Num2: " << ipts2.size() <<
             endl;
        cout << "Repeatability: " << (float)matchnum / min((float)ipts1.size(),
             (float)ipts2.size()) << endl;
        cout << "CUDA-SURF match took: " << bd_tick_d.tv_sec*1000 + bd_tick_d.tv_usec/1000 << " ms" << endl;
        cout << "GPU took: " << time << " ms" << endl << endl;

        // Draw the detected points
        drawIpoints(img1, ipts1, ipts1.size(), orts1);

        // Display the result
        showImage(img1);

        // Draw the detected points
        drawIpoints(img2, ipts2, ipts2.size(), orts2);

        // Display the result
        showImage(img2);

        cvReleaseImage(&img1);
        cvReleaseImage(&gray_img1);
        free(orts1);
        free(des1);

        cvReleaseImage(&img2);
        cvReleaseImage(&gray_img2);
        free(orts2);
        free(des2);
      }
  return 0;
}

