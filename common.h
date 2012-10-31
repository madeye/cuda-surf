/*
 * Copyright 2010 PPI FUDAN University
 * CUDA-SURF v0.5
 * Author: Max Lv
 * Revision: 25
 */

#pragma once

// Some head files not necessary, removed next edition
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <string>
#include <vector>
#include <unistd.h>
#include <pwd.h>
#include <dirent.h>
#include <cutil_inline.h>
#include "cv.h"
#include "highgui.h"

// Definitions
#define BLOCK_W  16
#define BLOCK_H  8
#define TNUM     11
#define TNUMD    16
#define TNUMM    8
#define IPTSNUM  5000

#define DIM 4

#define octaves     4
#define intervals   4
#define init_sample 2
#define thres       0.00001f

#define GET_TIME(start, end, duration)                                     \
   duration.tv_sec = (end.tv_sec - start.tv_sec);                         \
   if (end.tv_usec >= start.tv_usec) {                                     \
      duration.tv_usec = (end.tv_usec - start.tv_usec);                   \
   }                                                                       \
   else {                                                                  \
      duration.tv_usec = (1000000L - (start.tv_usec - end.tv_usec));   \
      duration.tv_sec--;                                                   \
   }                                                                       \
   if (duration.tv_usec >= 1000000L) {                                  \
      duration.tv_sec++;                                                   \
      duration.tv_usec -= 1000000L;                                     \
   }

__constant__ int i_width;
__constant__ int i_height;

// Homegrahpy
float H3[5][3][3] =
{
  //{{1, 2, 3},
  //{3, 3, 1},
  //{5, 3, 1}
  //},
  {{8.7976964e-01, 3.1245438e-01,-3.9430589e+01},
    {-1.8389418e-01, 9.3847198e-01, 1.5315784e+02},
    {1.9641425e-04,-1.6015275e-05, 1.0000000e+00}},

  {{7.6285898e-01,-2.9922929e-01, 2.2567123e+02},
    {3.3443473e-01, 1.0143901e+00,-7.6999973e+01},
    {3.4663091e-04,-1.4364524e-05, 1.0000000e+00}},

  {{6.6378505e-01, 6.8003334e-01,-3.1230335e+01},
    { -1.4495500e-01, 9.7128304e-01, 1.4877420e+02},
    {4.2518504e-04,-1.3930359e-05, 1.0000000e+00}},

  {{6.2544644e-01, 5.7759174e-02, 2.2201217e+02},
    {2.2240536e-01, 1.1652147e+00,-2.5605611e+01},
    {4.9212545e-04,-3.6542424e-05, 1.0000000e+00}},

  {{4.2714590e-01,-6.7181765e-01, 4.5361534e+02},
    {4.4106579e-01, 1.0133230e+00,-4.6534569e+01},
    {5.1887712e-04,-7.8853731e-05, 1.0000000e+00}}
};

// Texture m_det
texture<float, 1, cudaReadModeElementType> TexDet;

// Texture tex_des1
texture<float4, 1, cudaReadModeElementType> TexDes1;

// Texture tex_des2
texture<float4, 1, cudaReadModeElementType> TexDes2;

// Texture integral image
texture<float, 1, cudaReadModeElementType> TexInt;

//-------------------------------------------------------
//! Integral Box for CUDA
//! Bounding test not need here, because texture will return 0 when
//! out of index
__device__ inline float BoxIntegral(int row, int col, int rows, int cols)
{

  // The subtraction by one for row/col is because row/col is inclusive.
  int r1 = min(row,          i_height) - 1;
  int c1 = min(col,          i_width)  - 1;
  int r2 = min(row + rows,   i_height) - 1;
  int c2 = min(col + cols,   i_width)  - 1;

  float A(0.0f), B(0.0f), C(0.0f), D(0.0f);
  A = tex1Dfetch(TexInt, r1 * i_width + c1);
  B = tex1Dfetch(TexInt, r1 * i_width + c2);
  C = tex1Dfetch(TexInt, r2 * i_width + c1);
  D = tex1Dfetch(TexInt, r2 * i_width + c2);

  return fmax(0.f, A - B - C + D);
}

inline int fRound(float flt)
{
  return (int)floor(flt+0.5f);
}

//-------------------------------------------------------
//! Show the provided image and wait for keypress
void showImage(const IplImage *img)
{
  cvNamedWindow("Surf", CV_WINDOW_AUTOSIZE);
  cvShowImage("Surf", img);
  cvWaitKey(0);
}

//-------------------------------------------------------
//! Show the provided image in titled window and wait for keypress
void showImage(char *title,const IplImage *img)
{
  cvNamedWindow(title, CV_WINDOW_AUTOSIZE);
  cvShowImage(title, img);
  cvWaitKey(0);
}

//-------------------------------------------------------
//! Draw all the Ipoints in the provided vector
void drawIpoints(IplImage *img, std::vector<float4> &ipts, int size, float *orts)
{
  float4 ipt;
  float s, o;
  int r1, c1, r2, c2, lap;

  for (unsigned int i = 0; i < size; i++)
    {
      ipt = ipts[i];
      s = ((9.0f/1.2f) * ipt.z) / 3.0f;
      o = orts ? orts[i] : 0;
      lap = fRound(ipt.w);
      r1 = fRound(ipt.y);
      c1 = fRound(ipt.x);
      c2 = fRound(s * cos(o) + c1);
      r2 = fRound(s * sin(o) + r1);

      //    if (o) // Green line indicates orientation
      cvLine(img, cvPoint(c1, r1), cvPoint(c2, r2), cvScalar(0, 255, 0));
      //    else  // Green dot if using upright version
      //      cvCircle(img, cvPoint(c1,r1), 1, cvScalar(0, 255, 0),-1);

      if (lap >= 0)
        { // Blue circles indicate light blobs on dark backgrounds
          cvCircle(img, cvPoint(c1,r1), s, cvScalar(0, 0, 255),1);
        }
      else
        { // Red circles indicate light blobs on dark backgrounds
          cvCircle(img, cvPoint(c1,r1), s, cvScalar(255, 0, 0),1);
        }
    }

}

