/*
 * Copyright 2010 PPI FUDAN University
 * CUDA-SURF v0.5
 * Author: Max Lv
 * Revision: 25
 */

#ifndef _SURF_KERNEL_H_
#define _SURF_KERNEL_H_
//-------------------------------------------------------
//! SURF priors (these need not be done at runtime)
#define pi 3.14159f

__constant__ float gauss25 [7][7] =
{
  0.02350693969273,0.01849121369071,0.01239503121241,0.00708015417522,0.00344628101733,0.00142945847484,0.00050524879060,
  0.02169964028389,0.01706954162243,0.01144205592615,0.00653580605408,0.00318131834134,0.00131955648461,0.00046640341759,
  0.01706954162243,0.01342737701584,0.00900063997939,0.00514124713667,0.00250251364222,0.00103799989504,0.00036688592278,
  0.01144205592615,0.00900063997939,0.00603330940534,0.00344628101733,0.00167748505986,0.00069579213743,0.00024593098864,
  0.00653580605408,0.00514124713667,0.00344628101733,0.00196854695367,0.00095819467066,0.00039744277546,0.00014047800980,
  0.00318131834134,0.00250251364222,0.00167748505986,0.00095819467066,0.00046640341759,0.00019345616757,0.00006837798818,
  0.00131955648461,0.00103799989504,0.00069579213743,0.00039744277546,0.00019345616757,0.00008024231247,0.00002836202103
};

/*__constant__ float gauss33 [11][11] = {*/
/*0.014614763,0.013958917,0.012162744,0.00966788,0.00701053,0.004637568,0.002798657,0.001540738,0.000773799,0.000354525,0.000148179,*/
/*0.013958917,0.013332502,0.011616933,0.009234028,0.006695928,0.004429455,0.002673066,0.001471597,0.000739074,0.000338616,0.000141529,*/
/*0.012162744,0.011616933,0.010122116,0.008045833,0.005834325,0.003859491,0.002329107,0.001282238,0.000643973,0.000295044,0.000123318,*/
/*0.00966788,0.009234028,0.008045833,0.006395444,0.004637568,0.003067819,0.001851353,0.001019221,0.000511879,0.000234524,9.80224E-05,*/
/*0.00701053,0.006695928,0.005834325,0.004637568,0.003362869,0.002224587,0.001342483,0.000739074,0.000371182,0.000170062,7.10796E-05,*/
/*0.004637568,0.004429455,0.003859491,0.003067819,0.002224587,0.001471597,0.000888072,0.000488908,0.000245542,0.000112498,4.70202E-05,*/
/*0.002798657,0.002673066,0.002329107,0.001851353,0.001342483,0.000888072,0.000535929,0.000295044,0.000148179,6.78899E-05,2.83755E-05,*/
/*0.001540738,0.001471597,0.001282238,0.001019221,0.000739074,0.000488908,0.000295044,0.00016243,8.15765E-05,3.73753E-05,1.56215E-05,*/
/*0.000773799,0.000739074,0.000643973,0.000511879,0.000371182,0.000245542,0.000148179,8.15765E-05,4.09698E-05,1.87708E-05,7.84553E-06,*/
/*0.000354525,0.000338616,0.000295044,0.000234524,0.000170062,0.000112498,6.78899E-05,3.73753E-05,1.87708E-05,8.60008E-06,3.59452E-06,*/
/*0.000148179,0.000141529,0.000123318,9.80224E-05,7.10796E-05,4.70202E-05,2.83755E-05,1.56215E-05,7.84553E-06,3.59452E-06,1.50238E-06*/
/*};*/


__constant__ uint id[] = {6,5,4,3,2,1,0,1,2,3,4,5,6};

//-------------------------------------------------------
//! Calculate the value of the 2d gaussian at x,y
__device__ inline float gaussian(int x, int y, float sig)
{
  return (__fdividef(1.0f,(2.0f*pi*sig*sig))) *
         __expf(__fdividef(-(x*x+y*y),(2.0f*sig*sig)));
}

//-------------------------------------------------------
//! Calculate the value of the 2d gaussian at x,y
__device__ float gaussian(float x, float y, float sig)
{
  return __fdividef(1.0f,(2.0f*pi*sig*sig)) *
         __expf( __fdividef(-(x*x+y*y), (2.0f*sig*sig)));
}

//-------------------------------------------------------
//! Calculate Haar wavelet responses in x direction
__device__ float haarX(int row, int column, int s)
{
  return BoxIntegral(row-s/2, column, s, s/2)
         - BoxIntegral(row-s/2, column-s/2, s, s/2);
}

//-------------------------------------------------------
//! Calculate Haar wavelet responses in y direction
__device__ float haarY(int row, int column, int s)
{
  return BoxIntegral(row, column-s/2, s/2, s)
         - BoxIntegral(row-s/2, column-s/2, s/2, s);
}

//-------------------------------------------------------
//! Get the angle from the +ve x-axis of the vector given by (X Y)
__device__ inline float getAngle(float X, float Y)
{
  if (X >= 0 && Y >= 0)
    return atanf(Y/X);

  if (X < 0 && Y >= 0)
    return pi - atanf(-Y/X);

  if (X < 0 && Y < 0)
    return pi + atanf(Y/X);

  if (X >= 0 && Y < 0)
    return 2*pi - atanf(-Y/X);

  return 0;
}

//-------------------------------------------------------
//! Assign the supplied Ipoint an orientation
__global__ void getOrientationStep1(float4 *s_ipts, float3 *res)
{
  // theadIdx 0 - 10 => BlockDim = (11,11)
  uint i = threadIdx.x - 5;
  uint j = threadIdx.y - 5;
  uint idx = blockIdx.x;

  float4 ipt = s_ipts[idx];

  const int s = __float2int_rn(ipt.z),
                r = __float2int_rn(ipt.y),
                    c = __float2int_rn(ipt.x);

  float gauss = 0.f;
  float3 rs;
  rs.x = 0;
  rs.y = 0;
  rs.z = FLT_MAX;

  // calculate haar responses for points within radius of 6*scale
  if (i*i + j*j < 36)
    {
      gauss =gauss25[id[i+6]][id[j+6]];
      rs.x = gauss * haarX(r+j*s, c+i*s, 4*s);
      rs.y = gauss * haarY(r+j*s, c+i*s, 4*s);
      rs.z = getAngle(rs.x, rs.y);
    }

  res[idx * 121 + threadIdx.y * 11 + threadIdx.x] = rs;
}

__global__ void getOrientationStep2(float *d_ort, float3 *d_res)
{

  // 42 threads
  int tid = threadIdx.x;
  int idx = blockIdx.x;

  // calculate the dominant direction
  float ang1= 0.15f * (float)tid;
  float ang2 = ( ang1+pi/3.0f > 2*pi ? ang1-5.0f*pi/3.0f : ang1+pi/3.0f);
  __shared__ float3 res[121];

  if (tid == 0)
    for (uint k = 0; k < 121; ++k)
      res[k] = d_res[idx * 121 + k];

  __syncthreads();

  // loop slides pi/3 window around feature point
  __shared__ float2 sum[42];

  sum[tid] = make_float2(0.f, 0.f);

  for (uint k = 0; k < 121; ++k)
    {

      const float3 rs = res[k];

      // get angle from the x-axis of the sample point
      const float ang = rs.z;

      // determine whether the point is within the window
      int check1 = ang1 < ang2 && ang1 < ang && ang < ang2;
      int check2 = ang2 < ang1 && ((ang > 0 && ang < ang2) || (ang > ang1 && ang < 2*pi));
      int check = check1 || check2;
      sum[tid].x += rs.x * check;
      sum[tid].y += rs.y * check;
    }

  __syncthreads();

  // if the vector produced from this window is longer than all
  // previous vectors then this forms the new dominant direction

  if (tid < 21)
    if (sum[tid].x * sum[tid].x + sum[tid].y * sum[tid].y <
        sum[tid + 21].x * sum[tid + 21].x + sum[tid + 21].y * sum[tid + 21].y)
      {
        // store largest orientation
        sum[tid] = sum[tid + 21];
      }

  __syncthreads();


  if (tid == 19)
    if (sum[tid].x * sum[tid].x + sum[tid].y * sum[tid].y <
        sum[tid + 1].x * sum[tid + 1].x + sum[tid + 1].y * sum[tid + 1].y)
      {
        // store largest orientation
        sum[tid] = sum[tid + 1];
      }

  __syncthreads();


  if (tid < 10)
    if (sum[tid].x * sum[tid].x + sum[tid].y * sum[tid].y <
        sum[tid + 10].x * sum[tid + 10].x + sum[tid + 10].y * sum[tid + 10].y)
      {
        // store largest orientation
        sum[tid] = sum[tid + 10];
      }

  __syncthreads();


  if (tid < 5)
    if (sum[tid].x * sum[tid].x + sum[tid].y * sum[tid].y <
        sum[tid + 5].x * sum[tid + 5].x + sum[tid + 5].y * sum[tid + 5].y)
      {
        // store largest orientation
        sum[tid] = sum[tid + 5];
      }

  __syncthreads();

  if (tid == 3)
    if (sum[tid].x * sum[tid].x + sum[tid].y * sum[tid].y <
        sum[tid + 1].x * sum[tid + 1].x + sum[tid + 1].y * sum[tid + 1].y)
      {
        // store largest orientation
        sum[tid] = sum[tid + 1];
      }

  __syncthreads();

  if (tid < 2)
    if (sum[tid].x * sum[tid].x + sum[tid].y * sum[tid].y <
        sum[tid + 2].x * sum[tid + 2].x + sum[tid + 2].y * sum[tid + 2].y)
      {
        // store largest orientation
        sum[tid] = sum[tid + 2];
      }

  __syncthreads();

  if (tid == 0)
    {
      if (sum[0].x * sum[0].x + sum[0].y * sum[0].y <
          sum[1].x * sum[1].x + sum[1].y * sum[1].y)
        {
          // store largest orientation
          sum[0] = sum[1];
        }
      // assign orientation of the dominant response vector
      d_ort[idx] = getAngle(sum[0].x, sum[0].y);
    }

}



//-------------------------------------------------------
//! Get the modified descriptor. See Agrawal ECCV 08
//! Modified descriptor contributed by Pablo Fernandez
__global__ void getDescriptor(float4 *s_ipts, float4 *d_des, float *d_ort)
{

  __shared__ float des[64];
  __shared__ float len[16];

  __shared__ float rx[24][24];
  __shared__ float ry[24][24];

  float4 ipt;
  float orientation;
  float gauss_s1;

  int x;
  int y;
  float scale;
  float co;
  float si;

  // Dim(4, 4)
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  uint idx = blockIdx.x;
  uint tid = ty * 4 + tx;

  int i = -12 + 5 * tx;
  int j = -12 + 5 * ty;

  ipt = s_ipts[idx];
  orientation = d_ort[idx];

  x = __float2int_rn(ipt.x);
  y = __float2int_rn(ipt.y);
  scale = ipt.z;
  co = __cosf(orientation);
  si = __sinf(orientation);

  int ix , jx, xs, ys;
  float gauss_s2;
  float rrx, rry;

  float dx=0.f, dy=0.f, mdx=0.f, mdy=0.f;

  float scaleco = scale * co;
  float scalesi = scale * si;

  //Calculate descriptor for this interest point
  float cx = 0.5f + (float)(tx);
  float cy = 0.5f + (float)(ty);

  ix = i + 5;
  jx = j + 5;

  xs = __float2int_rn(x + ( -jx*scalesi + ix*scaleco));
  ys = __float2int_rn(y + ( jx*scaleco + ix*scalesi));

  for (int l = -12; l < 12; l++)
    {
      int k = tid - 12;
      //Get coords of sample point on the rotated axis
      int sample_x = __float2int_rn(x + (-l*scalesi + k*scaleco));
      int sample_y = __float2int_rn(y + ( l*scaleco + k*scalesi));

      //Get the gaussian weighted x and y responses
      rx[k + 12][l + 12] = haarX(sample_y, sample_x, 2 * __float2int_rn(scale));
      ry[k + 12][l + 12] = haarY(sample_y, sample_x, 2 * __float2int_rn(scale));
    }

  if (tid < 8)
    for (int l = -12; l < 12; l++)
      {
        int k = tid + 4;
        //Get coords of sample point on the rotated axis
        int sample_x = __float2int_rn(x + (-l*scalesi + k*scaleco));
        int sample_y = __float2int_rn(y + ( l*scaleco + k*scalesi));

        //Get the gaussian weighted x and y responses
        rx[k + 12][l + 12] = haarX(sample_y, sample_x, 2 * __float2int_rn(scale));
        ry[k + 12][l + 12] = haarY(sample_y, sample_x, 2 * __float2int_rn(scale));
      }

  __syncthreads();

#pragma unroll
  for (int k = i; k < i + 9; ++k)
    {
#pragma unroll
      for (int l = j; l < j + 9; ++l)
        {
          //Get coords of sample point on the rotated axis
          int sample_x = __float2int_rn(x + (-l*scalesi + k*scaleco));
          int sample_y = __float2int_rn(y + ( l*scaleco + k*scalesi));

          float t_rx = rx[k + 12][l + 12];
          float t_ry = ry[k + 12][l + 12];
          gauss_s1 = gaussian(xs-sample_x,ys-sample_y,2.5f*scale);

          //Get the gaussian weighted x and y responses on rotated axis
          rrx = gauss_s1*(-t_rx*si + t_ry*co);
          rry = gauss_s1*(t_rx*co + t_ry*si);

          dx += rrx;
          dy += rry;
          mdx += fabs(rrx);
          mdy += fabs(rry);

        }
    }

  //Add the values to the descriptor vector
  gauss_s2 = gaussian(cx-2.0f,cy-2.0f,1.5f);
  des[tid * 4 + 0] = dx*gauss_s2;
  des[tid * 4 + 1] = dy*gauss_s2;
  des[tid * 4 + 2] = mdx*gauss_s2;
  des[tid * 4 + 3] = mdy*gauss_s2;

  len[tid] = (dx*dx + dy*dy + mdx*mdx + mdy*mdy) * gauss_s2*gauss_s2;

  __syncthreads();

  if (tid < 8)
    len[tid] += len[tid + 8];

  __syncthreads();

  if (tid < 4)
    len[tid] += len[tid + 4];

  __syncthreads();

  if (tid < 2)
    len[tid] += len[tid + 2];

  __syncthreads();

  //Convert to Unit Vector
  if (tid == 0)
    len[0] = sqrtf(len[0] + len[1]);

  __syncthreads();

  float4 d;
  d.x = __fdividef(des[tid * 4 + 0], len[0]);
  d.y = __fdividef(des[tid * 4 + 1], len[0]);
  d.z = __fdividef(des[tid * 4 + 2], len[0]);
  d.w = __fdividef(des[tid * 4 + 3], len[0]);
  d_des[idx * 16 + tid] = d;

}

#endif
