/*
 * Copyright 2010 PPI FUDAN University
 * CUDA-SURF v0.6
 * Author: Max Lv
 * Revision: 27
 */

#ifndef _DETDES_KERNEL_H_
#define _DETDES_KERNEL_H_

//-------------------------------------------------------
// pre calculated lobe sizes
__constant__ int lobe_cache [] = {3,5,7,9,5,9,13,17,9,17,25,33,17,33,49,65};
__constant__ int lobe_cache_unique [] = {3,5,7,9,13,17,25,33,49,65};
__constant__ int lobe_map [] = {0,1,2,3,1,3,4,5,3,5,6,7,5,7,8,9};
__constant__ int border_cache [] = {14,26,50,98,194};
int border_cache_host [] = {14,26,50,98,194};
__device__ int counter;

//-------------------------------------------------------
//! Return the value of the approximated determinant of hessian
__device__ inline float getVal(int o, int i, int c, int r)
{
  return fabs(tex1Dfetch(TexDet,
                         (o*intervals+i)*(i_width*i_height)+(r*i_width+c)));
}

//-------------------------------------------------------
//! Return the sign of the laplacian (trace of the hessian)
__device__ inline int getLaplacian(int o, int i, int c, int r)
{
  float res = (tex1Dfetch(TexDet,
                          (o*intervals+i)*(i_width*i_height) + (r*i_width+c)));

  return (res >= 0 ? 1 : -1);
}


//-------------------------------------------------------
//! Non Maximal Suppression function
__device__ int isExtremum(int octave, int interval, int c, int r)
{
  int step = init_sample * (1 << octave);

  // Bounds check
  if (interval - 1 < 0 || interval + 1 > intervals - 1
      || c - step < 0 || c + step > i_width
      || r - step < 0 || r + step > i_height)
    {
      return 0;
    }

  float val = getVal(octave,interval, c, r);

  // Check for maximum
  for (int ii = interval-1; ii <= interval+1; ++ii )
    for (int cc = c - step; cc <= c + step; cc+=step )
      for (int rr = r - step; rr <= r + step; rr+=step )
        if (ii != 0 || cc != 0 || rr != 0)
          if (getVal(octave, ii, cc, rr) > val)
            return 0;

  return 1;
}

//-------------------------------------------------------
//! Matrix multiply
__device__ void inline mul( const float (*H_inv)[3], const float *dD, float *X )
{
#pragma unroll
  for ( int y = 0;  y < 3;  y ++ )
    {
      X[y] = 0;
#pragma unroll
      for ( int i = 0;  i < 3;  i ++ )
        X[y] +=  H_inv[y][i] * dD[i] * -1.0f;
    }

}

//-------------------------------------------------------
//! Invert the matrix
__host__ __device__ inline void invert( const float (*H)[3], float (*H_inv)[3])
{
  float determinant =
    +H[0][0]*(H[1][1]*H[2][2]-H[2][1]*H[1][2])
    -H[0][1]*(H[1][0]*H[2][2]-H[1][2]*H[2][0])
    +H[0][2]*(H[1][0]*H[2][1]-H[1][1]*H[2][0]);

  /*  if( fabs(determinant) < FLT_MIN )*/
  /*{*/
  /*memset( H_inv, 0, sizeof H_inv );*/
  /*return;*/
  /*}*/

  //float invdet = __fdividef(1.0f, determinant);
  float invdet = 1.0f / determinant;
  H_inv[0][0] =  (H[1][1]*H[2][2]-H[2][1]*H[1][2])*invdet;
  H_inv[0][1] = -(H[0][1]*H[2][2]-H[0][2]*H[2][1])*invdet;
  H_inv[0][2] =  (H[0][1]*H[1][2]-H[0][2]*H[1][1])*invdet;
  H_inv[1][0] = -(H[1][0]*H[2][2]-H[1][2]*H[2][0])*invdet;
  H_inv[1][1] =  (H[0][0]*H[2][2]-H[0][2]*H[2][0])*invdet;
  H_inv[1][2] = -(H[0][0]*H[1][2]-H[1][0]*H[0][2])*invdet;
  H_inv[2][0] =  (H[1][0]*H[2][1]-H[2][0]*H[1][1])*invdet;
  H_inv[2][1] = -(H[0][0]*H[2][1]-H[2][0]*H[0][1])*invdet;
  H_inv[2][2] =  (H[0][0]*H[1][1]-H[1][0]*H[0][1])*invdet;

  return;
}


//-------------------------------------------------------
//! Computes the partial derivatives in x, y, and scale of a pixel.
__device__ void inline deriv3D( int octv, int intvl, int r, int c, float *dI )
{
  float dx, dy, ds;
  int step = init_sample * (1 << octv);

  dx = __fdividef(( getVal(octv,intvl, c+step, r ) -
                    getVal( octv,intvl, c-step, r ) ), 2.0f);
  dy = __fdividef(( getVal( octv,intvl, c, r+step ) -
                    getVal( octv,intvl, c, r-step ) ), 2.0f);
  ds = __fdividef(( getVal( octv,intvl+1, c, r ) -
                    getVal( octv,intvl-1, c, r ) ), 2.0f);

  dI[0] = dx;
  dI[1] = dy;
  dI[2] = ds;

  return;
}


//-------------------------------------------------------
//! Computes the 3D Hessian matrix for a pixel.
__device__ inline void hessian3D(int octv, int intvl, int r, int c, float (*H)[3] )
{
  float v, dxx, dyy, dss, dxy, dxs, dys;
  int step = init_sample * (1 << octv);

  v = getVal( octv,intvl, c, r );
  dxx = ( getVal( octv,intvl, c+step, r ) +
          getVal( octv,intvl, c-step, r ) - 2 * v );
  dyy = ( getVal( octv,intvl, c, r+step ) +
          getVal( octv,intvl, c, r-step ) - 2 * v );
  dss = ( getVal( octv,intvl+1, c, r ) +
          getVal( octv,intvl-1, c, r ) - 2 * v );
  dxy = __fdividef(( getVal( octv,intvl, c+step, r+step ) -
                     getVal( octv,intvl, c-step, r+step ) -
                     getVal( octv,intvl, c+step, r-step ) +
                     getVal( octv,intvl, c-step, r-step ) ), 4.0f) ;
  dxs = __fdividef(( getVal( octv,intvl+1, c+step, r ) -
                     getVal( octv,intvl+1, c-step, r ) -
                     getVal( octv,intvl-1, c+step, r ) +
                     getVal( octv,intvl-1, c-step, r ) ), 4.0f);
  dys = __fdividef(( getVal( octv,intvl+1, c, r+step ) -
                     getVal( octv,intvl+1, c, r-step ) -
                     getVal( octv,intvl-1, c, r+step ) +
                     getVal( octv,intvl-1, c, r-step ) ), 4.0f);

  H[0][0] = dxx;
  H[0][1] = dxy;
  H[0][2] = dxs;
  H[1][0] = dxy;
  H[1][1] = dyy;
  H[1][2] = dys;
  H[2][0] = dxs;
  H[2][1] = dys;
  H[2][2] = dss;

  return;
}

//-------------------------------------------------------
//! Performs one step of extremum interpolation.
__device__ float4 interpolateStep( int octv, int intvl, int r, int c)
{
  register float dD[3] = {0}, H[3][3] = {0}, H_inv[3][3] = {0}, X[3] = {0};
  //__shared__ float dD[3], H[3][3], H_inv[3][3], X[3];
  float4 offset;

  deriv3D( octv, intvl, r, c, dD);
  hessian3D( octv, intvl, r, c, H);

  invert( H, H_inv);
  mul( H_inv, dD, X);

  offset.x = X[2];
  offset.y = X[1];
  offset.z = X[0];

  return offset;

}


//-------------------------------------------------------
//! Interpolates a scale-space extremum's location and scale to subpixel
//! accuracy to form an image feature.
__device__ inline bool interpolateExtremum(int octv, int intvl, int r, int c,
    float4 &ipt)
{
  float xi, xr, xc;
  float4 offset;
  int step = init_sample * (1 << octv);

  // Get the offsets to the actual location of the extremum
  offset = interpolateStep( octv, intvl, r, c);

  xi = offset.x;
  xr = offset.y;
  xc = offset.z;

  // If point is sufficiently close to the actual extremum
  if ( fabs( xi ) < 0.5f  &&  fabs( xr ) < 0.5f  &&  fabs( xc ) < 0.5f )
    {
      // Create Ipoint and push onto Ipoints vector
      ipt.x = (float)(c + (step)*xc);
      ipt.y = (float)(r + (step)*xr);
      ipt.z = (1.2f/9.0f) * (3*((float)(1 << (octv+1)) *
                                (intvl+xi+1.0f)+1.0f));
      ipt.w = (float)getLaplacian(octv, intvl, c, r);
      return true;
    }

  return false;
}

//-------------------------------------------------------
//! Calculate determinant of hessian responses
__global__ void buildDetKernel(float *m_det, int o, int border, int step)
{
  int l, w, b;
  float Dxx, Dyy, Dxy, inverse_area;

  const int c = border + (blockDim.x * blockIdx.x + threadIdx.x) * step;
  const int r = border + (blockDim.y * blockIdx.y + threadIdx.y) * step;
  const int i = threadIdx.z;

  if (c > i_width - border  || r >= i_height - border)
    return;

  l = lobe_cache[o*intervals + i];
  w = 3 * l;
  b = w / 2;
  inverse_area = __fdividef(1.0f, (w * w));

  Dxx = BoxIntegral(r - l + 1, c - b, 2*l - 1, w)
        - BoxIntegral(r - l + 1, c - l / 2, 2*l - 1, l)*3;
  Dyy = BoxIntegral(r - b, c - l + 1, w, 2*l - 1)
        - BoxIntegral(r - l / 2, c - l + 1, l, 2*l - 1)*3;
  Dxy = + BoxIntegral(r - l, c + 1, l, l)
        + BoxIntegral(r + 1, c - l, l, l)
        - BoxIntegral(r - l, c - l, l, l)
        - BoxIntegral(r + 1, c + 1, l, l);

  // Normalise the filter responses with respect to their size
  Dxx *= inverse_area;
  Dyy *= inverse_area;
  Dxy *= inverse_area;

  // Get the sign of the laplacian
  int lap_sign = (Dxx+Dyy >= 0 ? 1 : -1);

  // Get the determinant of hessian response
  float determinant = (Dxx*Dyy - 0.81f*Dxy*Dxy);

  m_det[(o*intervals+i)*(i_width*i_height) + (r*i_width+c)]
  = (determinant < 0 ? 0 : lap_sign * determinant);
}

__global__ void zeroDetKernel(float *d_det, int size)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= size)
    return;
  d_det[tid] = 0.f;
}

//-------------------------------------------------------
//! Calculate determinant of hessian responses
void buildDet(float *m_det, int width, int height)
{
  int border, step;

#pragma unroll
  for (int o = 0; o < octaves; o++)
    {
      step = init_sample * (1 << o);
      border = border_cache_host[o];

      int hs = (height - 2 * border + step - 1) / step;
      int ws = (width -  2 * border + step - 1) / step;

      dim3 threads(BLOCK_W, BLOCK_H, intervals);
      dim3 grids((ws + BLOCK_W - 1) / BLOCK_W, (hs + BLOCK_H - 1) / BLOCK_H);
      buildDetKernel<<< grids, threads >>> (m_det, o, border, step);
    }
}

//-------------------------------------------------------
//! Find the image features and write into vector of features
__global__ void getIpointsKernel(float4 *d_ipts, int o, int border, int step)
{

  const int c = border + (blockDim.x * blockIdx.x + threadIdx.x) * step * 2;
  const int r = border + (blockDim.y * blockIdx.y + threadIdx.y) * step * 2;
  const int i = 1 + threadIdx.z * 2;
  float4 ipt;
  bool check = false;

  if (c >= i_width - border || r >= i_height - border)
    return;

  int i_max = -1, r_max = -1, c_max = -1;
  float max_val = 0;

  // Scan the pixels in this block to find the local extremum.
  for (int ii = i; ii < min(i+2, intervals-1); ii += 1)
    for (int rr = r; rr < min(r+2*step, i_height - border); rr += step)
      for (int cc = c; cc < min(c+2*step, i_width - border); cc += step)
        {

          float val = getVal(o, ii, cc, rr);

          // record the max value and its location
          if (val > max_val)
            {
              max_val = val;
              i_max = ii;
              r_max = rr;
              c_max = cc;
            }
        }

  // Check the block extremum is an extremum across boundaries.
  if (max_val > thres && i_max != -1 && isExtremum(o, i_max, c_max, r_max))
    {
      check = interpolateExtremum(o, i_max, r_max, c_max, ipt);
    }
  if (check)
    d_ipts[atomicAdd(&counter, 1)] = ipt;
}

//-------------------------------------------------------
//! Find the image features and write into vector of features
void getIpoints(float4 *d_ipts, int width, int height)
{
  int border, step;

#pragma unroll
  for (int o = 0; o < octaves; o++)
    {
      step = init_sample * (1 << o);
      border = border_cache_host[o];

      int hs = (height - 2 * border + step * 2 - 1) / (step * 2);
      int ws = (width -  2 * border + step * 2 - 1) / (step * 2);
      //std::cout << border + (ws - 1) * step * 2 << " " << ws << std::endl;
      dim3 threads(BLOCK_W, BLOCK_H, (intervals - 2 + 2 - 1) / 2);
      dim3 grids((ws + BLOCK_W - 1) / BLOCK_W, (hs + BLOCK_H - 1) / BLOCK_H);

      getIpointsKernel<<< grids, threads >>> (d_ipts, o, border, step);
    }

}

#endif
