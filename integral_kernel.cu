/*
 * Copyright 2010 PPI FUDAN University
 * CUDA-SURF v0.5
 * Author: Max Lv
 * Revision: 25
 */

// Texture source image
texture<float, 1, cudaReadModeElementType> TexSrc;

//-------------------------------------------------------
//! Convert image to single channel 32F
IplImage *getGray(const IplImage *img)
{
  // Check we have been supplied a non-null img pointer
  IplImage *gray8, *gray32;

  gray32 = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );

  if ( img->nChannels == 1 )
    gray8 = (IplImage *) cvClone( img );
  else
    {
      gray8 = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
      cvCvtColor( img, gray8, CV_BGR2GRAY );
    }

  cvConvertScale( gray8, gray32, 1.0 / 255.0, 0);
  cvReleaseImage( &gray8 );

  return gray32;
}


__global__ void integralRowKernel(float *i_data,
                                  int width, int height)
{
  int r = blockDim.x * blockIdx.x + threadIdx.x;

  if (r >= height)
    return;

  float rs = 0.f;

  for (int c = 0; c < width; c++)
    {
      rs += tex1Dfetch(TexSrc, r * width + c);
      i_data[r * width + c] = rs;
    }
}

__global__ void integralColumnKernel(float *i_data,
                                     int width, int height)
{
  int c = blockDim.x * blockIdx.x + threadIdx.x;

  if (c >= width)
    return;

  float rs = i_data[c];

  for (int r = 1; r < height; r++)
    {
      rs += i_data[r * width + c];
      i_data[r * width + c] = rs;
    }
}

//-------------------------------------------------------
//! Computes the integral image of image img.  Assumes source image to be a
//! 32-bit floating point.  Returns IplImage of 32-bit float form.
void Integral(IplImage *img, float *d_idata, float *d_data)
{

  // set up variables for data access
  int height = img->height;
  int width = img->width;
  float *data   = (float *) img->imageData;

  CUDA_SAFE_CALL(cudaMemcpy(d_data, data, width * height * sizeof(float),
                            cudaMemcpyHostToDevice));

  CUDA_SAFE_CALL(cudaBindTexture(0, TexSrc,
                                 d_data, width * height * sizeof(float)));

  integralRowKernel<<< (height + 15) / 16, 16 >>> (d_idata, width, height);
  integralColumnKernel<<< (width + 7) / 8, 8 >>> (d_idata, width, height);

  // release the gray image
  CUDA_SAFE_CALL(cudaUnbindTexture(TexSrc));
  //CUDA_SAFE_CALL(cudaFree(d_data));
}

