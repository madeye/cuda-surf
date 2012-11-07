/*
 * Copyright 2010 PPI FUDAN University
 * CUDA-SURF v0.5
 * Author: Max Lv
 * Revision: 25
 */

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


__global__ void integralRowKernel(float* d_data, float *i_data,
                                  int width, int height, int step)
{
  int r = blockDim.x * blockIdx.x + threadIdx.x;

  if (r >= height)
    return;

  float rs = 0.f;

  for (int c = 0; c < width; c++)
    {
      rs += d_data[r * step + c];
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
void Integral(IplImage *img, float *d_int, float *tmp)
{

  // set up variables for data access
  int height = img->height;
  int width = img->width;
  int step = img->widthStep/sizeof(float);
  float *data   = (float *) img->imageData;

  CUDA_SAFE_CALL(cudaMemcpy(tmp, data, step * height * sizeof(float),
                            cudaMemcpyHostToDevice));

  integralRowKernel<<< (height + 15) / 16, 16 >>> (tmp, d_int, width, height,
          step);
  integralColumnKernel<<< (width + 7) / 8, 8 >>> (d_int, width, height);

  // release the gray image
  CUDA_SAFE_CALL(cudaFree(tmp));
}

void Integral_CPU(IplImage *img, float *d_int, float *tmp)
{

  // set up variables for data access
  int height = img->height;
  int width = img->width;
  int step = img->widthStep/sizeof(float);
  float *data   = (float *) img->imageData;  
  float *i_data = (float *) malloc (width*height*sizeof(float));  

  // first row only
  float rs = 0.0f;
  for(int j=0; j<width; j++) 
  {
    rs += data[j]; 
    i_data[j] = rs;
  }

  // remaining cells are sum above and to the left
  for(int i=1; i<height; ++i) 
  {
    rs = 0.0f;
    for(int j=0; j<width; ++j) 
    {
      rs += data[i*step+j]; 
      i_data[i*width+j] = rs + i_data[(i-1)*width+j];
    }
  }

  CUDA_SAFE_CALL(cudaMemcpy(d_int, i_data, width * height * sizeof(float),
                            cudaMemcpyHostToDevice));

  free(i_data);
  CUDA_SAFE_CALL(cudaFree(tmp));

}

