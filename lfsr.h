#ifndef LFSR_H
#define LFSR_H

#include "cv.h"
#include "cxcore.h"

#include <cutil_inline.h>
#include <vector>

struct feature {
    int index;
    float x;
    float y;
    float scl;
    float lap;
};

struct region {
  int left;
  int top;
  int right;
  int bottom;
  std::vector<struct feature> features;
};

std::vector<struct feature>& lfsr(IplImage *img, std::vector<float4> &origin);

#endif
