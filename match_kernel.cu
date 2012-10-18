/*
 * Copyright 2010 PPI FUDAN University
 * CUDA-SURF v0.5
 * Author: Max Lv
 * Revision: 25
 */

using namespace std;

//! Populate IpPairVec with matched ipts
__device__ int matchnum;
int matchnumGold;

void getMatchesGold(float4 *des1, float4 *des2, int size1, int size2, int2 *matches)
{
  float dist, d1, d2;


  for(int i = 0; i < size1; i++)
  {
    int match = -1;
    d1 = d2 = FLT_MAX;

    for (uint j = 0; j < size2; j++)
    {
      float sum=0.f;
      
      for (int n = 0; n < 16; ++n)
      {
        float4 v1 = des1[i*16 + n];
        float4 v2 = des2[j*16 + n];
        sum += (v1.x - v2.x) * (v1.x - v2.x);
        sum += (v1.y - v2.y) * (v1.y - v2.y);
        sum += (v1.z - v2.z) * (v1.z - v2.z);
        sum += (v1.w - v2.w) * (v1.w - v2.w);
      }

      dist = sqrt(sum);

      if (dist < d1) // if this feature matches better than current best
      {
        d2 = d1;
        d1 = dist;
        match = j;
      }
      else if (dist < d2) // this feature matches better than second best
      {
        d2 = dist;
      }

    }
    // If match has a d1:d2 ratio < 0.65 ipoints are a match
    if (d1 / d2 < 0.65f)
    {
      // Store the change in position
      int2 m = make_int2(i, match);
      matches[matchnumGold++] = m;
    }
  }

}

__global__ void getMatches(float4 *des1, int size1, int size2, int2 *matches)
{
  float dist, d1, d2;

  d1 = d2 = FLT_MAX;
  uint i = blockDim.x * blockIdx.x + threadIdx.x;
  uint tid = threadIdx.x;

  if (i >= size1)
    return;

  __shared__ float4 des[TNUMM * 16];

  for (int n = 0; n < 16; n++)
  {
    des[tid + n*TNUMM] = des1[i*16 + n];
    /*des[tid + (n*4 + 0)*TNUMM] = v.x;*/
    /*des[tid + (n*4 + 1)*TNUMM] = v.y;*/
    /*des[tid + (n*4 + 2)*TNUMM] = v.z;*/
    /*des[tid + (n*4 + 3)*TNUMM] = v.w;*/
  }


  int match = -1;
  for (uint j = 0; j < size2; j++)
  {
    float sum=0.f;
    for (int n = 0; n < 16; ++n)
    {
      float4 v1 = des[tid + n*TNUMM];
      /*v1.x = des[tid + (n*4 + 0)*TNUMM];*/
      /*v1.y = des[tid + (n*4 + 1)*TNUMM];*/
      /*v1.z = des[tid + (n*4 + 2)*TNUMM];*/
      /*v1.w = des[tid + (n*4 + 3)*TNUMM];*/
      float4 v2 = tex1Dfetch(TexDes2, j*16 + n);
      sum += (v1.x - v2.x) * (v1.x - v2.x);
      sum += (v1.y - v2.y) * (v1.y - v2.y);
      sum += (v1.z - v2.z) * (v1.z - v2.z);
      sum += (v1.w - v2.w) * (v1.w - v2.w);
    }
    dist = sum;

    if (dist < d1) // if this feature matches better than current best
    {
      d2 = d1;
      d1 = dist;
      match = j;
    }
    else if (dist < d2) // this feature matches better than second best
    {
      d2 = dist;
    }
  }

  // If match has a d1:d2 ratio < 0.65 ipoints are a match
  if (d1 / d2 < 0.65f)
  {
    // Store the change in position
    int2 m = make_int2(i, match);
    matches[atomicAdd(&matchnum, 1)] = m;
  }

}

// Project the region from pic1 to pic2 with the homegraphy
void project_region(vector<int2> &src_corners, vector<int2> &dst_corners, float h[3][3])
{
  for (int i = 0; i < src_corners.size(); i++ )
  {
    double x = src_corners[i].x, y = src_corners[i].y;
    double Z = 1./(h[2][0]*x + h[2][1]*y + h[2][2]);
    double X = (h[0][0]*x + h[0][1]*y + h[0][2])*Z;
    double Y = (h[1][0]*x + h[1][1]*y + h[1][2])*Z;
    dst_corners.push_back(make_int2(fRound(X), fRound(Y)));
  }
}

//-------------------------------------------------------
//! Point in the polygon

#define INFI 1e5
#define ESP 1e-5

struct LineSegment
{
  int2 pt1, pt2;
};

double Multiply(int2 p1, int2 p2, int2 p0)
{
  return ( (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y) );
}

bool IsOnline(int2 point, LineSegment line)
{
  return( ( fabs(Multiply(line.pt1, line.pt2, point)) < ESP ) &&
      ( ( point.x - line.pt1.x ) * ( point.x - line.pt2.x ) <= 0 ) &&
      ( ( point.y - line.pt1.y ) * ( point.y - line.pt2.y ) <= 0 ) );
}

bool Intersect(LineSegment L1, LineSegment L2)
{
  return( (max(L1.pt1.x, L1.pt2.x) >= min(L2.pt1.x, L2.pt2.x)) &&
      (max(L2.pt1.x, L2.pt2.x) >= min(L1.pt1.x, L1.pt2.x)) &&
      (max(L1.pt1.y, L1.pt2.y) >= min(L2.pt1.y, L2.pt2.y)) &&
      (max(L2.pt1.y, L2.pt2.y) >= min(L1.pt1.y, L1.pt2.y)) &&
      (Multiply(L2.pt1, L1.pt2, L1.pt1) * Multiply(L1.pt2, L2.pt2, L1.pt1) >= 0)
      &&
      (Multiply(L1.pt1, L2.pt2, L2.pt1) * Multiply(L2.pt2, L1.pt2, L2.pt1) >= 0)
      );
}

bool InPolygon(vector<int2> &polygon, int2 point)
{
  int count = 0;
  LineSegment line;
  line.pt1 = point;
  line.pt2.y = point.y;
  line.pt2.x = - INFI;

  for ( int i = 0; i < polygon.size(); i++ )
  {
    LineSegment side;
    side.pt1 = polygon[i];
    side.pt2 = polygon[(i + 1) % polygon.size()];

    if ( IsOnline(point, side) )
    {
      return 1;
    }

    if ( fabs(side.pt1.y - side.pt2.y) < ESP )
    {
      continue;
    }

    if ( IsOnline(side.pt1, line) )
    {
      if ( side.pt1.y > side.pt2.y ) count++;
    }
    else if ( IsOnline(side.pt2, line) )
    {
      if ( side.pt2.y > side.pt1.y ) count++;
    }
    else if ( Intersect(line, side) )
    {
      count++;
    }
  }

  if ( count % 2 == 1 )
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

/*bool IsLineSegmentCross(LineSegment line1, LineSegment line2)*/
/*{*/
/*int linep1, linep2;*/
/*linep1 = line1.pt1.x * (line2.pt1.y - line1.pt2.y) +*/
/*line1.pt2.x * (line1.pt1.y - line2.pt1.y) +*/
/*line2.pt1.x * (line1.pt2.y - line1.pt1.y);*/
/*linep2 = line1.pt1.x * (line2.pt2.y - line1.pt2.y) +*/
/*line1.pt2.x * (line1.pt1.y - line2.pt2.y) +*/
/*line2.pt2.x * (line1.pt2.y - line1.pt1.y);*/
/*if(((linep1 ^ linep2) >= 0) && !(linep1 == 0 && linep2 == 0))*/
/*{*/
/*return false;*/
/*}*/

/*linep1 = line2.pt1.x * (line1.pt1.y - line2.pt2.y) +*/
/*line2.pt2.x * (line2.pt1.y - line1.pt1.y) +*/
/*line1.pt1.x * (line2.pt2.y - line2.pt1.y);*/
/*linep2 = line2.pt1.x * (line1.pt2.y - line2.pt2.y) +*/
/*line2.pt2.x * (line2.pt1.y - line1.pt2.y) +*/
/*line1.pt2.x * (line2.pt2.y - line2.pt1.y);*/
/*if(((linep1 ^ linep2) >= 0) && !(linep1 == 0 && linep2 == 0))*/
/*{*/
/*return false;*/
/*}*/
/*return true;*/
/*}*/

int2 GetCrossPoint(LineSegment line1, LineSegment line2)
{
  int2 crossPoint;
  int tempLeft, tempRight;

  tempLeft = (line2.pt2.x - line2.pt1.x) * (line1.pt1.y - line1.pt2.y) -
    (line1.pt2.x - line1.pt1.x) * (line2.pt1.y - line2.pt2.y);
  tempRight = (line1.pt1.y - line2.pt1.y) * (line1.pt2.x - line1.pt1.x) * (line2.pt2.x - line2.pt1.x)
    + line2.pt1.x * (line2.pt2.y - line2.pt1.y) * (line1.pt2.x - line1.pt1.x)
    - line1.pt1.x * (line1.pt2.y - line1.pt1.y) * (line2.pt2.x - line2.pt1.x);
  crossPoint.x = (int)((double)tempRight / (double)tempLeft);

  tempLeft = (line1.pt1.x - line1.pt2.x) * (line2.pt2.y - line2.pt1.y) -
    (line1.pt2.y - line1.pt1.y) * (line2.pt1.x - line2.pt2.x);
  tempRight = (line2.pt2.x - line1.pt2.x) * (line2.pt2.y - line2.pt1.y) * (line1.pt1.y - line1.pt2.y)
    + line1.pt2.y * (line1.pt1.x - line1.pt2.x) * (line2.pt2.y - line2.pt1.y)
    - line2.pt2.y * (line2.pt1.x - line2.pt2.x) * (line1.pt2.y - line1.pt1.y);
  crossPoint.y = (int)((double)tempRight / (double)tempLeft);

  return crossPoint;
}

void Overlap(vector<int2> &src, vector<int2> &project_region, vector<int2> &overlap)
{
  for (int i = 0; i < project_region.size(); i++)
  {
    LineSegment line;
    line.pt1 = project_region[i];
    line.pt2 = project_region[(i + 1) % project_region.size()];
    if (project_region[i].x >= src[0].x &&
        project_region[i].y >= src[0].y &&
        project_region[i].x < src[2].x &&
        project_region[i].y < src[2].y)
      overlap.push_back(project_region[i]);
    /*line[0].pt1 = project_region[(i - 1) < 0 ? project_region.size() - 1 : i - 1];*/
    /*line[0].pt2 = project_region[i];*/


    /*      for( int m = 0; m < src.size(); m++ ) {*/
    /*LineSegment side;*/
    /*side.pt1 = src[m];*/
    /*side.pt2 = src[(m + 1) % src.size()];*/
    /*if(IsLineSegmentCross(line[0], side))*/
    /*{*/
    /*int2 crossPoint = GetCrossPoint(line[0], side);*/
    /*overlap.push_back(crossPoint);*/
    /*break;*/
    /*}*/
    /*}*/


    int count = 0;
    for ( int m = 0; m < src.size(); m++ )
    {
      LineSegment side;
      side.pt1 = src[m];
      side.pt2 = src[(m + 1) % src.size()];
      if (Intersect(line, side))
      {
        int2 crossPoint = GetCrossPoint(line, side);
        count++;
        overlap.push_back(crossPoint);
      }
    }

    if (count > 1)
    {
      int2 last = overlap[overlap.size() - 1];
      int2 seclast = overlap[overlap.size() - 2];
      int2 trdlast = project_region[i];

      int dist1 = (last.x - seclast.x) * (last.x - seclast.x)
        + (last.y - seclast.y) * (last.y - seclast.y);

      int dist2 = (last.x - trdlast.x) * (last.x - trdlast.x)
        + (last.y - trdlast.y) * (last.y - trdlast.y);

      if (dist1 > dist2)
      {
        overlap[overlap.size() - 1] = seclast;
        overlap[overlap.size() - 2] = last;
      }
    }
  }
}
