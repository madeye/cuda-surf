#include "lfsr.h"

#include <cxcore.h>
#include <cv.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

#define PI 3.1415926536
#define threshold 0.4f

//-------------------------------------------------------------------
// Predefines
static inline float gaussian_2d(int x, int y, float sig);
static inline float gaussian_2d(int x, int y, float sig);
static inline int border(vector<feature> &features, int w, int min_block_width, bool axis_x);
static inline void region_iponts(vector<feature> &features, struct region &reg);
static void segment(int w, int h, vector<struct feature> &features, vector<struct region> &regs);
static void estimate(int w, int h, vector<struct feature> &features, 
    vector<struct region> &estimates, float factor);
static void estimate_block(struct region &reg, 
    vector<struct region> &estimates, float factor);
static inline int fRound(float num) {return (int)(num + 0.5f);}
// End of predefines
//--------------------------------------------------------------------


// Calculate the value of the 2d gaussian at x
static inline float gaussian_2d(int x, int y, float sig)
{
    return (1.0f / (2.0f*PI*sig*sig)) * exp(-(float(x*x) + float(y*y)) / (2.0f*sig*sig));
}

// Calculate the value of the 1d gaussian at x
static inline float gaussian_1d(int x, float sig)
{
    return (1.0f / (sqrtf(2.0f*PI) * sig)) * exp(-((float) (x*x)) / (2.0f*sig*sig));
}

#define GAUSSIAN_2D

static inline int border(vector<feature> &features, int w, int min_block_width, bool axis_x) {

    const int size = features.size();

    for (int i = 0; i <= min_block_width; i++) {
        int x1 = w / 2 - i;
        int count1 = 0;

        int x2 = w / 2 + i;
        int count2 = 0;

        for (int j = 0; j < size; j++) {

            int s = fRound((9.0f/1.2f) * features[j].scl / 3.0f);

            int x_min = (int) features[j].x - s;
            int x_max = (int) features[j].x + s;

            if (!axis_x) {
                x_min = (int) features[j].y - s;
                x_max = (int) features[j].y + s;
            }

            if (x1 >= x_min && x1 <= x_max) {
                count1++;
            }

            if (x2 >= x_min && x2 <= x_max) {
                count2++;
            }
        }

        if (count1 < 1) {
            return x1;
        } else if (count2 < 1) {
            return x2;
        }
    }
    return -1;

}

static inline void region_iponts(vector<feature> &features, struct region &reg) {
    const int size = features.size();
    for (int i = 0; i < size; i++) {
        feature p = features[i];

        if (p.x >= reg.left && p.x <= reg.right
                && p.y >= reg.top && p.y <= reg.bottom) {
            reg.features.push_back(p);
        }

    }
}


// segment the image according to iponts postions
static void segment(int w, int h, vector<struct feature> &features,
    vector<struct region> &regs) {

    const int min_block_width = w / 4;
    const int min_block_height = h / 4;
    const int min_region_iponts = features.size() / 16;

    {
        int x = border(features, w, min_block_width, true);
        int y = border(features, h, min_block_height, false);

        if (x != -1 && y != -1) {
            struct region reg1, reg2, reg3, reg4;

            reg1.left = 0;
            reg1.top = 0;
            reg1.right = x;
            reg1.bottom = y;

            reg2.left = x;
            reg2.top = 0;
            reg2.right = w - 1;
            reg2.bottom = y;

            reg3.left = 0;
            reg3.top = y;
            reg3.right = x;
            reg3.bottom = h - 1;

            reg4.left = x;
            reg4.top = y;
            reg4.right = w - 1;
            reg4.bottom = h - 1;

            region_iponts(features, reg1);
            region_iponts(features, reg2);
            region_iponts(features, reg3);
            region_iponts(features, reg4);

            if (reg1.features.size() >= min_region_iponts)
                regs.push_back(reg1);
            if (reg2.features.size() >= min_region_iponts)
                regs.push_back(reg2);
            if (reg3.features.size() >= min_region_iponts)
                regs.push_back(reg3);
            if (reg4.features.size() >= min_region_iponts)
                regs.push_back(reg4);


        } else if (x != -1) {
            struct region reg1, reg2;

            reg1.left = 0;
            reg1.top = 0;
            reg1.right = x;
            reg1.bottom = h - 1;

            reg2.left = x;
            reg2.top = 0;
            reg2.right = w - 1;
            reg2.bottom = h - 1;

            region_iponts(features, reg1);
            region_iponts(features, reg2);

            if (reg1.features.size() >= min_region_iponts)
                regs.push_back(reg1);
            if (reg2.features.size() >= min_region_iponts)
                regs.push_back(reg2);


        } else if (y != -1) {
            struct region reg1, reg2;

            reg1.left = 0;
            reg1.top = 0;
            reg1.right = w - 1;
            reg1.bottom = y;

            reg2.left = 0;
            reg2.top = y;
            reg2.right = w - 1;
            reg2.bottom = h - 1;

            region_iponts(features, reg1);
            region_iponts(features, reg2);

            if (reg1.features.size() >= min_region_iponts)
                regs.push_back(reg1);
            if (reg2.features.size() >= min_region_iponts)
                regs.push_back(reg2);

        }

    }
}

// Algorithm adopted from ICME'2009
static void estimate_block(struct region &reg, 
    vector<struct region> &estimates, float factor)
{
    int size = reg.features.size();
    int center_x, center_y;
    int width, height;

    int w = reg.right - reg.left + 1;
    int h = reg.bottom - reg.top + 1;

    // compute gaussian 2d matrix
#ifdef GAUSSIAN_2D

    float *dis = new float[w*h];

    {
        float *kernel = new float[3*3];
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                kernel[y*3 + x] = gaussian_2d(x - 1, y - 1, 6.0f);
                cout << kernel[y*3 + x] << " ";
            }
            cout << endl;
        }

        float f = 1.0f / kernel[0];
        int bw = w / 3;
        int bh = h / 3;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int bx = min(x / bw, 2);
                int by = min(y / bh, 2);
                dis[y*w + x] = kernel[by*3 + bx] * f;
                //cout << dis[y*w+x] << " ";
            }
            //cout << endl;
        }

        delete[] kernel;
    }
#endif

    // compute gaussian 1d matrix

#ifdef GAUSSIAN_1D

    float *dis_x = new float[w];
    float *dis_y = new float[h];

    {
        int DX = 3;
        float *kernel = new float[DX];
        for (int x = 0; x < DX; x++) {
            kernel[x] = gaussian_1d(x - DX / 2, 8.0f);
            cout << kernel[x] << " ";
        }

        cout << endl;

        float f = 1.0f / kernel[0];
        int bw = w / DX;
        int bh = h / DX;

        for (int y = 0; y < h; y++) {
            int by = min(y / bh, DX - 1);
            dis_y[y] = kernel[by] * f;
        }

        for (int x = 0; x < w; x++) {
            int bx = min(x / bw, DX - 1);
            dis_x[x] = kernel[bx] * f;
        }

        delete[] kernel;
    }
#endif

    // compute mean X, Y
    {
        int total_x = 0, total_y = 0, count = 0;

        for (int i = 0; i < size; i++) {

            int x = fRound(reg.features[i].x);
            int y = fRound(reg.features[i].y);
            //int s = fRound(features[i].scale);
            int s = fRound((9.0f/1.2f) * reg.features[i].scl / 3.0f);

            for (int j = max(reg.left, x - s + 1); j <= min(reg.right, x + s - 1); j++) {
                for (int k = max(reg.top, y - s + 1); k <= min(reg.bottom, y + s - 1); k++) {
#ifdef GAUSSIAN_2D
                    float fx = dis[(k - reg.top) * w + j - reg.left];
                    float fy = dis[(k - reg.top) * w + j - reg.left];
#elif GAUSSIAN_1D
                    float fx = dis_x[j - reg.left];
                    float fy = dis_y[k - reg.top];
#else
                    float fx = 1.0f;
                    float fy = 1.0f;
#endif

                    total_x += j * fx;
                    total_y += k * fy;
                    count++;
                }
            }
        }

        if (count == 0) {
#ifdef GAUSSIAN_2D
            delete[] dis;
#endif

#ifdef GAUSSIAN_1D
            delete[] dis_x;
            delete[] dis_y;
#endif
            return;
        }

        center_x = total_x / count;
        center_y = total_y / count;

        //center_x = total_x / size;
        //center_y = total_y / size;
    }

    cout << endl << "center: " << center_x << " " << center_y << endl;

    // compute width and height, width (height) / 2 = standard deviation * factor
#if 1
    {
        float sd_x = 0.0f, sd_y = 0.0f, sum_x = 0.0f, sum_y = 0.0f;
        for (int i = 0; i < size; i++) {
            sum_x += (reg.features[i].x - center_x) * (reg.features[i].x - center_x);
            sum_y += (reg.features[i].y - center_y) * (reg.features[i].y - center_y);
        }

        sd_x = sqrtf(sum_x / size);
        sd_y = sqrtf(sum_y / size);

        width = (int) sd_x * factor * 2;
        height = (int) sd_y * factor * 2;
    }
#endif

    // compute width and height using a heuristic algorithm
#if 1
    {
        int wd = max(1, width / 16), hd = max(1, height / 16);
        while (width < w && height < h) {

            int count = 0;

            int left   = center_x - width  / 2;
            int top    = center_y - height / 2;
            int right  = center_x + width  / 2;
            int bottom = center_y + height / 2;

            left   = max(reg.left, left);
            top    = max(reg.top, top);
            right  = min(reg.right, right);
            bottom = min(reg.bottom, bottom);

            for (int j = 0; j < size; j++) {
                feature p = reg.features[j];
                if (p.x >= left && p.x <= right
                        && p.y >= top && p.y <= bottom) {
                    count++;
                }
            }

            float ratio = ((float) count) / ((float) size);

            if (ratio > threshold) {
                cout << "ratio: " << ratio << endl;
                break;
            }

            width += wd;
            height += hd;
        }
    }
#endif


    // get the extreme region
    struct region ex_rg;

    ex_rg.left   = center_x - width  / 2;
    ex_rg.top    = center_y - height / 2;
    ex_rg.right  = center_x + width  / 2;
    ex_rg.bottom = center_y + height / 2;

    ex_rg.left   = max(reg.left, ex_rg.left);
    ex_rg.top    = max(reg.top, ex_rg.top);
    ex_rg.right  = min(reg.right, ex_rg.right);
    ex_rg.bottom = min(reg.bottom, ex_rg.bottom);

    for (int i = 0; i < size; i++) {
        feature p = reg.features[i];

        if (p.x >= ex_rg.left && p.x <= ex_rg.right
                && p.y >= ex_rg.top && p.y <= ex_rg.bottom) {
            ex_rg.features.push_back(p);
        }

    }

    cout << ex_rg.left << " "
         << ex_rg.top  << " "
         << ex_rg.right  << " "
         << ex_rg.bottom  << " " << endl;

    estimates.push_back(ex_rg);

    cout << "rg" << estimates.size() << ": " << ex_rg.features.size() << endl;

    int max_region_iponts = -1;
    for (int i = 0; i < estimates.size(); i++) {
        int rg_size = estimates[i].features.size();
        if (rg_size > max_region_iponts) {
            max_region_iponts = rg_size;
        }
    }

    for (int i = 0; i < estimates.size();) {
        struct region rg = estimates[i];
        int rg_size = rg.features.size();
        if (rg_size < max_region_iponts / 4) {
            estimates.erase(estimates.begin() + i);
            cout << "\nerase: " << i + 1 << endl;
            continue;
        }
        i++;
    }


#ifdef GAUSSIAN_1D
    delete[] dis_x;
    delete[] dis_y;
#endif

#ifdef GAUSSIAN_2D
    delete[] dis;
#endif


}

// Algorithm adopted from ICME'2009
static void estimate(int w, int h, vector<struct feature> &features, 
    vector<struct region> &estimates, float factor)
{
    int size = features.size();
    int center_x, center_y;
    int width, height;


    // compute gaussian 2d matrix
#ifdef GAUSSIAN_2D

    float *dis = new float[w*h];

    {
        float *kernel = new float[3*3];
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                kernel[y*3 + x] = gaussian_2d(x - 1, y - 1, 6.0f);
                cout << kernel[y*3 + x] << " ";
            }
            cout << endl;
        }

        float f = 1.0f / kernel[0];
        int bw = w / 3;
        int bh = h / 3;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int bx = min(x / bw, 2);
                int by = min(y / bh, 2);
                dis[y*w + x] = kernel[by*3 + bx] * f;
                //cout << dis[y*w+x] << " ";
            }
            //cout << endl;
        }

        delete[] kernel;
    }
#endif

    // compute gaussian 1d matrix

#ifdef GAUSSIAN_1D

    float *dis_x = new float[w];
    float *dis_y = new float[h];

    {
        int DX = 3;
        float *kernel = new float[DX];
        for (int x = 0; x < DX; x++) {
            kernel[x] = gaussian_1d(x - DX / 2, 8.0f);
            cout << kernel[x] << " ";
        }

        cout << endl;

        float f = 1.0f / kernel[0];
        int bw = w / DX;
        int bh = h / DX;

        for (int y = 0; y < h; y++) {
            int by = min(y / bh, DX - 1);
            dis_y[y] = kernel[by] * f;
        }

        for (int x = 0; x < w; x++) {
            int bx = min(x / bw, DX - 1);
            dis_x[x] = kernel[bx] * f;
        }

        delete[] kernel;
    }
#endif

    // compute mean X, Y
    {
        int total_x = 0, total_y = 0, count = 0;

        for (int i = 0; i < size; i++) {

            int x = fRound(features[i].x);
            int y = fRound(features[i].y);
            //int s = fRound(features[i].scale);
            int s = fRound((9.0f/1.2f) * features[i].scl / 3.0f);

            for (int j = max(0, x - s + 1); j <= min(w - 1, x + s - 1); j++) {
                for (int k = max(0, y - s + 1); k <= min(h - 1, y + s - 1); k++) {
#ifdef GAUSSIAN_2D
                    float fx = dis[k*w + j];
                    float fy = dis[k*w + j];
#endif
                    //float f = 1.0f;

#ifdef GAUSSIAN_1D
                    float fx = dis_x[j];
                    float fy = dis_y[k];
#endif

                    total_x += j * fx;
                    total_y += k * fy;
                    count++;
                }
            }

            //float f = 1.0f;
            //total_x += features[i].x * f;
            //total_y += features[i].y * f;
            //count++;
        }

        if (count == 0) {
#ifdef GAUSSIAN_2D
            delete[] dis;
#endif

#ifdef GAUSSIAN_1D
            delete[] dis_x;
            delete[] dis_y;
#endif
            return;
        }

        center_x = total_x / count;
        center_y = total_y / count;

        //center_x = total_x / size;
        //center_y = total_y / size;
    }

    cout << endl << "center: " << center_x << " " << center_y << endl;

    // compute width and height, width (height) / 2 = standard deviation * factor
#if 1
    {
        float sd_x = 0.0f, sd_y = 0.0f, sum_x = 0.0f, sum_y = 0.0f;
        for (int i = 0; i < size; i++) {
            sum_x += (features[i].x - center_x) * (features[i].x - center_x);
            sum_y += (features[i].y - center_y) * (features[i].y - center_y);
        }

        sd_x = sqrtf(sum_x / size);
        sd_y = sqrtf(sum_y / size);

        width = (int) sd_x * factor * 2;
        height = (int) sd_y * factor * 2;
    }
#endif

    // compute width and height using a heuristic algorithm
#if 1
    {
        int wd = max(1, width / 16), hd = max(1, height / 16);

        while (width < w && height < h) {

            int count = 0;

            int left   = center_x - width  / 2;
            int top    = center_y - height / 2;
            int right  = center_x + width  / 2;
            int bottom = center_y + height / 2;

            left   = max(0, left);
            top    = max(0, top);
            right  = min(w - 1, right);
            bottom = min(h - 1, bottom);

            for (int j = 0; j < size; j++) {
                feature p = features[j];
                if (p.x >= left && p.x <= right
                        && p.y >= top && p.y <= bottom) {
                    count++;
                }
            }

            float ratio = ((float) count) / ((float) size);

            if (ratio > threshold) {
                cout << "ratio: " << ratio << endl;
                break;
            }

            width += wd;
            height += hd;
        }
    }
#endif


    // get the extreme region
    struct region ex_rg;

    ex_rg.left   = center_x - width  / 2;
    ex_rg.top    = center_y - height / 2;
    ex_rg.right  = center_x + width  / 2;
    ex_rg.bottom = center_y + height / 2;

    ex_rg.left   = max(0, ex_rg.left);
    ex_rg.top    = max(0, ex_rg.top);
    ex_rg.right  = min(w - 1, ex_rg.right);
    ex_rg.bottom = min(h - 1, ex_rg.bottom);

    for (int i = 0; i < size; i++) {
        feature p = features[i];

        if (p.x >= ex_rg.left && p.x <= ex_rg.right
                && p.y >= ex_rg.top && p.y <= ex_rg.bottom) {
            ex_rg.features.push_back(p);
        }

    }

    cout << "region: "
         << ex_rg.left << " "
         << ex_rg.top  << " "
         << ex_rg.right  << " "
         << ex_rg.bottom  << " " << endl;

    estimates.push_back(ex_rg);

#ifdef GAUSSIAN_1D
    delete[] dis_x;
    delete[] dis_y;
#endif

#ifdef GAUSSIAN_2D
    delete[] dis;
#endif
}

vector<struct feature>& lfsr(IplImage *img, vector<float4> &origin) {

    int size = origin.size();
    cout << "origin: " << size << endl;

    vector<struct feature> features;
    vector<struct region> estimates;
    vector<struct region> regions;

    for (int i = 0; i < origin.size(); i++) {
        struct feature f;
        f.index = i;
        f.x = origin[i].x;
        f.y = origin[i].y;
        f.scl = origin[i].z;
        f.lap = origin[i].w;
        features.push_back(f);
    }

    segment(img->width, img->height, features, regions);

    if (regions.size() == 0) {
        estimate(img->width, img->height, features, estimates, 1.0f);
    } else {
        for (int i = 0; i < regions.size(); i++) {
            estimate_block(regions[i], estimates, 1.0f);
        }
    }

    size = 0;
    vector<struct feature>* result = new vector<struct feature>();
    for (int n = 0; n < estimates.size(); n++) {
        for (int i = 0; i < estimates[n].features.size(); i++) {
            result->push_back(estimates[n].features[i]);
            size++;
        }
    }
    cout << "after: " << size << endl;

    return *result;
}
