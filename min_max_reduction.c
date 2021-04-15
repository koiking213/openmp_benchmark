#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <float.h>
#include <arm_neon.h>
#define N 4000000


static double get_time() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + t.tv_usec * 1e-6;
}
#define TIME(X) { double start = get_time(); X; printf("%lf    %s\n", get_time() - start, #X); }


void naive(float* data) {
    float max = FLT_MIN;
    float min = FLT_MAX;
    for (int i=0; i<N; i++) {
        max = fmaxf(max, data[i]);
        min = fminf(min, data[i]);
    }
    printf("max: %f, min: %f\n", max, min);
}

void naive_neon(float* data) {
    float32x4_t vmax = {FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN};
    float32x4_t vmin = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
    for (int i=0; i<N/4*4; i+=4) {
        vmax = vmaxq_f32(vmax, vld1q_f32(data+i));
        vmin = vminq_f32(vmin, vld1q_f32(data+i));
    }
    float max = fmaxf(fmaxf(vmax[0], vmax[1]), fmaxf(vmax[2], vmax[3]));
    float min = fminf(fminf(vmin[0], vmin[1]), fminf(vmin[2], vmin[3]));
    for (int i=N/4*4; i<N; i++) {
        max = fmaxf(max, data[i]);
        min = fminf(min, data[i]);
    }
    printf("max: %f, min: %f\n", max, min);
}

// atomic construct will not accept function call, 
// so this operation is not implementable by atomic construct
//void atomic(float *data) {
//    float max = FLT_MIN;
//    float min = FLT_MAX;
//#pragma omp parallel for
//    for (int i=0; i<N; i++) {
//        #pragma omp atomic capture
//        max = fmaxf(max, data[i]);
//        #pragma omp atomic update
//        min = fminf(min, data[i]);
//    }
//    printf("max: %f, min: %f\n", max, min);
//}

void reduction(float *data) {
    float max = FLT_MIN;
#pragma omp parallel for reduction(max: max)
    for (int i=0; i<N; i++) {
        max = fmaxf(max, data[i]);
    }
    float min = FLT_MAX;
#pragma omp parallel for reduction(min: min)
    for (int i=0; i<N; i++) {
        min = fminf(min, data[i]);
    }
    printf("max: %f, min: %f\n", max, min);
}



void reduction_neon(float *data) {
    float32x4_t vmax = {FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN};
#pragma omp parallel for reduction(max: vmax)
    for (int i=0; i<N; i++) {
        vmax = vmaxq_f32(max, data[i]);
    }
    float min = FLT_MAX;
#pragma omp parallel for reduction(min: min)
    for (int i=0; i<N; i++) {
        min = fminf(min, data[i]);
    }
    printf("max: %f, min: %f\n", max, min);
}


void shared_memory(float* data, int thread_num) {
    float maxs[thread_num];
    float mins[thread_num];
    for (int i=0; i<thread_num; i++) {
        maxs[i] = FLT_MIN;
        mins[i] = FLT_MAX;
    }
#pragma omp parallel for
    for (int i=0; i<N; i++) {
        maxs[omp_get_thread_num()] = fmaxf(maxs[omp_get_thread_num()], data[i]);
        mins[omp_get_thread_num()] = fminf(mins[omp_get_thread_num()], data[i]);
    }
    float max = FLT_MIN;
    float min = FLT_MAX;
    for (int i=0; i<thread_num; i++) {
        max = fmaxf(maxs[i], max);
        min = fminf(mins[i], min);
    }
    printf("max: %f, min: %f\n", max, min);

}

void shared_memory_neon(float* data, int thread_num) {
    float32x4_t vmax[thread_num];
    float32x4_t vmin[thread_num];
    for (int i=0; i<thread_num; i++) {
        vmax[i] = vdupq_n_f32(FLT_MIN);
        vmin[i] = vdupq_n_f32(FLT_MAX);
    }
    double start = get_time();
#pragma omp parallel for
    for (int i=0; i<N/4; i++) {
        vmax[omp_get_thread_num()] = vmaxq_f32(vmax[omp_get_thread_num()], vld1q_f32(data+i*4));
        vmin[omp_get_thread_num()] = vminq_f32(vmin[omp_get_thread_num()], vld1q_f32(data+i*4));
    }
    printf("midtime of shared_memory_neon: %lf\n", get_time()- start);
    float max = FLT_MIN;
    float min = FLT_MAX;
    for (int i=0; i<thread_num; i++) {
        max = fmaxf(max, fmaxf(fmaxf(vmax[i][0], vmax[i][1]), fmaxf(vmax[i][2], vmax[i][3])));
        min = fminf(min, fminf(fminf(vmin[i][0], vmin[i][1]), fminf(vmin[i][2], vmin[i][3])));
    }
    for (int i=N/4*4; i<N; i++) {
        max = fmaxf(max, data[i]);
        min = fminf(min, data[i]);
    }
    printf("max: %f, min: %f\n", max, min);
}

void wrong(float *data) {
    float max = FLT_MIN;
    float min = FLT_MAX;
    /* wrong program */
#pragma omp parallel for
    for (int i=0; i<N; i++) {
        max = fmaxf(max, data[i]);
        min = fminf(min, data[i]);
    }
    printf("max: %f, min: %f\n", max, min);
}

int main() {
    //float a[N];
    float *a = (float*)malloc(N*sizeof(float));
    float max_ref = FLT_MIN;
    float min_ref = FLT_MAX;
    for (int i=0; i<N; i++) {
        a[i] = (float)rand()/RAND_MAX;
        max_ref = fmaxf(max_ref, a[i]);
        min_ref = fminf(min_ref, a[i]);
    }
    printf("max_ref: %f, min_ref: %f\n", max_ref, min_ref);
    
    //TIME(atomic(a))
    TIME(naive(a))
    TIME(naive_neon(a))
    TIME(reduction(a))
    TIME(shared_memory(a, omp_get_max_threads()))
    TIME(shared_memory_neon(a, omp_get_max_threads()))

}