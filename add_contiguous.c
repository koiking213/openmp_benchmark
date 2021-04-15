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


void naive(float* in1, float* in2, float* out, size_t size) {
    for (int i=0; i<size; i++) {
        out[i] = in1[i] + in2[i];
    }
}

void naive_neon(float* in1, float* in2, float* out, size_t size) {
    for (int i=0; i<size; i+=4) {
        vst1q_f32(&(out[i]), vaddq_f32(vld1q_f32(&(in1[i])), vld1q_f32(&(in2[i]))));
    }
}

void parallel(float* in1, float* in2, float* out, size_t size) {
#pragma omp parallel for
    for (int i=0; i<size; i++) {
        out[i] = in1[i] + in2[i];
    }
}

void parallel_neon(float* in1, float* in2, float* out, size_t size) {
#pragma omp parallel for
    for (int i=0; i<size; i+=4) {
        vst1q_f32(&(out[i]), vaddq_f32(vld1q_f32(&(in1[i])), vld1q_f32(&(in2[i]))));
    }
}

int main() {
    //float a[N];
    float *a = (float*)malloc(N*sizeof(float));
    float *b = (float*)malloc(N*sizeof(float));
    float *c = (float*)malloc(N*sizeof(float));
    for (int i=0; i<N; i++) {
        a[i] = (float)rand()/RAND_MAX;
        b[i] = (float)rand()/RAND_MAX;
    }
    
    naive(a, b, c, N);
    TIME(naive_neon(a, b, c, N))
    TIME(naive(a, b, c, N))
    TIME(parallel_neon(a, b, c, N))
    TIME(parallel(a, b, c, N))
    //TIME(reduction(a))
    //TIME(shared_memory(a, omp_get_max_threads()))
    //TIME(shared_memory_neon(a, omp_get_max_threads()))

}
