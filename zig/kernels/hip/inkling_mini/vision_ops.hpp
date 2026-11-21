#pragma once
#include "engine.hpp"
namespace mini_vision {
// Row per token, channels contiguous. Patch coords within each merge block are TL,TR,BL,BR.
static __device__ int patch_x(int i,int cols) {return i/4%cols*2+i%2;}
static __device__ int patch_y(int i,int cols) {return i/(4*cols)*2+i/2%2;}
static __global__ void position(float* x,const float* table,const float* cls,const float* cls_pos,int rows,int cols) {
    int i=blockIdx.x-1,d=threadIdx.x+blockIdx.y*256;
    if(i<0) {x[d]=cls[d]+cls_pos[d];return;}
    float u=fminf(31,fmaxf(0,(patch_x(i,cols)+.5f)*16/cols-.5f));
    float v=fminf(31,fmaxf(0,(patch_y(i,cols)+.5f)*16/rows-.5f));
    int a=int(u),b=int(v); u-=a;v-=b;
    const float* t=table+d; int r=32*b,s=32*min(31,b+1),c=min(a+1,31);
    x[(i+1)*1024+d]+=(1-v)*((1-u)*t[(r+a)*1024]+u*t[(r+c)*1024])+v*((1-u)*t[(s+a)*1024]+u*t[(s+c)*1024]);
}
static __device__ float sum256(float v,float* shared) {
    int t=threadIdx.x; shared[t]=v; __syncthreads();
    for(int j=128;j;j/=2){if(t<j)shared[t]+=shared[t+j];__syncthreads();}
    float out=shared[0]; __syncthreads(); return out;
}
static __global__ void norm(const float* x,float* out,const float* w,const float* b) {
    __shared__ float tmp[256]; int t=threadIdx.x; x+=blockIdx.x*1024;out+=blockIdx.x*1024;
    float sum=0; for(int d=t;d<1024;d+=256) sum+=x[d]; float mean=sum256(sum,tmp)/1024;
    sum=0; for(int d=t;d<1024;d+=256) {float z=x[d]-mean;sum+=z*z;}
    float inv=rsqrtf(sum256(sum,tmp)/1024+1e-5f);
    for(int d=t;d<1024;d+=256)out[d]=(x[d]-mean)*inv*w[d]+b[d];
}
static __global__ void rope(float* q,float* k,int cols) {
    int i=blockIdx.x-1,p=threadIdx.x+blockIdx.y*256, d=(i+1)*1024+2*p;
    float angle=i<0?0:((p%32<16?patch_x(i,cols):patch_y(i,cols))+1)*powf(10000.f,-(p%16)/16.f);
    float c=cosf(angle),s=sinf(angle),a=q[d],b=q[d+1];q[d]=a*c-b*s;q[d+1]=a*s+b*c;
    a=k[d];b=k[d+1];k[d]=a*c-b*s;k[d+1]=a*s+b*c;
}
// One query/head per block. Store only one softmax row per resident block.
static __global__ void attention(const float* q,const float* k,const float* v,float* out,int n) {
    extern __shared__ float score[]; __shared__ float work[256];
    int t=threadIdx.x,base=blockIdx.x*1024+blockIdx.y*64;float high=-INFINITY;
    for(int j=t;j<n;j+=256){float dot=0;for(int d=0;d<64;++d)dot+=q[base+d]*k[j*1024+blockIdx.y*64+d];
        score[j]=dot*.125f; high=fmaxf(high,score[j]);}
    work[t]=high;__syncthreads();for(int s=128;s;s/=2){if(t<s)work[t]=fmaxf(work[t],work[t+s]);__syncthreads();}
    high=work[0];__syncthreads();float total=0;for(int j=t;j<n;j+=256){score[j]=expf(score[j]-high);total+=score[j];}
    total=sum256(total,work);if(t<64){float z=0;for(int j=0;j<n;++j)z+=score[j]*v[j*1024+blockIdx.y*64+t];out[base+t]=z/total;}
}
static __global__ void activate(float* x,const float* bias,int width,int count,bool gelu) {
    int i=blockIdx.x*256+threadIdx.x;if(i>=count)return;float a=x[i]+(bias?bias[i%width]:0);
    x[i]=gelu?.5f*a*(1+erff(.7071067811865475f*a)):a;
}
static __global__ void residual(float* x,const float* branch,const float* scale,int count) {
    int i=blockIdx.x*256+threadIdx.x;if(i<count)x[i]+=branch[i]*scale[i%1024];
}
}
