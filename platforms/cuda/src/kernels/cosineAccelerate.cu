
extern "C" __global__ void addCosAcceleration(const real4 *__restrict__ posq,
                                              const mixed4 *__restrict__ velm,
                                              real3 *__restrict__ forceExtra,
                                              real acceleration,
                                              const real4 invBoxSize) {

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < NUM_ATOMS; index += blockDim.x * gridDim.x) {
        forceExtra[index].x += acceleration * cos(2 * 3.1415926 * posq[index].z * invBoxSize.z) * RECIP(velm[index].w);
//        if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
//            printf("forceExtraAfter=%f %f %f\n", forceExtra[0].x, forceExtra[100].x, forceExtra[200].x);
//        }
    }
}

extern "C" __global__ void calcPeriodicVelocityBias(const real4 *__restrict__ posq,
                                                    const mixed4 *__restrict__ velm,
                                                    mixed *__restrict__ VBuffer,
                                                    const real4 invBoxSize) {

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < NUM_ATOMS; index += blockDim.x * gridDim.x) {
        if (velm[index].w == 0)
            VBuffer[index] = 0;
        else
            VBuffer[index] = RECIP(velm[index].w) * velm[index].x * 2 * cos(2 * 3.1415926 * posq[index].z * invBoxSize.z);
    }

//    if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
//        printf("buf[0]=%f, mass=%f, vx=%f, z=%f, lz=%f\n",
//                VBuffer[0], RECIP(velm[0].w), velm[0].x, posq[0].z, RECIP(invBoxSize.z));
//    }
}

extern "C" __global__ void sumV(mixed *__restrict__ VBuffer,
                                double invMassTotal,
                                int bufferSize) {
    /**
     * Sum VBuffer
     * The numThreads of this kernel equals to threadBlockSize.
     * So there is only one threadBlock for this kernel
     */
    extern __shared__ mixed temp[];
    unsigned int tid = threadIdx.x;

    temp[tid] = 0;
    for (unsigned int index = tid; index < bufferSize; index += blockDim.x) {
        temp[tid] += VBuffer[index];
    }
    __syncthreads();

    for (unsigned int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (tid < k)
            temp[tid] += temp[tid + k];
        __syncthreads();
    }

    if (tid == 0) {
        VBuffer[0] = temp[0] * invMassTotal;
//        printf("invMassTotal = %f; Vgpu = %f\n", invMassTotal, VBuffer[0]);
    }
}

extern "C" __global__ void removePeriodicVelocityBias(const real4 *__restrict__ posq,
                                                      mixed4 *__restrict__ velm,
                                                      const mixed *__restrict__ VBuffer,
                                                      const real4 invBoxSize) {

    mixed V = VBuffer[0];

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < NUM_ATOMS; index += blockDim.x * gridDim.x) {
        velm[index].x -= V * cos(2 * 3.1415926 * posq[index].z * invBoxSize.z);
    }
}


extern "C" __global__ void restorePeriodicVelocityBias(const real4 *__restrict__ posq,
                                                       mixed4 *__restrict__ velm,
                                                       const mixed *__restrict__ VBuffer,
                                                       const real4 invBoxSize) {
    mixed V = VBuffer[0];

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < NUM_ATOMS; index += blockDim.x * gridDim.x) {
        velm[index].x += V * cos(2 * 3.1415926 * posq[index].z * invBoxSize.z);
    }
}
