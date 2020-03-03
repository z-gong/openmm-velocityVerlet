
extern "C" __global__ void addExtraForceElectricField(real4 *__restrict__ posq,
                                                      real3 *__restrict__ forceExtra,
                                                      const int *__restrict__ particlesElectrolyte,
                                                      mixed efield,
                                                      mixed fscale) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NUM_PARTICLES_ELECTROLYTE; i += blockDim.x * gridDim.x) {
        int index = particlesElectrolyte[i];
        real charge = posq[index].w;
        forceExtra[index].z += fscale * efield * charge;
    }
}
