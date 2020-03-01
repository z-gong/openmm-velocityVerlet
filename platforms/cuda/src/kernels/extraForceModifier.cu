/**
 * Reset extra force
 */

extern "C" __global__ void resetExtraForce(real3 *__restrict__ forceExtra) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NUM_ATOMS; i += blockDim.x * gridDim.x) {
        forceExtra[i] = make_real3(0, 0, 0);
    }
}

extern "C" __global__ void addExtraForceDrudeLangevin(const mixed4 *__restrict__ velm,
                                                 real3 *__restrict__ forceExtra,
                                                 const int *__restrict__ normalParticles,
                                                 const int2 *__restrict__ pairParticles,
                                                 mixed dragFactor,
                                                 mixed randFactor,
                                                 mixed dragFactorDrude,
                                                 mixed randFactorDrude,
                                                 const float4 *__restrict__ random,
                                                 unsigned int randomIndex) {
    // Update normal particles

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NUM_NORMAL_PARTICLES_LD; i += blockDim.x * gridDim.x) {
        int index = normalParticles[i];
        mixed4 velocity = velm[index];
        if (velocity.w != 0) {
            mixed mass = RECIP(velocity.w);
            mixed sqrtMass = SQRT(mass);
            float4 rand = random[randomIndex + i];
            forceExtra[index].x += ( - dragFactor * mass * velocity.x + randFactor * sqrtMass * rand.x);
            forceExtra[index].y += ( - dragFactor * mass * velocity.y + randFactor * sqrtMass * rand.y);
            forceExtra[index].z += ( - dragFactor * mass * velocity.z + randFactor * sqrtMass * rand.z);
        }
    }
    // Update Drude particle pairs

    randomIndex += NUM_NORMAL_PARTICLES_LD;
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_PAIRS_LD; i += blockDim.x*gridDim.x) {
        int2 particles = pairParticles[i];
        mixed4 velocity1 = velm[particles.x];
        mixed4 velocity2 = velm[particles.y];
        mixed mass1 = RECIP(velocity1.w);
        mixed mass2 = RECIP(velocity2.w);
        mixed totMass = mass1+mass2;
        mixed sqrtTotMass = SQRT(totMass);
        mixed redMass = RECIP((mass1+mass2)*velocity1.w*velocity2.w);
        mixed sqrtRedMass = SQRT(redMass);
        mixed invTotMass = RECIP(totMass);
        mixed mass1fract = invTotMass*mass1;
        mixed mass2fract = invTotMass*mass2;
        mixed4 cmVel = velocity1*mass1fract+velocity2*mass2fract;
        mixed4 relVel = velocity2-velocity1;

        real3 cmForce;
        real3 relForce;
        float4 rand1 = random[randomIndex+2*i];
        float4 rand2 = random[randomIndex+2*i+1];

        cmForce.x = (-dragFactor * totMass * cmVel.x + randFactor * sqrtTotMass * rand1.x);
        cmForce.y = (-dragFactor * totMass * cmVel.y + randFactor * sqrtTotMass * rand1.y);
        cmForce.z = (-dragFactor * totMass * cmVel.z + randFactor * sqrtTotMass * rand1.z);
        relForce.x = (-dragFactorDrude * redMass * relVel.x + randFactorDrude * sqrtRedMass * rand2.x);
        relForce.y = (-dragFactorDrude * redMass * relVel.y + randFactorDrude * sqrtRedMass * rand2.y);
        relForce.z = (-dragFactorDrude * redMass * relVel.z + randFactorDrude * sqrtRedMass * rand2.z);

        forceExtra[particles.x] += mass1fract * cmForce - relForce;
        forceExtra[particles.y] += mass2fract * cmForce + relForce;
    }
}

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
