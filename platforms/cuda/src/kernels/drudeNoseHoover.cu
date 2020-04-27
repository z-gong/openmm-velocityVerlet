/**
 * Calculate the center of mass velocities of each molecules
 */

extern "C" __global__ void calcCOMVelocities(const mixed4 *__restrict__ velm,
                                             mixed4 *__restrict__ comVelm,
                                             const int2 *__restrict__ particlesInMolecules,
                                             const int *__restrict__ particlesSortedByMolId,
                                             const int *__restrict__ moleculesNH) {

    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_MOLECULES_NH; i += blockDim.x*gridDim.x) {
        int id_mol = moleculesNH[i];
        comVelm[id_mol] = make_mixed4(0,0,0,0);
        mixed comMass = 0.0;
        for (int j = 0; j < particlesInMolecules[id_mol].x; j++) {
            int index = particlesSortedByMolId[particlesInMolecules[id_mol].y + j];
            mixed4 velocity = velm[index];
            if (velocity.w != 0) {
                mixed mass = RECIP(velocity.w);
                comVelm[id_mol].x += velocity.x * mass;
                comVelm[id_mol].y += velocity.y * mass;
                comVelm[id_mol].z += velocity.z * mass;
                comMass += mass;
            }
        }
        comVelm[id_mol].w = RECIP(comMass);
        comVelm[id_mol].x *= comVelm[id_mol].w;
        comVelm[id_mol].y *= comVelm[id_mol].w;
        comVelm[id_mol].z *= comVelm[id_mol].w;
    }
}

/**
 * Calculate the relative velocities of each particles relative to the COM of the molecule
 */

extern "C" __global__ void normalizeVelocities(mixed4 *__restrict__ velm,
                                               const mixed4 *__restrict__ comVelm,
                                               const int *__restrict__ particleMolId,
                                               const int *__restrict__ particlesNH) {

    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_PARTICLES_NH; i += blockDim.x*gridDim.x) {
        int index = particlesNH[i];
        int id_mol = particleMolId[index];
        velm[index].x -= comVelm[id_mol].x;
        velm[index].y -= comVelm[id_mol].y;
        velm[index].z -= comVelm[id_mol].z;
    }
}

/**
 * Calculate the kinetic energies of each degree of freedom.
 */

extern "C" __global__ void computeNormalizedKineticEnergies(const mixed4 *__restrict__ velm,
                                                            const mixed4 *__restrict__ comVelm,
                                                            const int *__restrict__ normalParticles,
                                                            const int2 *__restrict__ pairParticles,
                                                            double *__restrict__ kineticEnergyBuffer,
                                                            const int *__restrict__ moleculesNH,
                                                            int bufferSize) {
    /**
     * the length of kineticEnergyBuff is numParticlesNH*NUM_TG
     * numThreads can be a little bit larger than numParticlesNH
     * each thread initialize NUM_TG sequential elements of kineticEnergyBuffer
     * careful to not cross the boundary of kineticEnergyBuffer
     */

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((tid + 1) * NUM_TG <= bufferSize) {
        for (int i = 0; i < NUM_TG; i++)
            kineticEnergyBuffer[tid * NUM_TG + i] = 0;
    }

    // Add kinetic energy of ordinary particles.
    for (int i = tid; i < NUM_NORMAL_PARTICLES_NH; i += blockDim.x * gridDim.x) {
        int index = normalParticles[i];
        mixed4 velocity = velm[index];
        if (velocity.w != 0) {
            kineticEnergyBuffer[tid * NUM_TG + TG_ATOM] +=
                    (velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z) / velocity.w;
        }
    }

#if NUM_TG > TG_COM
    // Add kinetic energy of molecular motions.
    for (int i = tid; i < NUM_MOLECULES_NH; i += blockDim.x * gridDim.x) {
        int id_mol = moleculesNH[i];
        mixed4 velocity = comVelm[id_mol];
        if (velocity.w != 0)
            kineticEnergyBuffer[tid * NUM_TG + TG_COM] +=
                    (velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z) / velocity.w;
    }
#endif

    // Add kinetic energy of Drude pairs.
    for (int i = tid; i < NUM_PAIRS_NH; i += blockDim.x*gridDim.x) {
        int2 pair = pairParticles[i];
        mixed4 velocity1 = velm[pair.x];
        mixed4 velocity2 = velm[pair.y];
        mixed mass1 = RECIP(velocity1.w);
        mixed mass2 = RECIP(velocity2.w);
        mixed invTotalMass = RECIP(mass1+mass2);
        mixed invReducedMass = (mass1+mass2)*velocity1.w*velocity2.w;
        mixed mass1fract = invTotalMass*mass1;
        mixed mass2fract = invTotalMass*mass2;
        mixed4 cmVel = velocity1*mass1fract+velocity2*mass2fract;
        mixed4 relVel = velocity1-velocity2;

        kineticEnergyBuffer[tid * NUM_TG + TG_ATOM] +=
                (cmVel.x * cmVel.x + cmVel.y * cmVel.y + cmVel.z * cmVel.z) * (mass1 + mass2);
        kineticEnergyBuffer[tid * NUM_TG + TG_DRUDE] +=
                (relVel.x * relVel.x + relVel.y * relVel.y + relVel.z * relVel.z) / invReducedMass;
    }
}

/**
 * Sum up the kinetic energies of each degree of freedom.
 */

extern "C" __global__ void sumNormalizedKineticEnergies(double *__restrict__ kineticEnergyBuffer,
                                                        double *__restrict__ kineticEnergies,
                                                        int bufferSize) {
    /**
     * The numThreads of this kernel equals to threadBlockSize.
     * So there is only one threadBlock for this kernel
     */
    extern __shared__ double temp[];
    unsigned int tid = threadIdx.x;

    for (unsigned int i = 0; i < NUM_TG; i++)
        temp[tid * NUM_TG + i] = 0;
    __syncthreads();

    for (unsigned int i = 0; i < NUM_TG; i++) {
        for (unsigned int index = tid * NUM_TG;
             index + i < bufferSize; index += blockDim.x * NUM_TG) {
            temp[tid * NUM_TG + i] += kineticEnergyBuffer[index + i];
        }
    }
    __syncthreads();
    for (unsigned int i = 0; i < NUM_TG; i++) {
        for (unsigned int k = blockDim.x / 2; k > 0; k >>= 1) {
            if (tid < k)
                temp[tid * NUM_TG + i] += temp[(tid + k) * NUM_TG + i];
            __syncthreads();
        }
    }
    if (tid == 0) {
        for (unsigned int i = 0; i < NUM_TG; i++) {
            kineticEnergies[i] = temp[i];
        }
    }
}

/**
 * Perform the velocity scaling of NoseHoover thermostat.
 */

extern "C" __global__ void scaleVelocity(mixed4 *__restrict__ velm,
                                         const mixed4 *__restrict__ comVelm,
                                         const int *__restrict__ particleMolId,
                                         const int *__restrict__ normalParticles,
                                         const int2 *__restrict__ pairParticles,
                                         const double *__restrict__ vscaleFactors) {

    double vscaleAtom = vscaleFactors[0];
    double vscaleCOM = vscaleFactors[1];
    double vscaleDrude = vscaleFactors[2];
    // Update normal particles.
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_NORMAL_PARTICLES_NH; i += blockDim.x*gridDim.x) {
        int index = normalParticles[i];
        int id_mol = particleMolId[index];
        mixed4 velCOM = comVelm[id_mol];
        if (velm[index].w != 0) {
            velm[index].x = vscaleAtom*velm[index].x + vscaleCOM*velCOM.x;
            velm[index].y = vscaleAtom*velm[index].y + vscaleCOM*velCOM.y;
            velm[index].z = vscaleAtom*velm[index].z + vscaleCOM*velCOM.z;
        }
    }
    
    // Update Drude particle pairs.
    
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_PAIRS_NH; i += blockDim.x*gridDim.x) {
        int2 particles = pairParticles[i];
        int id_mol = particleMolId[particles.x];
        mixed4 velAtom1 = velm[particles.x];
        mixed4 velAtom2 = velm[particles.y];
        mixed4 velCOM = comVelm[id_mol];
        mixed mass1 = RECIP(velAtom1.w);
        mixed mass2 = RECIP(velAtom2.w);
        mixed invTotalMass = RECIP(mass1+mass2);
        mixed mass1fract = invTotalMass*mass1;
        mixed mass2fract = invTotalMass*mass2;
        mixed4 cmVel = velAtom1*mass1fract+velAtom2*mass2fract;
        mixed4 relVel = velAtom2-velAtom1;
        cmVel.x = vscaleAtom*cmVel.x;
        cmVel.y = vscaleAtom*cmVel.y;
        cmVel.z = vscaleAtom*cmVel.z;
        relVel.x = vscaleDrude*relVel.x;
        relVel.y = vscaleDrude*relVel.y;
        relVel.z = vscaleDrude*relVel.z;
        velAtom1.x = cmVel.x-relVel.x*mass2fract + vscaleCOM*velCOM.x;
        velAtom1.y = cmVel.y-relVel.y*mass2fract + vscaleCOM*velCOM.y;
        velAtom1.z = cmVel.z-relVel.z*mass2fract + vscaleCOM*velCOM.z;
        velAtom2.x = cmVel.x+relVel.x*mass1fract + vscaleCOM*velCOM.x;
        velAtom2.y = cmVel.y+relVel.y*mass1fract + vscaleCOM*velCOM.y;
        velAtom2.z = cmVel.z+relVel.z*mass1fract + vscaleCOM*velCOM.z;
        velm[particles.x] = velAtom1;
        velm[particles.y] = velAtom2;
    }
}
