/**
 * Calculate the center of mass velocities of each residues
 */

extern "C" __global__ void calcCOMVelocities(const mixed4 *__restrict__ velm,
                                             mixed4 *__restrict__ comVelm,
                                             const int2 *__restrict__ particlesInResidues,
                                             const int *__restrict__ particlesSortedByResId,
                                             const int *__restrict__ residuesNH) {

    // Get COM velocities
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_RESIDUES_NH; i += blockDim.x*gridDim.x) {
        int resid = residuesNH[i];
        comVelm[resid] = make_mixed4(0,0,0,0);
        mixed comMass = 0.0;
        for (int j = 0; j < particlesInResidues[resid].x; j++) {
            int index = particlesSortedByResId[particlesInResidues[resid].y + j];
            mixed4 velocity = velm[index];
            if (velocity.w != 0) {
                mixed mass = RECIP(velocity.w);
                comVelm[resid].x += velocity.x * mass;
                comVelm[resid].y += velocity.y * mass;
                comVelm[resid].z += velocity.z * mass;
                comMass += mass;
            }
        }
        comVelm[resid].w = RECIP(comMass);
        comVelm[resid].x *= comVelm[resid].w;
        comVelm[resid].y *= comVelm[resid].w;
        comVelm[resid].z *= comVelm[resid].w;

//        if (i == 0)
//            printf("residue %d has %d particles and starts at %d and vel %f,%f,%f and mass is %f \n",
//                   i, particlesInResidues[i].x, particlesInResidues[i].y,
//                   comVelm[i].x, comVelm[i].y, comVelm[i].z, RECIP(comVelm[i].w));
    }

}

/**
 * Calculate the relative velocities of each particles relative to the center of mass of each residues
 */

extern "C" __global__ void normalizeVelocities(mixed4 *__restrict__ velm,
                                               const mixed4 *__restrict__ comVelm,
                                               const int *__restrict__ particleResId,
                                               const int *__restrict__ particlesNH) {

    // Get Normalized velocities
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_PARTICLES_NH; i += blockDim.x*gridDim.x) {
        int index = particlesNH[i];
        int resid = particleResId[index];
        velm[index].x -= comVelm[resid].x;
        velm[index].y -= comVelm[resid].y;
        velm[index].z -= comVelm[resid].z;

//        if (i == 0)
//            printf("Particle: %d, Norm velocity: %f, velocity: %f, comVel: %f, mass: %f\n",
//                   i, normVelm[i].x, velm[i].x, comVelm[resid].x, RECIP(normVelm[i].w));
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
                                                            const int *__restrict__ residuesNH,
                                                            int bufferSize) {
    /**
     * the length of kineticEnergyBuff is numParticlesNH*(NUM_TEMP_GROUPS+2)
     * numThreads can be a little bit larger than numParticlesNH
     * each thread initialize (NUM_TEMP_GROUPS+2) sequential elements of kineticEnergyBuffer
     * careful to not cross the boundary of kineticEnergyBuffer
     */

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ((tid + 1) * (NUM_TEMP_GROUPS + 2) <= bufferSize) {
        for (int i = 0; i < NUM_TEMP_GROUPS + 2; i++)
            kineticEnergyBuffer[tid * (NUM_TEMP_GROUPS + 2) + i] = 0;
    }

    // Add kinetic energy of molecular motions.
    for (int i = tid; i < NUM_RESIDUES_NH; i += blockDim.x * gridDim.x) {
        int resid = residuesNH[i];
        mixed4 velocity = comVelm[resid];
        if (velocity.w != 0)
            kineticEnergyBuffer[tid * (NUM_TEMP_GROUPS + 2) + NUM_TEMP_GROUPS] +=
                    (velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z) / velocity.w;
    }

    // Add kinetic energy of ordinary particles.
    for (int i = tid; i < NUM_NORMAL_PARTICLES_NH; i += blockDim.x * gridDim.x) {
        int index = normalParticles[i];
        mixed4 velocity = velm[index];
        if (velocity.w != 0) {
            kineticEnergyBuffer[tid * (NUM_TEMP_GROUPS + 2)] +=
                    (velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z) / velocity.w;
        }
    }

    // Add kinetic energy of Drude particle pairs.
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

        kineticEnergyBuffer[tid*(NUM_TEMP_GROUPS+2)] += (cmVel.x*cmVel.x + cmVel.y*cmVel.y + cmVel.z*cmVel.z)*(mass1+mass2);
        kineticEnergyBuffer[tid*(NUM_TEMP_GROUPS+2)+NUM_TEMP_GROUPS+1] += (relVel.x*relVel.x + relVel.y*relVel.y + relVel.z*relVel.z)*RECIP(invReducedMass);
    }
}

/**
 * Sum up the kinetic energies of each degree of freedom.
 */

extern "C" __global__ void sumNormalizedKineticEnergies(double *__restrict__ kineticEnergyBuffer,
                                                        double *__restrict__ kineticEnergies,
                                                        int bufferSize) {
    /**
     * Sum kineticEnergyBuffer
     * The numThreads of this kernel equals to threadBlockSize.
     * So there is only one threadBlock for this kernel
     */
    extern __shared__ double temp[];
    unsigned int tid = threadIdx.x;

    for (unsigned int i = 0; i < NUM_TEMP_GROUPS + 2; i++)
        temp[tid * (NUM_TEMP_GROUPS + 2) + i] = 0;
    __syncthreads();

    for (unsigned int i = 0; i < NUM_TEMP_GROUPS + 2; i++) {
        for (unsigned int index = tid * (NUM_TEMP_GROUPS + 2);
             index + i < bufferSize; index += blockDim.x * (NUM_TEMP_GROUPS + 2)) {
            temp[tid * (NUM_TEMP_GROUPS + 2) + i] += kineticEnergyBuffer[index + i];
        }
    }
    __syncthreads();
    for (unsigned int i = 0; i < NUM_TEMP_GROUPS + 2; i++) {
        for (unsigned int k = blockDim.x / 2; k > 0; k >>= 1) {
            if (tid < k)
                temp[tid * (NUM_TEMP_GROUPS + 2) + i] += temp[(tid + k) * (NUM_TEMP_GROUPS + 2) + i];
            __syncthreads();
        }
    }
    if (tid == 0) {
        for (unsigned int i = 0; i < NUM_TEMP_GROUPS + 2; i++) {
            kineticEnergies[i] = temp[i];
        }
    }
}

/**
 * Perform the velocity scaling of NoseHoover thermostat.
 */

extern "C" __global__ void integrateDrudeNoseHooverVelocityScale(mixed4 *__restrict__ velm,
                                                                 const mixed4 *__restrict__ comVelm,
                                                                 const int *__restrict__ particleResId,
                                                                 const int *__restrict__ normalParticles,
                                                                 const int2 *__restrict__ pairParticles,
                                                                 const mixed *__restrict__ vscaleFactors) {

    mixed vscaleAtom = vscaleFactors[0];
    mixed vscaleCOM = vscaleFactors[1];
    mixed vscaleDrude = vscaleFactors[2];
    // Update normal particles.
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_NORMAL_PARTICLES_NH; i += blockDim.x*gridDim.x) {
        int index = normalParticles[i];
        int resid = particleResId[index];
        mixed4 velCOM = comVelm[resid];
        if (velm[index].w != 0) {
            velm[index].x = vscaleAtom*velm[index].x + vscaleCOM*velCOM.x;
            velm[index].y = vscaleAtom*velm[index].y + vscaleCOM*velCOM.y;
            velm[index].z = vscaleAtom*velm[index].z + vscaleCOM*velCOM.z;
        }
    }
    
    // Update Drude particle pairs.
    
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_PAIRS_NH; i += blockDim.x*gridDim.x) {
        int2 particles = pairParticles[i];
        int resid = particleResId[particles.x];
        mixed4 velAtom1 = velm[particles.x];
        mixed4 velAtom2 = velm[particles.y];
        mixed4 velCOM = comVelm[resid];
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
