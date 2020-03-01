
/**
 * Perform the velocity update of half step velocity verlet.
 */

extern "C" __global__ void velocityVerletIntegrateVelocities(mixed4 *__restrict__ velm,
                                                             const long long *__restrict__ force,
                                                             const real3 *__restrict__ forceLD,
                                                             mixed4 *__restrict__ posDelta,
                                                             const int *__restrict__ particlesNH,
                                                             const mixed2 *__restrict__ dt,
                                                             const mixed fscale,
                                                             bool updatePosDelta) {

    mixed stepSize = dt[0].y;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < NUM_ATOMS; index += blockDim.x * gridDim.x) {
        mixed4 velocity = velm[index];

//        if (abs(forceLD[index].x > 0.00001) || abs(forceLD[index].y > 0.00001) || abs(forceLD[index].z > 0.00001) )
//            printf("FORCELD index %d , %f %f %f\n", index, forceLD[index].x, forceLD[index].y, forceLD[index].z);

        if (velocity.w != 0) {
            velocity.x += 0.5 * stepSize * velocity.w * forceLD[index].x + fscale * velocity.w * force[index];
            velocity.y += 0.5 * stepSize * velocity.w * forceLD[index].y + fscale * velocity.w * force[index + PADDED_NUM_ATOMS];
            velocity.z += 0.5 * stepSize * velocity.w * forceLD[index].z + fscale * velocity.w * force[index + PADDED_NUM_ATOMS * 2];
            velm[index] = velocity;
            if (updatePosDelta) {
                posDelta[index] = make_mixed4(stepSize * velocity.x, stepSize * velocity.y, stepSize * velocity.z, 0);
            }
        }
    }
}

/**
 * Perform the position update.
 */

extern "C" __global__ void velocityVerletIntegratePositions(real4 *__restrict__ posq,
                                                            real4 *__restrict__ posqCorrection,
                                                            const mixed4 *__restrict__ posDelta,
                                                            mixed4 *__restrict__ velm,
                                                            const mixed2 *__restrict__ dt,
                                                            const int *__restrict__ particlesNH) {
    double invStepSize = 1.0 / dt[0].y;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < NUM_ATOMS; index += blockDim.x * gridDim.x) {
        mixed4 vel = velm[index];
        if (vel.w != 0) {
#ifdef USE_MIXED_PRECISION

            real4 pos1 = posq[index];
            real4 pos2 = posqCorrection[index];
            mixed4 pos = make_mixed4(pos1.x+(mixed)pos2.x, pos1.y+(mixed)pos2.y, pos1.z+(mixed)pos2.z, pos1.w);
#else
            real4 pos = posq[index];
#endif
            mixed4 delta = posDelta[index];
            pos.x += delta.x;
            pos.y += delta.y;
            pos.z += delta.z;
            vel.x = (mixed) (invStepSize*delta.x);
            vel.y = (mixed) (invStepSize*delta.y);
            vel.z = (mixed) (invStepSize*delta.z);
#ifdef USE_MIXED_PRECISION
            posq[index] = make_real4((real) pos.x, (real) pos.y, (real) pos.z, (real) pos.w);
            posqCorrection[index] = make_real4(pos.x-(real) pos.x, pos.y-(real) pos.y, pos.z-(real) pos.z, 0);
#else
            posq[index] = pos;
#endif
            velm[index] = vel;
        }
    }
}

/**
 * Apply hard wall constraints
 */
extern "C" __global__ void applyHardWallConstraints(real4 *__restrict__ posq,
                                                    real4 *__restrict__ posqCorrection,
                                                    mixed4 *__restrict__ velm,
                                                    const int2 *__restrict__ allPairs,
                                                    const mixed2 *__restrict__ dt,
                                                    mixed maxDrudeDistance,
                                                    mixed hardwallscaleDrude) {

    mixed stepSize = dt[0].y;
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i < NUM_ALL_PAIRS; i += blockDim.x*gridDim.x) {
        int2 particles = allPairs[i];
#ifdef USE_MIXED_PRECISION
        real4 posReal1 = posq[particles.x];
        real4 posReal2 = posq[particles.y];
        real4 posCorr1 = posqCorrection[particles.x];
        real4 posCorr2 = posqCorrection[particles.y];
        mixed4 pos1 = make_mixed4(posReal1.x+(mixed)posCorr1.x, posReal1.y+(mixed)posCorr1.y, posReal1.z+(mixed)posCorr1.z, posReal1.w);
        mixed4 pos2 = make_mixed4(posReal2.x+(mixed)posCorr2.x, posReal2.y+(mixed)posCorr2.y, posReal2.z+(mixed)posCorr2.z, posReal2.w);
#else
        mixed4 pos1 = posq[particles.x];
        mixed4 pos2 = posq[particles.y];
#endif
        mixed4 delta = pos1-pos2;
        mixed r = SQRT(delta.x*delta.x + delta.y*delta.y + delta.z*delta.z);
        mixed rInv = RECIP(r);
        if (rInv*maxDrudeDistance < 1) {
            // The constraint has been violated, so make the inter-particle distance "bounce"
            // off the hard wall.

            mixed4 bondDir = delta*rInv;
            mixed4 vel1 = velm[particles.x];
            mixed4 vel2 = velm[particles.y];
            mixed mass1 = RECIP(vel1.w);
            mixed mass2 = RECIP(vel2.w);
            mixed deltaR = r-maxDrudeDistance;
            mixed deltaT = stepSize;
            mixed dotvr1 = vel1.x*bondDir.x + vel1.y*bondDir.y + vel1.z*bondDir.z;
            mixed4 vb1 = bondDir*dotvr1;
            mixed4 vp1 = vel1-vb1;
            if (vel2.w == 0) {
                // The parent particle is massless, so move only the Drude particle.

                if (dotvr1 != 0)
                    deltaT = deltaR/fabs(dotvr1);
                if (deltaT > stepSize)
                    deltaT = stepSize;
                dotvr1 = -dotvr1*hardwallscaleDrude/(fabs(dotvr1)*SQRT(mass1));
                mixed dr = -deltaR + deltaT*dotvr1;
                pos1.x += bondDir.x*dr;
                pos1.y += bondDir.y*dr;
                pos1.z += bondDir.z*dr;
#ifdef USE_MIXED_PRECISION
                posq[particles.x] = make_real4((real) pos1.x, (real) pos1.y, (real) pos1.z, (real) pos1.w);
                posqCorrection[particles.x] = make_real4(pos1.x-(real) pos1.x, pos1.y-(real) pos1.y, pos1.z-(real) pos1.z, 0);
#else
                posq[particles.x] = pos1;
#endif
                vel1.x = vp1.x + bondDir.x*dotvr1;
                vel1.y = vp1.y + bondDir.y*dotvr1;
                vel1.z = vp1.z + bondDir.z*dotvr1;
                velm[particles.x] = vel1;
            }
            else {
                // Move both particles.

                mixed invTotalMass = RECIP(mass1+mass2);
                mixed dotvr2 = vel2.x*bondDir.x + vel2.y*bondDir.y + vel2.z*bondDir.z;
                mixed4 vb2 = bondDir*dotvr2;
                mixed4 vp2 = vel2-vb2;
                mixed vbCMass = (mass1*dotvr1 + mass2*dotvr2)*invTotalMass;
                dotvr1 -= vbCMass;
                dotvr2 -= vbCMass;
                if (dotvr1 != dotvr2)
                    deltaT = deltaR/fabs(dotvr1-dotvr2);
                if (deltaT > stepSize)
                    deltaT = stepSize;
                mixed vBond = hardwallscaleDrude/SQRT(mass1);
                dotvr1 = -dotvr1*vBond*mass2*invTotalMass/fabs(dotvr1);
                dotvr2 = -dotvr2*vBond*mass1*invTotalMass/fabs(dotvr2);
                mixed dr1 = -deltaR*mass2*invTotalMass + deltaT*dotvr1;
                mixed dr2 = deltaR*mass1*invTotalMass + deltaT*dotvr2;
                dotvr1 += vbCMass;
                dotvr2 += vbCMass;
                pos1.x += bondDir.x*dr1;
                pos1.y += bondDir.y*dr1;
                pos1.z += bondDir.z*dr1;
                pos2.x += bondDir.x*dr2;
                pos2.y += bondDir.y*dr2;
                pos2.z += bondDir.z*dr2;
#ifdef USE_MIXED_PRECISION
                posq[particles.x] = make_real4((real) pos1.x, (real) pos1.y, (real) pos1.z, (real) pos1.w);
                posq[particles.y] = make_real4((real) pos2.x, (real) pos2.y, (real) pos2.z, (real) pos2.w);
                posqCorrection[particles.x] = make_real4(pos1.x-(real) pos1.x, pos1.y-(real) pos1.y, pos1.z-(real) pos1.z, 0);
                posqCorrection[particles.y] = make_real4(pos2.x-(real) pos2.x, pos2.y-(real) pos2.y, pos2.z-(real) pos2.z, 0);
#else
                posq[particles.x] = pos1;
                posq[particles.y] = pos2;
#endif
                vel1.x = vp1.x + bondDir.x*dotvr1;
                vel1.y = vp1.y + bondDir.y*dotvr1;
                vel1.z = vp1.z + bondDir.z*dotvr1;
                vel2.x = vp2.x + bondDir.x*dotvr2;
                vel2.y = vp2.y + bondDir.y*dotvr2;
                vel2.z = vp2.z + bondDir.z*dotvr2;
                velm[particles.x] = vel1;
                velm[particles.y] = vel2;
            }
        }
    }
}
