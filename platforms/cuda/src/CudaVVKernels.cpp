/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2013-2015 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CudaVVKernels.h"
#include "CudaVVKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/CMMotionRemover.h"
#include "CudaBondedUtilities.h"
#include "CudaForceInfo.h"
#include "CudaIntegrationUtilities.h"
#include "SimTKOpenMMRealType.h"
#include <algorithm>
#include <set>
#include <iostream>


using namespace OpenMM;
using namespace std;


CudaIntegrateVVStepKernel::~CudaIntegrateVVStepKernel() {
    if (particlesNH != NULL)
        delete particlesNH;
    if (residuesNH != NULL)
        delete residuesNH;
    if (normalParticlesNH != NULL)
        delete normalParticlesNH;
    if (pairParticlesNH != NULL)
        delete pairParticlesNH;
    if (vscaleFactors != NULL)
        delete vscaleFactors;
    if (particleResId != NULL)
        delete particleResId;
    if (particleTempGroup != NULL)
        delete particleTempGroup;
    if (particlesInResidues != NULL)
        delete particlesInResidues;
    if (particlesSortedByResId != NULL)
        delete particlesSortedByResId;
    if (comVelm != NULL)
        delete comVelm;
    if (normVelm != NULL)
        delete normVelm;
    if (kineticEnergyBuffer != NULL)
        delete kineticEnergyBuffer;
    if (kineticEnergies != NULL)
        delete kineticEnergies;

    if (allPairs != NULL)
        delete allPairs;
    if (forceExtra != NULL)
        delete forceExtra;

    if (particlesLD != NULL)
        delete particlesLD;
    if (normalParticlesLD != NULL)
        delete normalParticlesLD;
    if (pairParticlesLD != NULL)
        delete pairParticlesLD;

    if (particlesElectrolyte != NULL)
        delete particlesElectrolyte;

    if (VBuffer != NULL)
        delete VBuffer;
}

void CudaIntegrateVVStepKernel::initialize(const System& system, const VVIntegrator& integrator, const DrudeForce& force) {
    cu.getPlatformData().initializeContexts(system);
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();
    cu.getIntegrationUtilities().initRandomNumberGenerator((unsigned int) integrator.getRandomNumberSeed());

    numAtoms = cu.getNumAtoms();
    particlesNHVec = integrator.getParticlesNH();
    residuesNHVec = integrator.getResiduesNH();
    numTempGroups = integrator.getNumTempGroups();

    /**
     * By default, all atoms are in the first temperature group
     * Molecular COM motion is after
     * Drude relative motion is the last
     * particlesInResidueVec = pair(numOfParticlesInResidue, indexOfFirstParticleInResidue)
     * particlesSortedByResIdVec records the indexes of particles sorted by residue id
     * so that even when molecules are not successive, it still works
     */

    int numNHChains = integrator.getNumNHChains();
    etaMass = std::vector<vector<double> >(numTempGroups+2, std::vector<double>(numNHChains, 0.0));
    eta = std::vector<vector<double> >(numTempGroups+2, std::vector<double>(numNHChains, 0.0));
    etaDot = std::vector<vector<double> >(numTempGroups+2, std::vector<double>(numNHChains+1, 0.0));
    etaDotDot = std::vector<vector<double> >(numTempGroups+2, std::vector<double>(numNHChains, 0.0));
    tempGroupDof = std::vector<double>(numTempGroups+2, 0.0);

    // Identify particles, pairs and residues

    int id_start = 0;
    for (int resid =0; resid < integrator.getNumResidues(); resid++){
        int n_particles_in_res = 0;
        for (int i = 0; i < system.getNumParticles(); i++) {
            if (integrator.getParticleResId(i) == resid){
                n_particles_in_res ++;
                particlesSortedByResIdVec.push_back(i);
            }
        }
        particlesInResiduesVec.push_back(make_int2(n_particles_in_res, id_start));
        id_start += n_particles_in_res;
    }

    set<int> particlesNHSet, particlesLDSet;

    for (int i = 0; i < system.getNumParticles(); i++) {
        int tg;
        if (integrator.isParticleNH(i))
            particlesNHSet.insert(i);
        if (integrator.isParticleLD(i))
            particlesLDSet.insert(i);
        integrator.getParticleTempGroup(i, tg);
        particleTempGroupVec.push_back(tg);
        int resid = integrator.getParticleResId(i);
        particleResIdVec.push_back(resid);
        double mass = system.getParticleMass(i);
        double resInvMass = integrator.getResInvMass(resid);

        if (integrator.isParticleNH(i) && mass != 0.0) {
            tempGroupDof[tg] += 3;
            if (integrator.getUseCOMTempGroup()) {
                tempGroupDof[tg] -= 3 * mass * resInvMass;
            }
        }
    }
    for (int i = 0; i < force.getNumParticles(); i++) {
        int p, p1, p2, p3, p4;
        int tg, tg1;
        double charge, polarizability, aniso12, aniso34;
        force.getParticleParameters(i, p, p1, p2, p3, p4, charge, polarizability, aniso12, aniso34);
        integrator.getParticleTempGroup(p, tg);
        integrator.getParticleTempGroup(p1, tg1);
        if (tg != tg1)
            throw OpenMMException("Temperature group for Drude particle must be the same as the parent particle");

        allPairsVec.push_back(make_int2(p, p1));

        if (integrator.isParticleNH(p)){
            particlesNHSet.erase(p);
            particlesNHSet.erase(p1);
            tempGroupDof[tg] -= 3;
            tempGroupDof[numTempGroups+1] += 3;
            pairParticlesNHVec.push_back(make_int2(p, p1));
        }

        if (integrator.isParticleLD(p)){
            particlesLDSet.erase(p);
            particlesLDSet.erase(p1);
            pairParticlesLDVec.push_back(make_int2(p, p1));
        }
    }
    normalParticlesNHVec.insert(normalParticlesNHVec.begin(), particlesNHSet.begin(), particlesNHSet.end());
    particlesNH = CudaArray::create<int>(cu, max((int) particlesNHVec.size(), 1), "drudeParticlesNH");
    residuesNH = CudaArray::create<int>(cu, max((int) residuesNHVec.size(), 1), "drudeResiduesNH");
    normalParticlesNH = CudaArray::create<int>(cu, max((int) normalParticlesNHVec.size(), 1), "drudeNormalParticlesNH");
    pairParticlesNH = CudaArray::create<int2>(cu, max((int) pairParticlesNHVec.size(), 1), "drudePairParticlesNH");
    particleResId = CudaArray::create<int>(cu, max((int) particleResIdVec.size(), 1), "drudeParticleResId");
    particleTempGroup = CudaArray::create<int>(cu, max((int) particleTempGroupVec.size(), 1), "drudeParticleTempGroups");
    particlesInResidues = CudaArray::create<int2>(cu, max((int) particlesInResiduesVec.size(), 1), "drudeParticlesInResidues");
    particlesSortedByResId = CudaArray::create<int>(cu, max((int) particlesSortedByResIdVec.size(), 1), "drudeParticlesSortedByResId");
    kineticEnergyBuffer = CudaArray::create<double>(cu, max((int) particlesNHVec.size() * (numTempGroups + 2), 1), "drudeKineticEnergyBuffer");
    kineticEnergies = CudaArray::create<double>(cu, max(numTempGroups+2, 1), "kineticEnergies");
    vscaleFactors = CudaArray::create<double>(cu, max(numTempGroups+2, 1), "drudeScaleFactors");
    allPairs = CudaArray::create<int2>(cu, max((int) allPairsVec.size(), 1), "drudeAllPairs");

    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        comVelm = CudaArray::create<double4>(cu, max(integrator.getNumResidues(), 1), "drudeComVelm");
        normVelm = CudaArray::create<double4>(cu, numAtoms, "drudeNormVelm");
    }
    else {
        comVelm = CudaArray::create<float4>(cu, max(integrator.getNumResidues(), 1), "drudeComVelm");
        normVelm = CudaArray::create<float4>(cu, numAtoms, "drudeNormVelm");
    }

//    std::cout << "ParticlesNH: ";
//    for (auto p: particlesNHVec) {
//        std::cout << p << " ";
//    }
//    std::cout << "\nResiduesNH: ";
//    for (auto p: residuesNHVec) {
//        std::cout << p << " ";
//    }
//    std::cout << "\nParticlesInResidues: ";
//    for (auto &p: particlesInResiduesVec) {
//        std::cout << p.x << ": " << p.y << "; ";
//    }
//    std::cout << "\n";

    if (!particlesNHVec.empty())
        particlesNH->upload(particlesNHVec);
    if (!residuesNHVec.empty())
        residuesNH->upload(residuesNHVec);
    if (!normalParticlesNHVec.empty())
        normalParticlesNH->upload(normalParticlesNHVec);
    if (!pairParticlesNHVec.empty())
        pairParticlesNH->upload(pairParticlesNHVec);
    if (!particleResIdVec.empty())
        particleResId->upload(particleResIdVec);
    if (!particleTempGroupVec.empty())
        particleTempGroup->upload(particleTempGroupVec);
    if (!particlesInResiduesVec.empty())
        particlesInResidues->upload(particlesInResiduesVec);
    if (!particlesSortedByResIdVec.empty())
        particlesSortedByResId->upload(particlesSortedByResIdVec);
    if (!allPairsVec.empty())
        allPairs->upload(allPairsVec);

    // reduce real d.o.f by number of constraints, and 3 if CMMotion remove is true
    for (int i = 0; i < system.getNumConstraints(); i++) {
        int p,p1,tg,tg1;
        double distance;
        system.getConstraintParameters(i, p, p1, distance);
        integrator.getParticleTempGroup(p, tg);
        integrator.getParticleTempGroup(p1, tg1);
        if (tg != tg1)
            throw OpenMMException("Temperature group of constrained particles must be the same");

        if (integrator.isParticleNH(p))
            tempGroupDof[tg] -= 1;
    }
    if (integrator.getUseCOMTempGroup()) {
        tempGroupDof[numTempGroups] = 3 * residuesNHVec.size();
    }

    /** TODO gongzheng @ 2019-12-12
     * BUG -- if not useCOMTempGroup, then COM motion DOF not subtracted from other temp group
     */
    if (integrator.getUseCOMTempGroup()) {
        for (int i = 0; i < system.getNumForces(); i++) {
            if (typeid(system.getForce(i)) == typeid(CMMotionRemover)) {
                tempGroupDof[numTempGroups] -= 3;
                break;
            }
        }
    }

    /**
     *  set DOF to zero if it's negative
     *  just in case, though i cannot image when this would happen
     */
    for (int i = 0; i < numTempGroups + 2; i++)
        tempGroupDof[i] = max(tempGroupDof[i], (double) 0);

    // calculate etaMass
    realkbT = BOLTZ * integrator.getTemperature();
    drudekbT = BOLTZ * integrator.getDrudeTemperature();
    double realEtaMassUnit = realkbT * pow(integrator.getCouplingTime(), 2);
    double drudeEtaMassUnit = drudekbT * pow(integrator.getDrudeCouplingTime(), 2);
    for (int i=0; i < numTempGroups+1; i++) {
        tempGroupNkbT.push_back(tempGroupDof[i] * realkbT);
        etaMass[i][0] = tempGroupDof[i] * realEtaMassUnit;
        for (int ich=1; ich < integrator.getNumNHChains(); ich++) {
            etaMass[i][ich] = realEtaMassUnit;
            etaDotDot[i][ich] = (etaMass[i][ich-1] * etaDot[i][ich-1] * etaDot[i][ich-1] - realkbT) / etaMass[i][ich];
        }
    }
    // drude temp group
    int itg = numTempGroups+1;
    tempGroupNkbT.push_back(tempGroupDof[itg] * drudekbT);
    etaMass[itg][0] = tempGroupDof[itg] * drudeEtaMassUnit;
    for (int ich=1; ich < integrator.getNumNHChains(); ich++) {
        etaMass[itg][ich] = drudeEtaMassUnit;
        if (integrator.getUseDrudeNHChains()) {
            etaDotDot[itg][ich] = (etaMass[itg][ich-1] * etaDot[itg][ich-1] * etaDot[itg][ich-1] - drudekbT) / etaMass[itg][ich];
        }
    }

    cout << "TGNH integrator initialization finished\n"
         << "    Num molecules in NH thermostat: " << residuesNHVec.size() << " / " << integrator.getNumResidues() << "\n"
         << "    Real T: " << integrator.getTemperature() << ", Drude T: " << integrator.getDrudeTemperature() << "\n"
         << "    Real coupling Time: " << integrator.getCouplingTime() << ", Drude coupling Time: " << integrator.getDrudeCouplingTime() << "\n";
    for (int i=0; i< numTempGroups+2; i++) {
        cout << "    NkbT[" << i << "]: " << tempGroupNkbT[i] << ", etaMass[" << i << "]: " << etaMass[i][0] << ", DOF[" << i << "]: " << tempGroupDof[i] << "\n";
    }
    cout << "    Loops per NH Step: " << integrator.getLoopsPerStep() << ", Num NH Chain: " << integrator.getNumNHChains() << ", useDrudeNHChains: " << integrator.getUseDrudeNHChains() << "\n"
         << "    Num Temperature groups: " << numTempGroups << "\n"
         << "    Use COM Temperature group: " << integrator.getUseCOMTempGroup() << "\n";

    // Create kernels.
    map<string, string> definesNH;
    definesNH["NUM_PARTICLES_NH"] = cu.intToString(particlesNHVec.size());
    definesNH["NUM_RESIDUES_NH"] = cu.intToString(residuesNHVec.size());
    definesNH["NUM_ATOMS"] = cu.intToString(numAtoms);
    definesNH["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    definesNH["NUM_NORMAL_PARTICLES_NH"] = cu.intToString(normalParticlesNHVec.size());
    definesNH["NUM_PAIRS_NH"] = cu.intToString(pairParticlesNHVec.size());
    definesNH["NUM_ALL_PAIRS"] = cu.intToString(allPairsVec.size());
    definesNH["NUM_TEMP_GROUPS"] = cu.intToString(numTempGroups);

    cout << "    NUM_ATOMS: " << numAtoms << ", PADDED_NUM_ATOMS: " << cu.getPaddedNumAtoms() << "\n"
         << "    NH NUM_NORMAL_PARTICLES: " << normalParticlesNHVec.size() << ", NUM_PAIRS: " << pairParticlesNHVec.size() << "\n"
         << "    numThreadBlocks: " << cu.getNumThreadBlocks() << " ThreadBlockSize: " << cu.ThreadBlockSize << "\n";

    CUmodule moduleNH = cu.createModule(CudaVVKernelSources::vectorOps + CudaVVKernelSources::drudeNoseHoover, definesNH, "");
    kernelCOMVel = cu.getKernel(moduleNH, "calcCOMVelocities");
    kernelNormVel = cu.getKernel(moduleNH, "normalizeVelocities");
    kernelKE = cu.getKernel(moduleNH, "computeNormalizedKineticEnergies");
    kernelKESum = cu.getKernel(moduleNH, "sumNormalizedKineticEnergies");
    kernelScale = cu.getKernel(moduleNH, "integrateDrudeNoseHooverVelocityScale");

    map<string, string> defines;
    defines["NUM_ATOMS"] = cu.intToString(numAtoms);
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    CUmodule module = cu.createModule(CudaVVKernelSources::vectorOps + CudaVVKernelSources::velocityVerlet, defines, "");
    kernelVel = cu.getKernel(module, "velocityVerletIntegrateVelocities");
    kernelPos = cu.getKernel(module, "velocityVerletIntegratePositions");
    hardwallKernel = cu.getKernel(module, "applyHardWallConstraints");

    prevStepSize = -1.0;
    cout << "CUDA modules for TGNH integrator are created\n" << flush;

    normalParticlesLDVec.insert(normalParticlesLDVec.begin(), particlesLDSet.begin(), particlesLDSet.end());
    normalParticlesLD = CudaArray::create<int>(cu, max((int) normalParticlesLDVec.size(), 1), "drudeNormalParticlesLD");
    pairParticlesLD = CudaArray::create<int2>(cu, max((int) pairParticlesLDVec.size(), 1), "drudePairParticlesLD");

    if (!normalParticlesLDVec.empty())
        normalParticlesLD->upload(normalParticlesLDVec);
    if (!pairParticlesLDVec.empty())
        pairParticlesLD->upload(pairParticlesLDVec);

    // init forceExtra
    if (cu.getUseDoublePrecision()) {
        forceExtra = CudaArray::create<double3>(cu, max(numAtoms, 1), "drudeExtraForce");
        auto forceExtraVec = std::vector<double3>(numAtoms, make_double3(0, 0, 0));
        forceExtra->upload(forceExtraVec);
    }
    else {
        forceExtra = CudaArray::create<float3>(cu, max(numAtoms, 1), "drudeExtraForce");
        auto forceExtraVec = std::vector<float3>(numAtoms, make_float3(0, 0, 0));
        forceExtra->upload(forceExtraVec);
    }

    const auto& particlesElectrolyteVec = integrator.getParticlesElectrolyte();
    particlesElectrolyte = CudaArray::create<int>(cu, max((int) particlesElectrolyteVec.size(), 1), "particlesElectrolyte");
    if (!particlesElectrolyteVec.empty())
        particlesElectrolyte->upload(particlesElectrolyteVec);

    map<string, string> definesExtra;
    definesExtra["NUM_ATOMS"] = cu.intToString(numAtoms);
    definesExtra["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    definesExtra["NUM_NORMAL_PARTICLES_LD"] = cu.intToString(normalParticlesLDVec.size());
    definesExtra["NUM_PAIRS_LD"] = cu.intToString(pairParticlesLDVec.size());
    definesExtra["NUM_PARTICLES_ELECTROLYTE"] = cu.intToString(particlesElectrolyteVec.size());
    CUmodule moduleExtra = cu.createModule(CudaVVKernelSources::vectorOps + CudaVVKernelSources::extraForceModifier, definesExtra, "");
    kernelResetExtraForce = cu.getKernel(moduleExtra, "resetExtraForce");
    kernelApplyLangevin = cu.getKernel(moduleExtra, "addExtraForceDrudeLangevin");
    kernelApplyElectricField = cu.getKernel(moduleExtra, "addExtraForceElectricField");

    if (integrator.getCosAcceleration() != 0.0){
        map<string, string> definesPP;
        definesPP["NUM_ATOMS"] = cu.intToString(numAtoms);
        definesPP["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
        CUmodule modulePP= cu.createModule(CudaVVKernelSources::vectorOps + CudaVVKernelSources::periodicPerturbation, definesPP, "");
        kernelCosAccelerate = cu.getKernel(modulePP, "addCosAcceleration");
        kernelCalcBias = cu.getKernel(modulePP, "calcPeriodicVelocityBias");
        kernelRemoveBias = cu.getKernel(modulePP, "removePeriodicVelocityBias");
        kernelRestoreBias = cu.getKernel(modulePP, "restorePeriodicVelocityBias");
        kernelSumV = cu.getKernel(modulePP, "sumV");

        if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision())
            VBuffer = CudaArray::create<double>(cu, numAtoms, "periodicPerturbationVBuffer");
        else
            VBuffer = CudaArray::create<float>(cu, numAtoms, "periodicPerturbationVBuffer");

        double massTotal = 0;
        for (int i = 0; i < numAtoms; i++)
            massTotal += system.getParticleMass(i);
        invMassTotal = 1.0 / massTotal;
    }
}

/* ----------------------------------------------------------------------
   perform half-step update of chain thermostat variables
------------------------------------------------------------------------- */
void CudaIntegrateVVStepKernel::propagateNHChain(ContextImpl& context, const VVIntegrator& integrator) {
    cu.setAsCurrent();

    double stepSize = integrator.getStepSize();
    int    numLoopsPerStep = integrator.getLoopsPerStep();
    double dtc = stepSize/numLoopsPerStep;
    double dtc2 = dtc/2.0;
    double dtc4 = dtc/4.0;
    double dtc8 = dtc/8.0;

    bool useCOMGroup = integrator.getUseCOMTempGroup();
    void *useCOMTempGroupPtr = &useCOMGroup;
    void *argsCOMVel[] = {&cu.getVelm().getDevicePointer(),
                          &particlesInResidues->getDevicePointer(),
                          &particlesSortedByResId->getDevicePointer(),
                          &comVelm->getDevicePointer(),
                          useCOMTempGroupPtr,
                          &residuesNH->getDevicePointer()};
    cu.executeKernel(kernelCOMVel, argsCOMVel, residuesNHVec.size());

    void *argsNormVel[] = {&cu.getVelm().getDevicePointer(),
                           &particleResId->getDevicePointer(),
                           &comVelm->getDevicePointer(),
                           &normVelm->getDevicePointer(),
                           &particlesNH->getDevicePointer()};
    cu.executeKernel(kernelNormVel, argsNormVel, particlesNHVec.size());

    /**
     * Run kinetic energy calculations in the CUDA kernel
     * Set bufferSize = particlesNHVec.size() * (numTempGroups + 2)
     * instead of kineticEnergyBuffer->getSize()
     * in case there's no particle in NH thermostat
     */
    int bufferSize = particlesNHVec.size() * (numTempGroups + 2);
    void *argsKE[] = {&comVelm->getDevicePointer(),
                      &normVelm->getDevicePointer(),
                      &particleTempGroup->getDevicePointer(),
                      &normalParticlesNH->getDevicePointer(),
                      &pairParticlesNH->getDevicePointer(),
                      &kineticEnergyBuffer->getDevicePointer(),
                      &residuesNH->getDevicePointer(),
                      &bufferSize};
    cu.executeKernel(kernelKE, argsKE, particlesNHVec.size());

    // Use only one threadBlock for this kernel because we use shared memory
    void *argsKESum[] = {&kineticEnergyBuffer->getDevicePointer(),
                         &kineticEnergies->getDevicePointer(),
                         &bufferSize};
    cu.executeKernel(kernelKESum, argsKESum, cu.ThreadBlockSize, cu.ThreadBlockSize,
                     cu.ThreadBlockSize * (numTempGroups + 2) * kineticEnergyBuffer->getElementSize());

    kineticEnergiesVec = std::vector<double>(numTempGroups + 2);
    kineticEnergies->download(kineticEnergiesVec);

//    std::cout << cu.getStepCount() << " NH group kinetic energies: ";
//    for (auto ke: kineticEnergiesVec) {
//        std::cout<< ke / 2 << "; ";
//    }
//    std::cout<< "\n";

    // Calculate scaling factor for velocities for each temperature group using Nose-Hoover chain
    vscaleFactorsVec = std::vector<double>(numTempGroups + 2, 1.0);
    vector<double> expfac(numTempGroups+2);
    for (int itg = 0; itg < numTempGroups+2; itg++) {
        double kbT = itg < numTempGroups+1? realkbT: drudekbT;
        if (etaMass[itg][0]>0)
            etaDotDot[itg][0] = (kineticEnergiesVec[itg] - tempGroupNkbT[itg]) / etaMass[itg][0];
        for (int iloop = 0; iloop < numLoopsPerStep; iloop++) {
            if (itg < numTempGroups+1 || integrator.getUseDrudeNHChains()) {
                for (int i = integrator.getNumNHChains() - 1; i > 0; i--) {
                    expfac[itg] = exp(-dtc8 * etaDot[itg][i + 1]);
                    etaDot[itg][i] *= expfac[itg];
                    etaDot[itg][i] += etaDotDot[itg][i] * dtc4;
                    etaDot[itg][i] *= expfac[itg];
                }
            }
            expfac[itg] = exp(-dtc8 * etaDot[itg][1]);
            etaDot[itg][0] *= expfac[itg];
            etaDot[itg][0] += etaDotDot[itg][0] * dtc4;
            etaDot[itg][0] *= expfac[itg];

            vscaleFactorsVec[itg] *= exp(-dtc2 * etaDot[itg][0]);
            kineticEnergiesVec[itg] *= exp(-dtc * etaDot[itg][0]);
            if (itg < numTempGroups +1 || integrator.getUseDrudeNHChains()) {
                for (int i = 0; i < integrator.getNumNHChains(); i++) {
                    eta[itg][i] += dtc2 * etaDot[itg][i];
                }
            }

            if (etaMass[itg][0]>0)
                etaDotDot[itg][0] = (kineticEnergiesVec[itg] - tempGroupNkbT[itg]) / etaMass[itg][0];

            etaDot[itg][0] *= expfac[itg];
            etaDot[itg][0] += etaDotDot[itg][0] * dtc4;
            etaDot[itg][0] *= expfac[itg];
            if (itg < numTempGroups +1 || integrator.getUseDrudeNHChains()) {
                for (int i = 1; i < integrator.getNumNHChains(); i++) {
                    expfac[itg] = exp(-dtc8 * etaDot[itg][i + 1]);
                    etaDot[itg][i] *= expfac[itg];
                    etaDotDot[itg][i] = (etaMass[itg][i-1] * etaDot[itg][i-1] * etaDot[itg][i-1] - kbT) / etaMass[itg][i];
                    etaDot[itg][i] += etaDotDot[itg][i] * dtc4;
                    etaDot[itg][i] *= expfac[itg];
                }
            }
        }
    }

//    std::cout << cu.getStepCount() << " NH Velocity scaling factors: ";
//    for (auto scale: vscaleFactorsVec) {
//        std::cout<< scale << "; ";
//    }
//    std::cout<< "\n";
}

void CudaIntegrateVVStepKernel::scaleVelocity(ContextImpl& context) {
    cu.setAsCurrent();

    vscaleFactors->upload(vscaleFactorsVec);
    void *argsChain[] = {&cu.getVelm().getDevicePointer(),
                         &normVelm->getDevicePointer(),
                         &normalParticlesNH->getDevicePointer(),
                         &pairParticlesNH->getDevicePointer(),
                         &particleTempGroup->getDevicePointer(),
                         &vscaleFactors->getDevicePointer()};
    cu.executeKernel(kernelScale, argsChain, particlesNHVec.size());
}

void CudaIntegrateVVStepKernel::firstIntegrate(ContextImpl& context, const VVIntegrator& integrator, bool& forcesAreValid) {
    cu.setAsCurrent();
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();

    // Compute integrator coefficients.

    double stepSize = integrator.getStepSize();
    double fscale = 0.5*stepSize/(double) 0x100000000;
    double maxDrudeDistance = integrator.getMaxDrudeDistance();
    double hardwallScaleDrude = sqrt(BOLTZ * integrator.getDrudeTemperature());
    if (stepSize != prevStepSize) {
        if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
            double2 ss = make_double2(0, stepSize);
            integration.getStepSize().upload(&ss);
        }
        else {
            float2 ss = make_float2(0, (float) stepSize);
            integration.getStepSize().upload(&ss);
        }
        prevStepSize = stepSize;
    }

    // Create appropriate pointer for the precision mode.

    float fscaleFloat = (float) fscale;
    float maxDrudeDistanceFloat =(float) maxDrudeDistance;
    float hardwallScaleDrudeFloat = (float) hardwallScaleDrude;
    void *fscalePtr, *maxDrudeDistancePtr, *hardwallScaleDrudePtr;
    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        fscalePtr = &fscale;
        maxDrudeDistancePtr = &maxDrudeDistance;
        hardwallScaleDrudePtr = &hardwallScaleDrude;
    }
    else {
        fscalePtr = &fscaleFloat;
        maxDrudeDistancePtr = &maxDrudeDistanceFloat;
        hardwallScaleDrudePtr = &hardwallScaleDrudeFloat;
    }

    // Call the first half of velocity integration kernel. (both thermostat and actual velocity update)
    bool updatePosDelta = true;
    void *updatePosDeltaPtr = &updatePosDelta;
    void *argsVel[] = {&cu.getVelm().getDevicePointer(),
                       &cu.getForce().getDevicePointer(),
                       &forceExtra->getDevicePointer(),
                       &integration.getPosDelta().getDevicePointer(),
                       &particlesNH->getDevicePointer(),
                       &integration.getStepSize().getDevicePointer(),
                       fscalePtr,
                       updatePosDeltaPtr};
    cu.executeKernel(kernelVel, argsVel, numAtoms);

    // Apply position constraints.
    integration.applyConstraints(integrator.getConstraintTolerance());

    // Call the position integration kernel.
    CUdeviceptr posCorrection = (cu.getUseMixedPrecision() ? cu.getPosqCorrection().getDevicePointer() : 0);
    void *argsPos[] = {&cu.getPosq().getDevicePointer(),
                       &posCorrection,
                       &integration.getPosDelta().getDevicePointer(),
                       &cu.getVelm().getDevicePointer(),
                       &integration.getStepSize().getDevicePointer(),
                       &particlesNH->getDevicePointer()};
    cu.executeKernel(kernelPos, argsPos, numAtoms);

    // Apply hard wall constraints.
    if (maxDrudeDistance > 0 and allPairs->getSize() > 0) {
        void *hardwallArgs[] = {&cu.getPosq().getDevicePointer(),
                                &posCorrection,
                                &cu.getVelm().getDevicePointer(),
                                &allPairs->getDevicePointer(),
                                &integration.getStepSize().getDevicePointer(),
                                maxDrudeDistancePtr,
                                hardwallScaleDrudePtr};
        cu.executeKernel(hardwallKernel, hardwallArgs, allPairs->getSize());
    }

    integration.computeVirtualSites();

    /** gongzheng @ 2020-02-20
     *  reorder atoms after first half integration instead of the end of the step
     *  so that the atomIndex of Langevin Forces are correct at next step
     */

    cu.reorderAtoms();
    if (cu.getAtomsWereReordered()) {
        forcesAreValid = false;
    }
}
void CudaIntegrateVVStepKernel::resetExtraForce(ContextImpl& context, const VVIntegrator& integrator) {
    cu.setAsCurrent();
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();

    void *args1[] = {&forceExtra->getDevicePointer()};
    cu.executeKernel(kernelResetExtraForce, args1, numAtoms);
}

void CudaIntegrateVVStepKernel::calcLangevinForce(ContextImpl& context, const VVIntegrator& integrator) {
    cu.setAsCurrent();
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();

    // Compute integrator coefficients.

    double stepSize = integrator.getStepSize();
    double dragFactor = integrator.getFriction(); // * mass
    double randFactor = sqrt(2.0 * BOLTZ *integrator.getTemperature() * dragFactor/ stepSize); // * sqrt(mass)
    double dragFactorDrude = integrator.getDrudeFriction(); // * mass
    double randFactorDrude = sqrt(2.0 * BOLTZ *integrator.getDrudeTemperature() * dragFactorDrude/ stepSize); // * sqrt(mass)

    // Create appropriate pointer for the precision mode.

    float dragFactorFloat = (float) dragFactor;
    float randFactorFloat = (float) randFactor;
    float dragFactorDrudeFloat = (float) dragFactorDrude;
    float randFactorDrudeFloat = (float) randFactorDrude;
    void *dragPtr, *randPtr, *dragDrudePtr, *randDrudePtr;
    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        dragPtr = &dragFactor;
        randPtr = &randFactor;
        dragDrudePtr = &dragFactorDrude;
        randDrudePtr = &randFactorDrude;
    }
    else {
        dragPtr = &dragFactorFloat;
        randPtr = &randFactorFloat;
        dragDrudePtr = &dragFactorDrudeFloat;
        randDrudePtr = &randFactorDrudeFloat;
    }

    // Call the Langevin force kernel

    int randomIndex = integration.prepareRandomNumbers(normalParticlesLD->getSize() + 2 * pairParticlesLD->getSize());
    void *args1[] = {&cu.getVelm().getDevicePointer(),
                     &forceExtra->getDevicePointer(),
                     &normalParticlesLD->getDevicePointer(),
                     &pairParticlesLD->getDevicePointer(),
                     dragPtr, randPtr, dragDrudePtr, randDrudePtr,
                     &integration.getRandom().getDevicePointer(),
                     &randomIndex};
    cu.executeKernel(kernelApplyLangevin, args1, integrator.getParticlesLD().size());
}

void CudaIntegrateVVStepKernel::calcElectricFieldForce(ContextImpl& context, const VVIntegrator& integrator) {
    cu.setAsCurrent();
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();

    double efield = integrator.getElectricField(); // kJ/nm.e
    double fscale = AVOGADRO;  // convert from kJ/nm.e to kJ/mol.nm.e
    float efieldFloat = (float) efield;
    float fscaleFloat = (float) fscale;
    void *efieldPtr, *fscalePtr;
    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        efieldPtr = &efield;
        fscalePtr = &fscale;
    } else {
        efieldPtr = &efieldFloat;
        fscalePtr = &fscaleFloat;
    }

    void *args1[] = {&cu.getPosq().getDevicePointer(),
                     &forceExtra->getDevicePointer(),
                     &particlesElectrolyte->getDevicePointer(),
                     efieldPtr,
                     fscalePtr};
    cu.executeKernel(kernelApplyElectricField, args1, particlesElectrolyte->getSize());
}

void CudaIntegrateVVStepKernel::calcPeriodicPerturbationForce(ContextImpl& context, const VVIntegrator& integrator) {
    cu.setAsCurrent();
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();

    double acceleration = integrator.getCosAcceleration(); //
    float accelerationFloat = (float) acceleration;
    void *accelerationPtr;
    if (cu.getUseDoublePrecision()) {
        accelerationPtr = &acceleration;
    } else {
        accelerationPtr = &accelerationFloat;
    }

    void *args1[] = {&cu.getPosq().getDevicePointer(),
                     &cu.getVelm().getDevicePointer(),
                     &forceExtra->getDevicePointer(),
                     accelerationPtr,
                     cu.getInvPeriodicBoxSizePointer()};
    cu.executeKernel(kernelCosAccelerate, args1, numAtoms);
}

void CudaIntegrateVVStepKernel::calcPeriodicVelocityBias(ContextImpl& context, const VVIntegrator& integrator) {
    cu.setAsCurrent();
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();

    void *args1[] = {&cu.getPosq().getDevicePointer(),
                     &cu.getVelm().getDevicePointer(),
                     &VBuffer->getDevicePointer(),
                     cu.getInvPeriodicBoxSizePointer()};
    cu.executeKernel(kernelCalcBias, args1, numAtoms);

    int bufferSize = VBuffer->getSize();
    // Use only one threadBlock for this kernel because we use shared memory
    void *args2[] = {&VBuffer->getDevicePointer(),
                     &invMassTotal,
                     &bufferSize};
    cu.executeKernel(kernelSumV, args2, cu.ThreadBlockSize, cu.ThreadBlockSize,
                     cu.ThreadBlockSize * VBuffer->getElementSize());
}

void CudaIntegrateVVStepKernel::calcViscosity(ContextImpl& context, const VVIntegrator& integrator, double& vMax, double& invVis) {
    cu.setAsCurrent();
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();

    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        auto VBufferVec = std::vector<double>(numAtoms, 0);
        VBuffer->download(VBufferVec);
        vMax = VBufferVec[0];
    } else {
        auto VBufferVec = std::vector<float>(numAtoms, 0);
        VBuffer->download(VBufferVec);
        vMax = (double) VBufferVec[0];
    }

    double4 box = cu.getPeriodicBoxSize();
    double vol = box.x * box.y * box.z;

    invVis = vMax * vol * invMassTotal / integrator.getCosAcceleration()
             * (2 * 3.1415926 / box.z) * (2 * 3.1415926 / box.z);
}

void CudaIntegrateVVStepKernel::removePeriodicVelocityBias(ContextImpl& context, const VVIntegrator& integrator) {
    cu.setAsCurrent();
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();

    void *args1[] = {&cu.getPosq().getDevicePointer(),
                     &cu.getVelm().getDevicePointer(),
                     &VBuffer->getDevicePointer(),
                     cu.getInvPeriodicBoxSizePointer()};
    cu.executeKernel(kernelRemoveBias, args1, numAtoms);
}

void CudaIntegrateVVStepKernel::restorePeriodicVelocityBias(ContextImpl& context, const VVIntegrator& integrator) {
    cu.setAsCurrent();
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();

    void *args1[] = {&cu.getPosq().getDevicePointer(),
                     &cu.getVelm().getDevicePointer(),
                     &VBuffer->getDevicePointer(),
                     cu.getInvPeriodicBoxSizePointer()};
    cu.executeKernel(kernelRestoreBias, args1, numAtoms);
}

void CudaIntegrateVVStepKernel::secondIntegrate(ContextImpl &context,
                                                const VVIntegrator &integrator,
                                                bool &forcesAreValid) {
    cu.setAsCurrent();
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();

    // Compute integrator coefficients.

    double stepSize = integrator.getStepSize();
    double fscale = 0.5 * stepSize / (double) 0x100000000;
    if (stepSize != prevStepSize) {
        if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
            double2 ss = make_double2(0, stepSize);
            integration.getStepSize().upload(&ss);
        } else {
            float2 ss = make_float2(0, (float) stepSize);
            integration.getStepSize().upload(&ss);
        }
        prevStepSize = stepSize;
    }

    // Create appropriate pointer for the precision mode.

    float fscaleFloat = (float) fscale;
    void *fscalePtr;
    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        fscalePtr = &fscale;
    } else {
        fscalePtr = &fscaleFloat;
    }

    // Call the second half of velocity integration kernel. (actual velocity update only)
    bool updatePosDelta = false;
    void *updatePosDeltaPtr = &updatePosDelta;
    void *argsVel2[] = {&cu.getVelm().getDevicePointer(),
                        &cu.getForce().getDevicePointer(),
                        &forceExtra->getDevicePointer(),
                        &integration.getPosDelta().getDevicePointer(),
                        &particlesNH->getDevicePointer(),
                        &integration.getStepSize().getDevicePointer(),
                        fscalePtr,
                        updatePosDeltaPtr};
    cu.executeKernel(kernelVel, argsVel2, numAtoms);

    // Apply velocity constraints
    integration.applyVelocityConstraints(integrator.getConstraintTolerance());

    // Update the time and step count.
    cu.setTime(cu.getTime()+stepSize);
    cu.setStepCount(cu.getStepCount()+1);

//    // check temperature
//    if (cu.getStepCount() % 1 == 0) {
//        std::cout << "Step " << cu.getStepCount() << " Temperature: ";
//        for (int i = 0; i < integrator.getNumTempGroups() + 2; i++) {
//            double T = kineticEnergiesVec[i] / tempGroupDof[i] / BOLTZ;
//            std::cout << T << " ";
//        }
//        std::cout << "\n";
//    }
}

double CudaIntegrateVVStepKernel::computeKineticEnergy(ContextImpl& context, const VVIntegrator& integrator) {
    return cu.getIntegrationUtilities().computeKineticEnergy(0);
}

CudaModifyImageChargeKernel::~CudaModifyImageChargeKernel() {
    if (imagePairs != NULL)
        delete imagePairs;
}

void CudaModifyImageChargeKernel::initialize(const System& system, const VVIntegrator& integrator) {
    cu.getPlatformData().initializeContexts(system);

    // Identify particle pairs and ordinary particles.

    for (auto pair: integrator.getImagePairs())
        imagePairsVec.push_back(make_int2(pair.first, pair.second));
    imagePairs = CudaArray::create<int2>(cu, max((int) imagePairsVec.size(), 1), "imagePairs");
    if (!imagePairsVec.empty())
        imagePairs->upload(imagePairsVec);

    map<string, string> defines;
    defines["NUM_IMAGES"] = cu.intToString(imagePairsVec.size());
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    CUmodule module = cu.createModule(CudaVVKernelSources::vectorOps+CudaVVKernelSources::imageChargeModifier, defines, "");
    kernelImage = cu.getKernel(module, "updateImagePositions");

    cout << "CUDA modules for ImageChargeModifier are created\n" << flush;
}

void CudaModifyImageChargeKernel::updateImagePositions(ContextImpl& context, const VVIntegrator& integrator) {
    cu.setAsCurrent();

    /**
     * Considering fix the posCellOffset of images particles  so that visualization wil be prettier
     * Since there is no API to update posCellOffsets in context,
     * we have to get periodicBoxSize whenever the box changes,
     * which is inefficient
     * So we just ignore the cellOffsets because it's not really necessary for correct simulation
     */

    double mirrorLocation = integrator.getMirrorLocation();
    float mirrorLocationFloat = (float) mirrorLocation;
    void *mirrorPtr;
    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        mirrorPtr = &mirrorLocation;
    }
    else {
        mirrorPtr = &mirrorLocationFloat;
    }

    CUdeviceptr posCorrection = (cu.getUseMixedPrecision() ? cu.getPosqCorrection().getDevicePointer() : 0);
    void *args2[] = {&cu.getPosq().getDevicePointer(),
                     &posCorrection,
                     &imagePairs->getDevicePointer(),
                     mirrorPtr};
    cu.executeKernel(kernelImage, args2, imagePairsVec.size());
}
