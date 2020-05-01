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

enum{TG_ATOM, TG_COM, TG_DRUDE, NUM_TG_MAX};

CudaIntegrateVVStepKernel::~CudaIntegrateVVStepKernel() {
    delete forceExtra;
    delete drudePairs;
}

void CudaIntegrateVVStepKernel::initialize(const System& system, const VVIntegrator& integrator, const DrudeForce* force) {
    if (integrator.getDebugEnabled())
        cout << "Initializing CudaVVIntegrator...\n" << flush;

    cu.getPlatformData().initializeContexts(system);
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();
    cu.getIntegrationUtilities().initRandomNumberGenerator((unsigned int) integrator.getRandomNumberSeed());

    numAtoms = cu.getNumAtoms();

    if (force != NULL) {
        for (int i = 0; i < force->getNumParticles(); i++) {
            int p, p1, p2, p3, p4;
            double charge, polarizability, aniso12, aniso34;
            force->getParticleParameters(i, p, p1, p2, p3, p4, charge, polarizability, aniso12, aniso34);
            drudePairsVec.push_back(make_int2(p, p1));
        }
    }
    drudePairs = CudaArray::create<int2>(cu, max((int) drudePairsVec.size(), 1), "vvDrudePairs");
    if (!drudePairsVec.empty())
        drudePairs->upload(drudePairsVec);

    // init forceExtra
    if (cu.getUseDoublePrecision()) {
        forceExtra = CudaArray::create<double3>(cu, numAtoms, "vvForceExtra");
        auto forceExtraVec = std::vector<double3>(numAtoms, make_double3(0, 0, 0));
        forceExtra->upload(forceExtraVec);
    }
    else {
        forceExtra = CudaArray::create<float3>(cu, numAtoms, "vvForceExtra");
        auto forceExtraVec = std::vector<float3>(numAtoms, make_float3(0, 0, 0));
        forceExtra->upload(forceExtraVec);
    }

    map<string, string> defines;
    defines["NUM_ATOMS"] = cu.intToString(numAtoms);
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    defines["NUM_DRUDE_PAIRS"] = cu.intToString(drudePairsVec.size());
    CUmodule module = cu.createModule(CudaVVKernelSources::vectorOps + CudaVVKernelSources::velocityVerlet, defines, "");
    kernelVel = cu.getKernel(module, "velocityVerletIntegrateVelocities");
    kernelPos = cu.getKernel(module, "velocityVerletIntegratePositions");
    kernelResetExtraForce = cu.getKernel(module, "resetExtraForce");
    if (force != NULL and integrator.getMaxDrudeDistance() > 0)
        kernelDrudeHardwall = cu.getKernel(module, "applyHardWallConstraints");

    cout << "CUDA modules for velocity-Verlet integrator are created\n"
         << "    NUM_ATOMS: " << numAtoms << ", PADDED_NUM_ATOMS: " << cu.getPaddedNumAtoms() << "\n"
         << "    Num Drude pairs: " << drudePairsVec.size() << ", Drude hardwall distance: " << integrator.getMaxDrudeDistance() << " nm\n"
         << "    Num thread blocks: " << cu.getNumThreadBlocks() << ", Thread block size: " << cu.ThreadBlockSize << "\n";

    prevStepSize = -1.0;
}

void CudaIntegrateVVStepKernel::firstIntegrate(ContextImpl& context, const VVIntegrator& integrator, bool& forcesAreValid) {
    if (integrator.getDebugEnabled())
        cout << "VVIntegrator first-half integration\n" << flush;

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
                       &integration.getStepSize().getDevicePointer()};
    cu.executeKernel(kernelPos, argsPos, numAtoms);

    // Apply hard wall constraints.
    if (maxDrudeDistance > 0 and !drudePairsVec.empty()) {
        void *hardwallArgs[] = {&cu.getPosq().getDevicePointer(),
                                &posCorrection,
                                &cu.getVelm().getDevicePointer(),
                                &drudePairs->getDevicePointer(),
                                &integration.getStepSize().getDevicePointer(),
                                maxDrudeDistancePtr,
                                hardwallScaleDrudePtr};
        cu.executeKernel(kernelDrudeHardwall, hardwallArgs, drudePairs->getSize());
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
    if (integrator.getDebugEnabled())
        cout << "VVIntegrator reset extra force\n" << flush;

    cu.setAsCurrent();
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();

    void *args1[] = {&forceExtra->getDevicePointer()};
    cu.executeKernel(kernelResetExtraForce, args1, numAtoms);
}

void CudaIntegrateVVStepKernel::secondIntegrate(ContextImpl &context, const VVIntegrator &integrator, bool &forcesAreValid) {
    if (integrator.getDebugEnabled())
        cout << "VVIntegrator second-half integration\n" << flush;

    cu.setAsCurrent();
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();

    // Create appropriate pointer for the precision mode.

    double stepSize = integrator.getStepSize();
    double fscale = 0.5 * stepSize / (double) 0x100000000;
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

CudaModifyDrudeNoseKernel::~CudaModifyDrudeNoseKernel() {
    delete particlesNH;
    delete moleculesNH;
    delete normalParticlesNH;
    delete pairParticlesNH;
    delete particleMolId;
    delete particlesInMolecules;
    delete particlesSortedByMolId;
    delete comVelm;
    delete kineticEnergyBufferNH;
    delete kineticEnergiesNH;
    delete vscaleFactorsNH;
}

void CudaModifyDrudeNoseKernel::initialize(const System &system, const VVIntegrator &integrator, const DrudeForce* force) {
    if (integrator.getDebugEnabled())
        cout << "Initializing CudaModifyDrudeNoseKernel...\n" << flush;

    cu.getPlatformData().initializeContexts(system);
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();

    numAtoms = cu.getNumAtoms();
    particlesNHVec = integrator.getParticlesNH();
    moleculesNHVec = integrator.getMoleculesNH();
    tempGroupDof = std::vector<double>(NUM_TG_MAX, 0.0);

    /**
     * Atomic motion is the first temperature group
     * Molecular COM motion is after
     * Drude relative motion is the last
     * particlesInMoleculeVec = pair(numOfParticlesInMolecule, indexOfFirstParticleInMolecule)
     * particlesSortedByMolIdVec records the indexes of particles sorted by molecule id
     * so that even when molecules are not successive, it still works
     */

    // Identify particles, pairs and residues

    int id_start = 0;
    for (int id_mol =0; id_mol < integrator.getNumMolecules(); id_mol++){
        int n_particles_in_mol = 0;
        for (int i = 0; i < system.getNumParticles(); i++) {
            if (integrator.getParticleMolId(i) == id_mol){
                n_particles_in_mol ++;
                particlesSortedByMolIdVec.push_back(i);
            }
        }
        particlesInMoleculesVec.push_back(make_int2(n_particles_in_mol, id_start));
        id_start += n_particles_in_mol;
    }

    set<int> particlesNHSet;
    for (int i = 0; i < system.getNumParticles(); i++) {
        if (integrator.isParticleNH(i))
            particlesNHSet.insert(i);
        int id_mol = integrator.getParticleMolId(i);
        particleMolIdVec.push_back(id_mol);
        double mass = system.getParticleMass(i);
        double molInvMass = integrator.getMoleculeInvMass(id_mol);

        if (integrator.isParticleNH(i) && mass != 0.0) {
            tempGroupDof[TG_ATOM] += 3;
            if (integrator.getUseCOMTempGroup()) {
                tempGroupDof[TG_ATOM] -= 3 * mass * molInvMass;
            }
        }
    }

    if (force != NULL){
        for (int i = 0; i < force->getNumParticles(); i++) {
            int p, p1, p2, p3, p4;
            double charge, polarizability, aniso12, aniso34;
            force->getParticleParameters(i, p, p1, p2, p3, p4, charge, polarizability, aniso12, aniso34);
            if (integrator.isParticleNH(p) != integrator.isParticleNH(p1))
                throw OpenMMException("Drude particle and its parent atom should be in the same thermostat");
            if (integrator.isParticleNH(p)){
                particlesNHSet.erase(p);
                particlesNHSet.erase(p1);
                pairParticlesNHVec.push_back(make_int2(p, p1));
                tempGroupDof[TG_ATOM] -= 3;
                tempGroupDof[TG_DRUDE] += 3;
            }
        }
    }
    normalParticlesNHVec.insert(normalParticlesNHVec.begin(), particlesNHSet.begin(), particlesNHSet.end());

    // Subtract constraint DOFs from internal motions
    for (int i = 0; i < system.getNumConstraints(); i++) {
        int p, p1;
        double distance;
        system.getConstraintParameters(i, p, p1, distance);
        if (integrator.isParticleNH(p) != integrator.isParticleNH(p1))
            throw OpenMMException("Constrained particle pair should be in the same thermostat");
        if (integrator.isParticleNH(p)) {
            tempGroupDof[TG_ATOM] -= 1;
        }
    }
    /**
     * 3 DOFs should be subtracted if CMMotionRemover presents
     * if useCOMTempGroup, subtract it from molecular motion
     * otherwise, subtract it from first temperature group
     */
    if (integrator.getUseCOMTempGroup()) {
        tempGroupDof[TG_COM] = 3 * moleculesNHVec.size();
    }
    for (int i = 0; i < system.getNumForces(); i++) {
        if (typeid(system.getForce(i)) == typeid(CMMotionRemover)) {
            if (integrator.getUseCOMTempGroup())
                tempGroupDof[TG_COM] -= 3;
            else
                tempGroupDof[TG_ATOM] -= 3;
            break;
        }
    }
    /**
     *  set DOF to zero if it's negative
     *  just in case, though i cannot image when this would happen
     */
    for (int i = 0; i < NUM_TG_MAX; i++)
        tempGroupDof[i] = max(tempGroupDof[i], (double) 0);

    // determine how many temperature groups we need
    numTempGroup = 3;
    if (tempGroupDof[TG_DRUDE] == 0){
        numTempGroup = 2;
        if (tempGroupDof[TG_COM] == 0){
            numTempGroup = 1;
        }
    }

    // Initialize NH chain particles

    int numNHChains = integrator.getNumNHChains();
    etaMass = std::vector<vector<double> >(numTempGroup, std::vector<double>(numNHChains, 0.0));
    eta = std::vector<vector<double> >(numTempGroup, std::vector<double>(numNHChains, 0.0));
    etaDot = std::vector<vector<double> >(numTempGroup, std::vector<double>(numNHChains + 1, 0.0));
    etaDotDot = std::vector<vector<double> >(numTempGroup, std::vector<double>(numNHChains, 0.0));

    realKbT = BOLTZ * integrator.getTemperature();
    drudeKbT = BOLTZ * integrator.getDrudeTemperature();
    for (int i = 0; i < numTempGroup; i++) {
        double tgKbT = i == TG_DRUDE ? drudeKbT : realKbT;
        double tgMass = i == TG_DRUDE ?
                        drudeKbT / pow(integrator.getDrudeFrequency(), 2) :
                        realKbT / pow(integrator.getFrequency(), 2);
        tempGroupNkbT.push_back(tempGroupDof[i] * tgKbT);
        etaMass[i][0] = tempGroupDof[i] * tgMass;
        for (int ich=1; ich < integrator.getNumNHChains(); ich++)
            etaMass[i][ich] = tgMass;
    }

    // Initialize CudaArray
    particlesNH = CudaArray::create<int>(cu, (int) particlesNHVec.size(), "particlesNH");
    moleculesNH = CudaArray::create<int>(cu, (int) moleculesNHVec.size(), "moleculesNH");
    normalParticlesNH = CudaArray::create<int>(cu, max((int) normalParticlesNHVec.size(), 1), "normalParticlesNH");
    pairParticlesNH = CudaArray::create<int2>(cu, max((int) pairParticlesNHVec.size(), 1), "pairParticlesNH");
    particleMolId = CudaArray::create<int>(cu, (int) particleMolIdVec.size(), "particleMolId");
    particlesInMolecules = CudaArray::create<int2>(cu, (int) particlesInMoleculesVec.size(), "particlesInMolecules");
    particlesSortedByMolId = CudaArray::create<int>(cu, (int) particlesSortedByMolIdVec.size(), "particlesSortedByMolId");
    kineticEnergyBufferNH = CudaArray::create<double>(cu, (int) particlesNHVec.size() * numTempGroup, "kineticEnergyBufferNH");
    kineticEnergiesNH = CudaArray::create<double>(cu, numTempGroup, "kineticEnergiesNH");
    vscaleFactorsNH = CudaArray::create<double>(cu, numTempGroup, "vscaleFactorsNH");

    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        comVelm = CudaArray::create<double4>(cu, integrator.getNumMolecules(), "comVelm");
        auto vec = std::vector<double4>(integrator.getNumMolecules(), make_double4(0, 0, 0, 0));
        comVelm->upload(vec);
    }
    else {
        comVelm = CudaArray::create<float4>(cu, integrator.getNumMolecules(), "comVelm");
        auto vec = std::vector<float4>(integrator.getNumMolecules(), make_float4(0, 0, 0, 0));
        comVelm->upload(vec);
    }

    if (!particlesNHVec.empty())
        particlesNH->upload(particlesNHVec);
    if (!moleculesNHVec.empty())
        moleculesNH->upload(moleculesNHVec);
    if (!normalParticlesNHVec.empty())
        normalParticlesNH->upload(normalParticlesNHVec);
    if (!pairParticlesNHVec.empty())
        pairParticlesNH->upload(pairParticlesNHVec);
    if (!particleMolIdVec.empty())
        particleMolId->upload(particleMolIdVec);
    if (!particlesInMoleculesVec.empty())
        particlesInMolecules->upload(particlesInMoleculesVec);
    if (!particlesSortedByMolIdVec.empty())
        particlesSortedByMolId->upload(particlesSortedByMolIdVec);

    // Create kernels.
    map<string, string> defines;
    defines["NUM_PARTICLES_NH"] = cu.intToString(particlesNHVec.size());
    defines["NUM_MOLECULES_NH"] = cu.intToString(moleculesNHVec.size());
    defines["NUM_NORMAL_PARTICLES_NH"] = cu.intToString(normalParticlesNHVec.size());
    defines["NUM_PAIRS_NH"] = cu.intToString(pairParticlesNHVec.size());
    defines["NUM_TG"] = cu.intToString(numTempGroup);
    defines["TG_ATOM"] = cu.intToString(TG_ATOM);
    defines["TG_COM"] = cu.intToString(TG_COM);
    defines["TG_DRUDE"] = cu.intToString(TG_DRUDE);

    CUmodule module = cu.createModule(CudaVVKernelSources::vectorOps + CudaVVKernelSources::drudeNoseHoover, defines, "");
    kernelCOMVel = cu.getKernel(module, "calcCOMVelocities");
    kernelNormVel = cu.getKernel(module, "normalizeVelocities");
    kernelKE = cu.getKernel(module, "computeNormalizedKineticEnergies");
    kernelKESum = cu.getKernel(module, "sumNormalizedKineticEnergies");
    kernelScale = cu.getKernel(module, "scaleVelocity");

    cout << "CUDA modules for Nose-Hoover thermostat are created\n"
         << "    Num molecules in NH thermostat: " << moleculesNHVec.size() << " / " << integrator.getNumMolecules() << "\n"
         << "    Num normal particles: " << normalParticlesNHVec.size() << ", Num Drude pairs: " << pairParticlesNHVec.size() << "\n"
         << "    Real T: " << integrator.getTemperature() << " K, Drude T: " << integrator.getDrudeTemperature() << " K\n"
         << "    Real coupling frequency: " << integrator.getFrequency() << " /ps, Drude coupling frequency: " << integrator.getDrudeFrequency() << " /ps\n"
         << "    Num NH chain: " << integrator.getNumNHChains() << ", Loops per NH step: " << integrator.getLoopsPerStep() << "\n"
         << "    Use COM temperature group: " << integrator.getUseCOMTempGroup() << "\n";
    for (int i = 0; i < numTempGroup; i++) {
        cout << "    DOF[" << i << "]: " << tempGroupDof[i] << ", NkbT[" << i << "]: " << tempGroupNkbT[i] << ", etaMass[" << i << "]: " << etaMass[i][0] << "\n";
    }
}


void CudaModifyDrudeNoseKernel::scaleVelocity(ContextImpl& context, const VVIntegrator& integrator) {
    if (integrator.getDebugEnabled())
        cout << "DrudeNoseModifier scale velocity\n" << flush;

    cu.setAsCurrent();

    if (integrator.getUseCOMTempGroup()){
        void *argsCOMVel[] = {&cu.getVelm().getDevicePointer(),
                              &comVelm->getDevicePointer(),
                              &particlesInMolecules->getDevicePointer(),
                              &particlesSortedByMolId->getDevicePointer(),
                              &moleculesNH->getDevicePointer()};
        cu.executeKernel(kernelCOMVel, argsCOMVel, moleculesNHVec.size());

        void *argsNormVel[] = {&cu.getVelm().getDevicePointer(),
                               &comVelm->getDevicePointer(),
                               &particleMolId->getDevicePointer(),
                               &particlesNH->getDevicePointer()};
        cu.executeKernel(kernelNormVel, argsNormVel, particlesNHVec.size());
    }

    int bufferSize = kineticEnergyBufferNH->getSize();
    void *argsKE[] = {&cu.getVelm().getDevicePointer(),
                      &comVelm->getDevicePointer(),
                      &normalParticlesNH->getDevicePointer(),
                      &pairParticlesNH->getDevicePointer(),
                      &kineticEnergyBufferNH->getDevicePointer(),
                      &moleculesNH->getDevicePointer(),
                      &bufferSize};
    cu.executeKernel(kernelKE, argsKE, particlesNHVec.size());

    // Use only one threadBlock for this kernel because we use shared memory
    int workGroupSize = 512;
    void *argsKESum[] = {&kineticEnergyBufferNH->getDevicePointer(),
                         &kineticEnergiesNH->getDevicePointer(),
                         &bufferSize};
    cu.executeKernel(kernelKESum, argsKESum, workGroupSize, workGroupSize,
                     workGroupSize * numTempGroup * kineticEnergyBufferNH->getElementSize());

    kineticEnergiesNHVec = std::vector<double>(numTempGroup);
    kineticEnergiesNH->download(kineticEnergiesNHVec);

//    std::cout << cu.getStepCount() << " NH group kinetic energies: ";
//    for (auto ke: kineticEnergiesNHVec) {
//        std::cout<< ke / 2 << "; ";
//    }
//    std::cout<< "\n";


    // Calculate scaling factor for velocities for each temperature group using Nose-Hoover chain
    vscaleFactorsNHVec = std::vector<double>(numTempGroup, 1.0);
    for (int itg = 0; itg < numTempGroup; itg++) {
        const double T = itg == TG_DRUDE ? integrator.getDrudeTemperature() : integrator.getTemperature();
        if (etaMass[itg][0] > 0)
            integrator.propagateNHChain(eta[itg], etaDot[itg], etaDotDot[itg], etaMass[itg],
                                        kineticEnergiesNHVec[itg], tempGroupNkbT[itg], T,
                                        vscaleFactorsNHVec[itg]);
    }

//    std::cout << cu.getStepCount() << " NH Velocity scaling factors: ";
//    for (auto scale: vscaleFactorsNHVec) {
//        std::cout<< scale << "; ";
//    }
//    std::cout<< "\n";

    vscaleFactorsNH->upload(vscaleFactorsNHVec);
    void *argsChain[] = {&cu.getVelm().getDevicePointer(),
                         &comVelm->getDevicePointer(),
                         &particleMolId->getDevicePointer(),
                         &normalParticlesNH->getDevicePointer(),
                         &pairParticlesNH->getDevicePointer(),
                         &vscaleFactorsNH->getDevicePointer()};
    cu.executeKernel(kernelScale, argsChain, particlesNHVec.size());
}

CudaModifyDrudeLangevinKernel::~CudaModifyDrudeLangevinKernel() {
    delete normalParticlesLD;
    delete pairParticlesLD;
}

void CudaModifyDrudeLangevinKernel::initialize(const System &system, const VVIntegrator &integrator, const DrudeForce* force, Kernel& vvKernel) {
    if (integrator.getDebugEnabled())
        cout << "Initializing CudaModifyDrudeLangevinKernel...\n" << flush;

    vvStepKernel = &vvKernel.getAs<CudaIntegrateVVStepKernel>();
    cu.getPlatformData().initializeContexts(system);
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();
    cu.getIntegrationUtilities().initRandomNumberGenerator((unsigned int) integrator.getRandomNumberSeed());

    set<int> particlesLDSet;
    for (int i = 0; i < system.getNumParticles(); i++) {
        if (integrator.isParticleLD(i))
            particlesLDSet.insert(i);
    }

    if (force != NULL){
        for (int i = 0; i < force->getNumParticles(); i++) {
            int p, p1, p2, p3, p4;
            double charge, polarizability, aniso12, aniso34;
            force->getParticleParameters(i, p, p1, p2, p3, p4, charge, polarizability, aniso12, aniso34);
            if (integrator.isParticleLD(p) != integrator.isParticleLD(p1))
                throw OpenMMException("Drude particle and its parent atom should be in the same thermostat");
            if (integrator.isParticleLD(p)){
                particlesLDSet.erase(p);
                particlesLDSet.erase(p1);
                pairParticlesLDVec.push_back(make_int2(p, p1));
            }
        }
    }

    for (int i = 0; i < system.getNumConstraints(); i++) {
        int p, p1;
        double distance;
        system.getConstraintParameters(i, p, p1, distance);
        if (integrator.isParticleLD(p) != integrator.isParticleLD(p1))
            throw OpenMMException("Constrained particle pair should be in the same thermostat");
    }

    normalParticlesLDVec.insert(normalParticlesLDVec.begin(), particlesLDSet.begin(), particlesLDSet.end());

    normalParticlesLD = CudaArray::create<int>(cu, max((int) normalParticlesLDVec.size(), 1), "normalParticlesLD");
    pairParticlesLD = CudaArray::create<int2>(cu, max((int) pairParticlesLDVec.size(), 1), "drudePairParticlesLD");

    if (!normalParticlesLDVec.empty())
        normalParticlesLD->upload(normalParticlesLDVec);
    if (!pairParticlesLDVec.empty())
        pairParticlesLD->upload(pairParticlesLDVec);

    map<string, string> defines;
    defines["NUM_NORMAL_PARTICLES_LD"] = cu.intToString(normalParticlesLDVec.size());
    defines["NUM_PAIRS_LD"] = cu.intToString(pairParticlesLDVec.size());
    CUmodule module = cu.createModule(CudaVVKernelSources::vectorOps + CudaVVKernelSources::drudeLangevin, defines, "");
    kernelApplyLangevin = cu.getKernel(module, "addExtraForceDrudeLangevin");

    cout << "CUDA modules for DrudeLangevinModifier are created\n"
         << "    Num normal particles: " << normalParticlesLDVec.size() << ", Num Drude pairs: " << pairParticlesLDVec.size() << "\n"
         << "    Real T: " << integrator.getTemperature() << " K, Drude T: " << integrator.getDrudeTemperature() << " K\n"
         << "    Real friction: " << integrator.getFriction() << " /ps, Drude friction: " << integrator.getDrudeFriction() << " /ps\n";
}

void CudaModifyDrudeLangevinKernel::applyLangevinForce(ContextImpl& context, const VVIntegrator& integrator) {
    if (integrator.getDebugEnabled())
        cout << "CudaModifyDrudeLangevinKernel apply Langevin force\n" << flush;

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
                     &vvStepKernel->getForceExtra()->getDevicePointer(),
                     &normalParticlesLD->getDevicePointer(),
                     &pairParticlesLD->getDevicePointer(),
                     dragPtr, randPtr, dragDrudePtr, randDrudePtr,
                     &integration.getRandom().getDevicePointer(),
                     &randomIndex};
    cu.executeKernel(kernelApplyLangevin, args1, integrator.getParticlesLD().size());
}

CudaModifyImageChargeKernel::~CudaModifyImageChargeKernel() {
    delete imagePairs;
}

void CudaModifyImageChargeKernel::initialize(const System& system, const VVIntegrator& integrator) {
    if (integrator.getDebugEnabled())
        cout << "Initializing CudaModifyImageChargeKernel...\n" << flush;

    cu.getPlatformData().initializeContexts(system);

    // Identify particle pairs and ordinary particles.

    auto imagePairsVec = std::vector<int2>(0);
    for (auto pair: integrator.getImagePairs())
        imagePairsVec.push_back(make_int2(pair.first, pair.second));

    imagePairs = CudaArray::create<int2>(cu, max((int) imagePairsVec.size(), 1), "imagePairs");
    if (!imagePairsVec.empty())
        imagePairs->upload(imagePairsVec);

    map<string, string> defines;
    defines["NUM_IMAGES"] = cu.intToString(imagePairsVec.size());
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    CUmodule module = cu.createModule(CudaVVKernelSources::vectorOps+CudaVVKernelSources::imageCharge, defines, "");
    kernelImage = cu.getKernel(module, "updateImagePositions");

    cout << "CUDA modules for ImageChargeModifier are created\n"
         << "    Num image pairs: " << imagePairsVec.size() << "\n"
         << "    Mirror location (z): " << integrator.getMirrorLocation() << " nm\n";
}

void CudaModifyImageChargeKernel::updateImagePositions(ContextImpl& context, const VVIntegrator& integrator) {
    if (integrator.getDebugEnabled())
        cout << "CudaModifyImageChargeKernel update image positions\n" << flush;

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
    cu.executeKernel(kernelImage, args2, integrator.getImagePairs().size());
}

CudaModifyElectricFieldKernel::~CudaModifyElectricFieldKernel() {
    delete particlesElectrolyte;
}

void CudaModifyElectricFieldKernel::initialize(const System &system, const VVIntegrator &integrator, Kernel& vvKernel) {
    if (integrator.getDebugEnabled())
        cout << "Initializing CudaModifyElectricFieldKernel...\n" << flush;

    vvStepKernel = &vvKernel.getAs<CudaIntegrateVVStepKernel>();
    cu.getPlatformData().initializeContexts(system);
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();
    cu.getIntegrationUtilities().initRandomNumberGenerator((unsigned int) integrator.getRandomNumberSeed());

    const auto& particlesElectrolyteVec = integrator.getParticlesElectrolyte();
    particlesElectrolyte = CudaArray::create<int>(cu, max((int) particlesElectrolyteVec.size(), 1), "particlesElectrolyte");
    if (!particlesElectrolyteVec.empty())
        particlesElectrolyte->upload(particlesElectrolyteVec);

    map<string, string> defines;
    defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    defines["NUM_PARTICLES_ELECTROLYTE"] = cu.intToString(particlesElectrolyteVec.size());
    CUmodule module = cu.createModule(CudaVVKernelSources::vectorOps + CudaVVKernelSources::electricField, defines, "");
    kernelApplyElectricForce = cu.getKernel(module, "addExtraForceElectricField");

    cout << "CUDA modules for ElectricFieldModifier are created\n"
         << "    Num electrolyte particles: " << particlesElectrolyteVec.size() << "\n"
         << "    Electric field strength (z): " << integrator.getElectricField() * 6.241509629152651e21 << " V/nm\n";
}

void CudaModifyElectricFieldKernel::applyElectricForce(ContextImpl& context, const VVIntegrator& integrator) {
    if (integrator.getDebugEnabled())
        cout << "CudaModifyElectricFieldKernel apply electric force\n" << flush;

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
                     &vvStepKernel->getForceExtra()->getDevicePointer(),
                     &particlesElectrolyte->getDevicePointer(),
                     efieldPtr,
                     fscalePtr};
    cu.executeKernel(kernelApplyElectricForce, args1, particlesElectrolyte->getSize());
}

CudaModifyCosineAccelerateKernel::~CudaModifyCosineAccelerateKernel() {
    delete vMaxBuffer;
}

void CudaModifyCosineAccelerateKernel::initialize(const System &system, const VVIntegrator &integrator, Kernel& vvKernel) {
    if (integrator.getDebugEnabled())
        cout << "Initializing CosineAccelerateModifier...\n" << flush;

    vvStepKernel = &vvKernel.getAs<CudaIntegrateVVStepKernel>();
    cu.getPlatformData().initializeContexts(system);
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();
    cu.getIntegrationUtilities().initRandomNumberGenerator((unsigned int) integrator.getRandomNumberSeed());

    numAtoms = cu.getNumAtoms();
    map<string, string> defines;
    defines["NUM_ATOMS"] = cu.intToString(numAtoms);
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    CUmodule module= cu.createModule(CudaVVKernelSources::vectorOps + CudaVVKernelSources::cosineAccelerate, defines, "");
    kernelAccelerate = cu.getKernel(module, "addCosAcceleration");
    kernelCalcV = cu.getKernel(module, "calcPeriodicVelocityBias");
    kernelRemoveBias = cu.getKernel(module, "removePeriodicVelocityBias");
    kernelRestoreBias = cu.getKernel(module, "restorePeriodicVelocityBias");
    kernelSumV = cu.getKernel(module, "sumV");

    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision())
        vMaxBuffer = CudaArray::create<double>(cu, numAtoms, "cosAccelerateVMaxBuffer");
    else
        vMaxBuffer = CudaArray::create<float>(cu, numAtoms, "cosAccelerateVMaxBuffer");

    double massTotal = 0;
    for (int i = 0; i < numAtoms; i++)
        massTotal += system.getParticleMass(i);
    invMassTotal = 1.0 / massTotal;

    cout << "CUDA modules for CosineAccelerateModifier are created\n"
         << "    Cosine acceleration strength: " << integrator.getCosAcceleration() << " nm/ps^2\n";
}

void CudaModifyCosineAccelerateKernel::applyCosineForce(ContextImpl& context, const VVIntegrator& integrator) {
    if (integrator.getDebugEnabled())
        cout << "CosineAccelerateModifier apply cosine acceleration force\n" << flush;

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
                     &vvStepKernel->getForceExtra()->getDevicePointer(),
                     accelerationPtr,
                     cu.getInvPeriodicBoxSizePointer()};
    cu.executeKernel(kernelAccelerate, args1, numAtoms);
}

void CudaModifyCosineAccelerateKernel::calcVelocityBias(ContextImpl& context, const VVIntegrator& integrator) {
    if (integrator.getDebugEnabled())
        cout << "CosineAccelerateModifier calculate velocity bias\n" << flush;

    cu.setAsCurrent();
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();

    void *args1[] = {&cu.getPosq().getDevicePointer(),
                     &cu.getVelm().getDevicePointer(),
                     &vMaxBuffer->getDevicePointer(),
                     cu.getInvPeriodicBoxSizePointer()};
    cu.executeKernel(kernelCalcV, args1, numAtoms);

    int bufferSize = vMaxBuffer->getSize();
    // Use only one threadBlock for this kernel because we use shared memory
    int workGroupSize = 512;
    void *args2[] = {&vMaxBuffer->getDevicePointer(),
                     &invMassTotal,
                     &bufferSize};
    cu.executeKernel(kernelSumV, args2, workGroupSize, workGroupSize,
                     workGroupSize * vMaxBuffer->getElementSize());
}

void CudaModifyCosineAccelerateKernel::removeVelocityBias(ContextImpl& context, const VVIntegrator& integrator) {
    cu.setAsCurrent();
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();

    if (integrator.getDebugEnabled())
        cout << "CosineAccelerateModifier remove velocity bias\n" << flush;

    void *args1[] = {&cu.getPosq().getDevicePointer(),
                     &cu.getVelm().getDevicePointer(),
                     &vMaxBuffer->getDevicePointer(),
                     cu.getInvPeriodicBoxSizePointer()};
    cu.executeKernel(kernelRemoveBias, args1, numAtoms);
}

void CudaModifyCosineAccelerateKernel::restoreVelocityBias(ContextImpl& context, const VVIntegrator& integrator) {
    if (integrator.getDebugEnabled())
        cout << "CosineAccelerateModifier restore velocity bias\n" << flush;

    cu.setAsCurrent();
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();

    void *args1[] = {&cu.getPosq().getDevicePointer(),
                     &cu.getVelm().getDevicePointer(),
                     &vMaxBuffer->getDevicePointer(),
                     cu.getInvPeriodicBoxSizePointer()};
    cu.executeKernel(kernelRestoreBias, args1, numAtoms);
}

void CudaModifyCosineAccelerateKernel::calcViscosity(ContextImpl& context, const VVIntegrator& integrator, double& vMax, double& invVis) {
    if (integrator.getDebugEnabled())
        cout << "CosineAccelerateModifier calculate viscosity\n" << flush;

    cu.setAsCurrent();
    CudaIntegrationUtilities &integration = cu.getIntegrationUtilities();

    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        auto vMaxBufferVec = std::vector<double>(numAtoms, 0);
        vMaxBuffer->download(vMaxBufferVec);
        vMax = vMaxBufferVec[0];
    } else {
        auto vMaxBufferVec = std::vector<float>(numAtoms, 0);
        vMaxBuffer->download(vMaxBufferVec);
        vMax = (double) vMaxBufferVec[0];
    }

    double4 box = cu.getPeriodicBoxSize();
    double vol = box.x * box.y * box.z;

    invVis = vMax * vol * invMassTotal / integrator.getCosAcceleration()
             * (2 * 3.1415926 / box.z) * (2 * 3.1415926 / box.z);
}
