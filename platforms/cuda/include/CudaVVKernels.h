#ifndef CUDA_VV_KERNELS_H_
#define CUDA_VV_KERNELS_H_

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

#include "openmm/VVKernels.h"
#include "CudaContext.h"
#include "CudaArray.h"

namespace OpenMM {


/**
 * This kernel is invoked by DrudeNoseHooverIntegrator to take one time step
 */
class CudaIntegrateVVStepKernel : public IntegrateVVStepKernel {
public:
    CudaIntegrateVVStepKernel(std::string name, const Platform &platform, CudaContext &cu) :
            IntegrateVVStepKernel(name, platform), cu(cu) {
    }
    ~CudaIntegrateVVStepKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the DrudeNoseHooverIntegrator this kernel will be used for
     * @param force      the DrudeForce to get particle parameters from
     */
    void initialize(const System& system, const VVIntegrator& integrator, const DrudeForce& force);
    /**
     * Perform first-half velocity-verlet integration
     *
     * @param context        the context in which to execute this kernel
     * @param integrator     the DrudeNoseHooverIntegrator this kernel is being used for
     */
    void firstIntegrate(ContextImpl& context, const VVIntegrator& integrator, bool& forcesAreValid);
    /**
     * Reset the extra forces to zero so that we can calculate langein force, external electric force etc
     * @param context
     * @param integrator
     */
    void resetExtraForce(ContextImpl& context, const VVIntegrator& integrator);
    /**
     * Perform the second-half velocity-verlet integration
     *
     * @param context        the context in which to execute this kernel
     * @param integrator     the DrudeNoseHooverIntegrator this kernel is being used for
     */
    void secondIntegrate(ContextImpl& context, const VVIntegrator& integrator, bool& forcesAreValid);
    /**
     * Compute the kinetic energy.
     *
     * @param context       the context in which to execute this kernel
     * @param integrator    the DrudeNoseHooverIntegrator this kernel is being used for
     */
    double computeKineticEnergy(ContextImpl& context, const VVIntegrator& integrator);

    CudaArray* getForceExtra(){
        return forceExtra;
    }
private:
    CudaContext& cu;
    double prevStepSize;
    int numAtoms;
    std::vector<int2> drudePairsVec;
    CudaArray *forceExtra;
    CudaArray *drudePairs;
    CUfunction kernelVel, kernelPos, kernelDrudeHardwall, kernelResetExtraForce;
};

/**
 * This kernel is invoked by DrudeNoseHooverIntegrator to take one time step
 */
    class CudaModifyDrudeNoseKernel : public ModifyDrudeNoseKernel {
    public:
        CudaModifyDrudeNoseKernel(std::string name, const Platform &platform, CudaContext &cu) :
                ModifyDrudeNoseKernel(name, platform), cu(cu) {
        }

        ~CudaModifyDrudeNoseKernel();

        /**
         * Initialize the kernel.
         *
         * @param system     the System this kernel will be applied to
         * @param integrator the DrudeNoseHooverIntegrator this kernel will be used for
         * @param force      the DrudeForce to get particle parameters from
         */
        void initialize(const System &system, const VVIntegrator &integrator, const DrudeForce &force);
        /**
         * Calculate the kinetic energies of temperature groups and propagate the NH chains
         *
         * @param context        the context in which to execute this kernel
         * @param integrator     the DrudeNoseHooverIntegrator this kernel is being used for
         */
        void calcGroupKineticEnergies(ContextImpl &context, const VVIntegrator &integrator);
        /**
         * Calculate the kinetic energies of temperature groups and propagate the NH chains
         *
         * @param context        the context in which to execute this kernel
         * @param integrator     the DrudeNoseHooverIntegrator this kernel is being used for
         */
        void propagateNHChain(ContextImpl &context, const VVIntegrator &integrator);

        /**
         * Scale the velocity based on the results of propagation of NH chains
         *
         * @param context        the context in which to execute this kernel
         * @param integrator     the DrudeNoseHooverIntegrator this kernel is being used for
         */
        void scaleVelocity(ContextImpl &context);

    private:
        CudaContext &cu;
        double drudekbT, realkbT;
        int numAtoms, numTempGroups;
        std::vector<int> particlesNHVec, residuesNHVec;
        CudaArray *particlesNH;
        CudaArray *residuesNH;
        CudaArray *normalParticlesNH;
        CudaArray *pairParticlesNH;
        CudaArray *vscaleFactors;
        CudaArray *particleResId;
        CudaArray *particleTempGroup;
        CudaArray *particlesInResidues;
        CudaArray *particlesSortedByResId;
        CudaArray *comVelm;
        CudaArray *normVelm;
        CudaArray *kineticEnergyBuffer;
        CudaArray *kineticEnergies; // 2 * kinetic energy
        std::vector<std::vector<double> > etaMass;
        std::vector<std::vector<double> > eta;
        std::vector<std::vector<double> > etaDot;
        std::vector<std::vector<double> > etaDotDot;
        std::vector<double> tempGroupDof;
        std::vector<int> particleResIdVec;
        std::vector<int2> particlesInResiduesVec;
        std::vector<int> particlesSortedByResIdVec;
        std::vector<int> particleTempGroupVec;
        std::vector<int> normalParticlesNHVec;
        std::vector<int2> pairParticlesNHVec;
        std::vector<double> tempGroupNkbT;
        std::vector<double> vscaleFactorsVec;
        std::vector<double> kineticEnergiesVec; // 2 * kinetic energy
        CUfunction kernelKE, kernelKESum, kernelScale, kernelNormVel, kernelCOMVel;
    };

/**
 * This kernel is invoked by DrudeNoseHooverIntegrator to take one time step
 */
    class CudaModifyDrudeLangevinKernel : public ModifyDrudeLangevinKernel {
    public:
        CudaModifyDrudeLangevinKernel(std::string name, const Platform &platform, CudaContext &cu) :
                ModifyDrudeLangevinKernel(name, platform), cu(cu) {
        }

        ~CudaModifyDrudeLangevinKernel();

        /**
         * Initialize the kernel.
         *
         * @param system     the System this kernel will be applied to
         * @param integrator the DrudeNoseHooverIntegrator this kernel will be used for
         * @param force      the DrudeForce to get particle parameters from
         */
        void initialize(const System &system, const VVIntegrator &integrator, const DrudeForce &force, Kernel& vvKernel);

        /**
         * Calculate the Langevin force for particles thermolized by Langevin dynamics
         * @param context
         * @param integrator
         */
        void applyLangevinForce(ContextImpl &context, const VVIntegrator &integrator);

    private:
        CudaIntegrateVVStepKernel* vvStepKernel;
        CudaContext &cu;
        std::vector<int> normalParticlesLDVec;
        std::vector<int2> pairParticlesLDVec;
        CudaArray *particlesLD;
        CudaArray *normalParticlesLD;
        CudaArray *pairParticlesLD;
        CUfunction kernelApplyLangevin;
    };

    /**
     * This kernel is invoked by DrudeNoseHooverIntegrator to take one time step.
     */
    class CudaModifyImageChargeKernel : public ModifyImageChargeKernel {
    public:
        CudaModifyImageChargeKernel(std::string name, const Platform &platform, CudaContext &cu)
                : ModifyImageChargeKernel(name, platform), cu(cu) {
        }

        ~CudaModifyImageChargeKernel();

        /**
         * Initialize the kernel.
         *
         * @param system     the System this kernel will be applied to
         * @param integrator the DrudeNoseHooverIntegrator this kernel will be used for
         * @param force      the DrudeForce to get particle parameters from
         */
        void initialize(const System &system, const VVIntegrator &integrator);

        /**
         * Execute the kernel.
         *
         * @param context        the context in which to execute this kernel
         * @param integrator     the DrudeNoseHooverIntegrator this kernel is being used for
         */
        void updateImagePositions(ContextImpl &context, const VVIntegrator &integrator);

    private:
        CudaContext &cu;
        CudaArray *imagePairs;
        std::vector<int2> imagePairsVec;
        CUfunction kernelImage;
    };

/**
 * This kernel is invoked by DrudeNoseHooverIntegrator to update image charge positions
 */
    class CudaModifyElectricFieldKernel: public ModifyElectricFieldKernel {
    public:
        CudaModifyElectricFieldKernel(std::string name, const Platform &platform, CudaContext &cu)
        : ModifyElectricFieldKernel(name, platform), cu(cu) {
        }

        ~CudaModifyElectricFieldKernel();
        /**
         * Initialize the kernel.
         *
         * @param system     the System this kernel will be applied to
         * @param integrator the DrudeNoseHooverIntegrator this kernel will be used for
         * @param force      the DrudeForce to get particle parameters from
         */
        void initialize(const System& system, const VVIntegrator& integrator, Kernel& vvKernel);
        /**
         * Execute the kernel.
         *
         * @param context        the context in which to execute this kernel
         * @param integrator     the DrudeNoseHooverIntegrator this kernel is being used for
         */
        void applyElectricForce(ContextImpl& context, const VVIntegrator& integrator);

    private:
        CudaIntegrateVVStepKernel* vvStepKernel;
        CudaContext &cu;
        CudaArray *particlesElectrolyte;
        CUfunction kernelApplyElectricForce;
    };

/**
 * This kernel is invoked by DrudeNoseHooverIntegrator to take one time step
 */
    class CudaModifyPeriodicPerturbationKernel: public ModifyPeriodicPerturbationKernel{
    public:
        CudaModifyPeriodicPerturbationKernel(std::string name, const Platform &platform, CudaContext &cu) :
                ModifyPeriodicPerturbationKernel(name, platform), cu(cu) {
        }
        ~CudaModifyPeriodicPerturbationKernel();
        /**
         * Initialize the kernel.
         *
         * @param system     the System this kernel will be applied to
         * @param integrator the DrudeNoseHooverIntegrator this kernel will be used for
         * @param force      the DrudeForce to get particle parameters from
         */
        void initialize(const System& system, const VVIntegrator& integrator, Kernel& vvKernel);
        /**
         * Apply the periodic perturbation force for viscosity calculation
         * @param context
         * @param integrator
         */
        void applyCosForce(ContextImpl& context, const VVIntegrator& integrator);
        /**
         * Calculate the velocity bias because of the periodic perturbation force
         * @param context
         * @param integrator
         */
        void calcVelocityBias(ContextImpl& context, const VVIntegrator& integrator);
        /**
         * Remove the velocity bias before thermostat
         * @param context
         * @param integrator
         */
        void removeVelocityBias(ContextImpl& context, const VVIntegrator& integrator);
        /**
         * Restore the velocity bias after thermostat
         * @param context
         * @param integrator
         */
        void restoreVelocityBias(ContextImpl& context, const VVIntegrator& integrator);
        /**
         * Calculate the reciprocal viscosity from the velocity profile because of the cos acceleration
         * @param context
         * @param integrator
         * @param vMax
         * @param invVis
         */
        void calcViscosity(ContextImpl& context, const VVIntegrator& integrator, double& vMax, double& invVis);
    private:
        CudaIntegrateVVStepKernel* vvStepKernel;
        CudaContext& cu;
        int numAtoms;
        CUfunction kernelCos;
        CUfunction kernelCalcBias;
        CUfunction kernelSumV;
        CUfunction kernelRemoveBias;
        CUfunction kernelRestoreBias;
        CudaArray* vMaxBuffer;
        double invMassTotal;
    };

} // namespace OpenMM

#endif /*CUDA_VV_KERNELS_H_*/
