#ifndef VV_KERNELS_H_
#define VV_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2013 Stanford University and the Authors.           *
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

#include "openmm/DrudeForce.h"
#include "openmm/VVIntegrator.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/Vec3.h"
#include <string>
#include <vector>

namespace OpenMM {

/**
 * This kernel is invoked by DrudeNoseHooverIntegrator to take one time step.
 */
class IntegrateVVStepKernel : public KernelImpl {
public:
    static std::string Name() {
        return "IntegrateVVStep";
    }
    IntegrateVVStepKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the DrudeNoseHooverIntegrator this kernel will be used for
     * @param force      the DrudeForce to get particle parameters from
     */
    virtual void initialize(const System& system, const VVIntegrator& integrator, const DrudeForce& force) = 0;
    /**
     * Execute the kernel.
     *
     * @param context        the context in which to execute this kernel
     * @param integrator     the DrudeNoseHooverIntegrator this kernel is being used for
     */
    virtual void firstIntegrate(ContextImpl& context, const VVIntegrator& integrator, bool& forcesAreValid) = 0;
    /**
     * Reset the extra forces to zero so that we can calculate langein force, external electric force etc
     * @param context
     * @param integrator
     */
    virtual void resetExtraForce(ContextImpl& context, const VVIntegrator& integrator) = 0;
    /**
     * Execute the kernel.
     *
     * @param context        the context in which to execute this kernel
     * @param integrator     the DrudeNoseHooverIntegrator this kernel is being used for
     */
    virtual void secondIntegrate(ContextImpl& context, const VVIntegrator& integrator, bool& forcesAreValid) = 0;
    /**
     * Compute the kinetic energy.
     */
    virtual double computeKineticEnergy(ContextImpl& context, const VVIntegrator& integrator) = 0;
};

/**
 * This kernel is invoked by DrudeNoseHooverIntegrator to take one time step.
 */
    class ModifyDrudeNoseKernel: public KernelImpl {
    public:
        static std::string Name() {
            return "DrudeNoseHoover";
        }
        ModifyDrudeNoseKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
        }
        /**
         * Initialize the kernel.
         *
         * @param system     the System this kernel will be applied to
         * @param integrator the DrudeNoseHooverIntegrator this kernel will be used for
         * @param force      the DrudeForce to get particle parameters from
         */
        virtual void initialize(const System& system, const VVIntegrator& integrator, const DrudeForce& force) = 0;
        /**
         * Calculate the kinetic energies of temperature groups and propagate the NH chains
         *
         * @param context        the context in which to execute this kernel
         * @param integrator     the DrudeNoseHooverIntegrator this kernel is being used for
         */
        virtual void calcGroupKineticEnergies(ContextImpl& context, const VVIntegrator& integrator) = 0;
        /**
         * Calculate the kinetic energies of temperature groups and propagate the NH chains
         *
         * @param context        the context in which to execute this kernel
         * @param integrator     the DrudeNoseHooverIntegrator this kernel is being used for
         */
        virtual void propagateNHChain(ContextImpl& context, const VVIntegrator& integrator) = 0;
        /**
         * Scale the velocity based on the results of propagation of NH chains
         *
         * @param context        the context in which to execute this kernel
         * @param integrator     the DrudeNoseHooverIntegrator this kernel is being used for
         */
        virtual void scaleVelocity(ContextImpl& context) = 0;
    };

/**
 * This kernel is invoked by DrudeNoseHooverIntegrator to update image charge positions
 */
    class ModifyDrudeLangevinKernel: public KernelImpl {
    public:
        static std::string Name() {
            return "ModifyLangevin";
        }
        ModifyDrudeLangevinKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
        }
        /**
         * Initialize the kernel.
         *
         * @param system     the System this kernel will be applied to
         * @param integrator the DrudeNoseHooverIntegrator this kernel will be used for
         * @param force      the DrudeForce to get particle parameters from
         */
        virtual void initialize(const System& system, const VVIntegrator& integrator, const DrudeForce& force, Kernel& vvKernel) = 0;
        /**
         * Execute the kernel.
         *
         * @param context        the context in which to execute this kernel
         * @param integrator     the DrudeNoseHooverIntegrator this kernel is being used for
         */
        virtual void applyLangevinForce(ContextImpl& context, const VVIntegrator& integrator) = 0;
    };

/**
 * This kernel is invoked by DrudeNoseHooverIntegrator to update image charge positions
 */
class ModifyImageChargeKernel: public KernelImpl {
    public:
        static std::string Name() {
            return "ModifyImageCharge";
        }
        ModifyImageChargeKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
        }
        /**
         * Initialize the kernel.
         *
         * @param system     the System this kernel will be applied to
         * @param integrator the DrudeNoseHooverIntegrator this kernel will be used for
         * @param force      the DrudeForce to get particle parameters from
         */
        virtual void initialize(const System& system, const VVIntegrator& integrator) = 0;
        /**
         * Execute the kernel.
         *
         * @param context        the context in which to execute this kernel
         * @param integrator     the DrudeNoseHooverIntegrator this kernel is being used for
         */
        virtual void updateImagePositions(ContextImpl& context, const VVIntegrator& integrator) = 0;
    };

/**
 * This kernel is invoked by DrudeNoseHooverIntegrator to update image charge positions
 */
    class ModifyElectricFieldKernel: public KernelImpl {
    public:
        static std::string Name() {
            return "ModifyElectricField";
        }
        ModifyElectricFieldKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
        }
        /**
         * Initialize the kernel.
         *
         * @param system     the System this kernel will be applied to
         * @param integrator the DrudeNoseHooverIntegrator this kernel will be used for
         * @param force      the DrudeForce to get particle parameters from
         */
        virtual void initialize(const System& system, const VVIntegrator& integrator, Kernel& vvKernel) = 0;
        /**
         * Execute the kernel.
         *
         * @param context        the context in which to execute this kernel
         * @param integrator     the DrudeNoseHooverIntegrator this kernel is being used for
         */
        virtual void applyElectricForce(ContextImpl& context, const VVIntegrator& integrator) = 0;
    };

/**
 * This kernel is invoked by DrudeNoseHooverIntegrator to update image charge positions
 */
    class ModifyPeriodicPerturbationKernel: public KernelImpl {
    public:
        static std::string Name() {
            return "ModifyPeriodicPerturbation";
        }
        ModifyPeriodicPerturbationKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
        }
        /**
         * Initialize the kernel.
         *
         * @param system     the System this kernel will be applied to
         * @param integrator the DrudeNoseHooverIntegrator this kernel will be used for
         * @param force      the DrudeForce to get particle parameters from
         */
        virtual void initialize(const System& system, const VVIntegrator& integrator, Kernel& vvKernel) = 0;
        /**
         * Execute the kernel.
         *
         * @param context        the context in which to execute this kernel
         * @param integrator     the DrudeNoseHooverIntegrator this kernel is being used for
         */
        virtual void applyCosForce(ContextImpl& context, const VVIntegrator& integrator) = 0;
        virtual void calcVelocityBias(ContextImpl& context, const VVIntegrator& integrator) = 0;
        virtual void removeVelocityBias(ContextImpl& context, const VVIntegrator& integrator) = 0;
        virtual void restoreVelocityBias(ContextImpl& context, const VVIntegrator& integrator) = 0;
        virtual void calcViscosity(ContextImpl& context, const VVIntegrator& integrator, double& vMax, double& invVis) = 0;
    };

} // namespace OpenMM

#endif /*VV_KERNELS_H_*/
