//
// Created by zheng on 2020-03-01.
//

#ifndef OPENMM_VELOCITYVERLET_VVMODIFIER_H
#define OPENMM_VELOCITYVERLET_VVMODIFIER_H

#include <algorithm>
#include "openmm/Integrator.h"
#include "openmm/Kernel.h"
#include "openmm/internal/windowsExportDrude.h"
#include "VVIntegrator.h"

namespace OpenMM {
    /**
     * A modifier can modify the force, velocity and/or position during a velocity verlet step
     */
    class OPENMM_EXPORT_DRUDE VVModifier {
    public:
        /**
         * Create a modifier fot VVIntegrator
         *
         * @param integrator
         */
        VVModifier(VVIntegrator &integrator):isThermostat(false), hasVelocityBias(false) {};

        virtual void beforeFirstIntegration() = 0;
        virtual void afterFirstIntegration() = 0;
        virtual void beforeForceCalculation() = 0;
        virtual void afterForceCalculation() = 0;
        virtual void afterSecondIntegration() = 0;

        const bool& getIsThermostat() const {
            return isThermostat;
        }

        void setIsThermostat(bool is){
            isThermostat = is;
        }

        const bool& getHasVelocityBias() const {
            return hasVelocityBias;
        }

        void setHasVelocityBias(bool has){
            hasVelocityBias = has;
        }

    private:
        bool isThermostat;
        bool hasVelocityBias;
    };

    class DrudeNoseThermostatModifier: VVModifier {
    public:
        DrudeNoseThermostatModifier(VVIntegrator &integrator);

        ~DrudeNoseThermostatModifier();

        void beforeFirstIntegration();
        void afterFirstIntegration();
    };

    class LangevinModifier: VVModifier {
    public:
        LangevinModifier(VVIntegrator &integrator);

        ~LangevinModifier();

        void afterForceCalculation();
    };

    class ElectricFieldModifier : VVModifier {
    public:
        ElectricFieldModifier(VVIntegrator &integrator);

        ~ElectricFieldModifier();

        void afterForceCalculation();
    };

    class ImageChargeModifier: VVModifier {
    public:
        ImageChargeModifier(VVIntegrator &integrator);

        ~ImageChargeModifier();

        void afterFirstIntegration();
    };

    class PeriodicPerturbationModifier: VVModifier {
    public:
        PeriodicPerturbationModifier(VVIntegrator &integrator);

        ~PeriodicPerturbationModifier();

        void afterForceCalculation();
    };

} // namespace OpenMM

#endif //OPENMM_VELOCITYVERLET_VVMODIFIER_H
