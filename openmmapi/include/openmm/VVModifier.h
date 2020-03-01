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
        VVModifier(VVIntegrator &integrator) {};

        /**
         * Get whether this modifier will modify force
         */
        virtual bool getModifyForce() = 0;

        /**
         * Get whether this modifier will modify velocity
         */
        virtual bool getModifyVelocity() = 0;

        /**
         * Get whether this modifier will modify position
         */
        virtual bool getModifyPosition() = 0;
    };

    class ElectricFieldModifier : VVModifier {
    public:
        ElectricFieldModifier(VVIntegrator &integrator) : VVModifier(integrator) {

        };

        ~ElectricFieldModifier() {

        };

        bool getModifyForce() const {
            return true;
        }

        bool getModifyVelocity() const {
            return false;
        }

        bool getModifyPosition() const {
            return false;
        }
    };

} // namespace OpenMM

#endif //OPENMM_VELOCITYVERLET_VVMODIFIER_H
