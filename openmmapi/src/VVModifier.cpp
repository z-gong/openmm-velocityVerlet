//
// Created by zheng on 2020-03-01.
//

#include "openmm/VVIntegrator.h"
#include "openmm/VVModifier.h"

using namespace OpenMM;

DrudeNoseThermostatModifier::DrudeNoseThermostatModifier(VVIntegrator &integrator) : VVModifier(integrator) {
    setIsThermostat(true);
}

DrudeNoseThermostatModifier::~DrudeNoseThermostatModifier() {

}

LangevinModifier::LangevinModifier(VVIntegrator &integrator) : VVModifier(integrator) {
    setIsThermostat(true);
}

LangevinModifier::~LangevinModifier() {

}

ElectricFieldModifier::ElectricFieldModifier(VVIntegrator &integrator) : VVModifier(integrator) {

}

ElectricFieldModifier::~ElectricFieldModifier() {

}

ImageChargeModifier::ImageChargeModifier(VVIntegrator &integrator) : VVModifier(integrator) {

}

ImageChargeModifier::~ImageChargeModifier() {

}

PeriodicPerturbationModifier::PeriodicPerturbationModifier(VVIntegrator &integrator) : VVModifier(integrator) {
    setHasVelocityBias(true);
}

PeriodicPerturbationModifier::~PeriodicPerturbationModifier() {

}
