#ifndef OPENMM_VV_INTEGRATOR_H_
#define OPENMM_VV_INTEGRATOR_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2015 Stanford University and the Authors.      *
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

#include <algorithm>
#include "openmm/Integrator.h"
#include "openmm/Kernel.h"
#include "openmm/internal/windowsExportDrude.h"

namespace OpenMM {

/**
 * This Integrator simulates systems with velocity-Verlet algorithm. It works for both
 * non-polarizable model and Drude Model. Nose-Hoover thermostat and/or Langevin thermostat
 * can be applied for different parts of the system. For Drude model, the temperature-grouped
 * Nose-Hoover thermostat is supported.
 */

class OPENMM_EXPORT_DRUDE VVIntegrator : public Integrator {
public:
    /**
     * Create a VVIntegrator with Nose-Hoover thermostat
     *
     * @param temperature        the temperature of the main heat bath (in Kelvin)
     * @param Frequency          the characteristic frequency which couples the system to the main heat bath (in /picoseconds)
     * @param drudeTemperature   the temperature of the heat bath applied to internal coordinates of Drude particles (in Kelvin)
     * @param drudeFrequency     the characteristic frequency with which couples the system to the heat bath applied to internal coordinates of Drude particles (in /picoseconds)
     * @param stepSize           the step size with which to integrator the system (in picoseconds)
     * @param numNHChains        the number of Nose-Hoover chains (integer)
     * @param loopsPerStep       the number of loops of NH velocity scaling per single step (integer)
     */
    VVIntegrator(double temperature, double frequency, double drudeTemperature, double drudeFrequency, double stepSize, int numNHChains=3, int loopsPerStep=1);

    virtual ~VVIntegrator();
    /**
     * Get the temperature of the main heat bath (in Kelvin).
     *
     * @return the temperature of the heat bath, measured in Kelvin
     */
    double getTemperature() const {
        return temperature;
    }
    /**
     * Set the temperature of the main heat bath (in Kelvin).
     *
     * @param temp    the temperature of the heat bath, measured in Kelvin
     */
    void setTemperature(double temp) {
        temperature = temp;
    }
    /**
     * Get the coupling strength which determines how strongly the system is coupled to
     * the main heat bath (in /ps).
     *
     * @return the coupling strength, measured in /ps
     */
    double getFrequency() const {
        return frequency;
    }
    /**
     * Set the coupling strength which determines how strongly the system is coupled to
     * the main heat bath (in /ps).
     *
     * @param tau    the coupling strength, measured in /ps
     */
    void setFrequency(double tau) {
        frequency = tau;
    }
    /**
     * Get the temperature of the heat bath applied to internal coordinates of Drude particles (in Kelvin).
     *
     * @return the temperature of the heat bath, measured in Kelvin
     */
    double getDrudeTemperature() const {
        return drudeTemperature;
    }
    /**
     * Set the temperature of the heat bath applied to internal coordinates of Drude particles (in Kelvin).
     *
     * @param temp    the temperature of the heat bath, measured in Kelvin
     */
    void setDrudeTemperature(double temp) {
        drudeTemperature = temp;
    }
    /**
     * Get the coupling strength which determines how strongly the internal coordinates of Drude particles
     * are coupled to the heat bath (in /ps).
     *
     * @return the coupling strength, measured in /ps
     */
    double getDrudeFrequency() const {
        return drudeFrequency;
    }
    /**
     * Set the coupling strength which determines how strongly the internal coordinates of Drude particles
     * are coupled to the heat bath (in /ps).
     *
     * @param tau    the coupling strength, measured in /ps
     */
    void setDrudeFrequency(double tau) {
        drudeFrequency = tau;
    }
    /**
     * Get the number of Nose-Hoover chains (integer)
     *
     * @return the number of Nose-Hoover chains
     */
    int getNumNHChains() const {
        return numNHChains;
    }
    /**
     * Set the number of Nose-Hoover chains (integer)
     *
     * @param numChains    the number of Nose-Hoover chains
     */
    void setNumNHChains(int numChains) {
        numNHChains = numChains;
    }
    /**
     * Get the number of loops per Nose-Hoover propagation step
     *
     * @return
     */
    int getLoopsPerStep() const {
        return loopsPerStep;
    }
    /**
     * Set the number of loops per Nose-Hoover propagation step
     *
     * @param loops
     */
    void setLoopsPerStep(int loops) {
        loopsPerStep= loops;
    }
    /**
     * Get whether to use COM Temperature group or not
     *
     * @return whether to use COM Temperature group
     */
    bool getUseCOMTempGroup() const {
        return useCOMTempGroup;
    }
    /**
     * Set whether to use COM Temperature group or not
     */
    void setUseCOMTempGroup(bool use) {
        useCOMTempGroup = use;
        autoSetCOMTempGroup = false;
    }
    /**
     * Get the maximum distance a Drude particle can ever move from its parent particle, measured in nm.  This is implemented
     * with a hard wall constraint.  If this distance is set to 0 (the default), the hard wall constraint is omitted.
     */
    double getMaxDrudeDistance() const {
        return maxDrudeDistance;
    };
    /**
     * Set the maximum distance a Drude particle can ever move from its parent particle, measured in nm.  This is implemented
     * with a hard wall constraint.  If this distance is set to 0 (the default), the hard wall constraint is omitted.
     */
    void setMaxDrudeDistance(double distance) {
        maxDrudeDistance = distance;
    };
    /**
     * Thermolize this particle with Langevin thermostat instead of Nose-Hoover
     * @param particle    The index of particle to be themrostated by Langevin dynamics
     * @return the number of atoms that will be thermostated by Langevin dynamics
     */
    int addParticleLangevin(int particle) {
        particlesLD.push_back(particle);
        return particlesLD.size();
    };
    /**
     * Get the friction of Langevin thermostat for real atoms (in /ps).
     *
     * @return the friction, measured in /ps
     */
    double getFriction() const {
        return friction;
    }
    /**
     * Set the friction of Langevin thermostat for real atoms (in /ps).
     */
    void setFriction(double fric) {
        friction = fric;
    }
    /**
     * Get the friction of Langevin thermostat for Drude particles (in /ps).
     *
     * @return the friction, measured in /ps
     */
    double getDrudeFriction() const {
        return drudeFriction;
    }
    /**
     * Set the friction of Langevin thermostat for Drude particles (in /ps).
     */
    void setDrudeFriction(double fric) {
        drudeFriction = fric;
    }
    /**
     * Get the random number seed. See setRandomNumberSeed() for details.
     */
    int getRandomNumberSeed() const {
        return randomNumberSeed;
    }
    /**
     * Set the random number seed.  The precise meaning of this parameter is undefined, and is left up
     * to each Platform to interpret in an appropriate way.  It is guaranteed that if two simulations
     * are run with different random number seeds, the sequence of random forces will be different.  On
     * the other hand, no guarantees are made about the behavior of simulations that use the same seed.
     * In particular, Platforms are permitted to use non-deterministic algorithms which produce different
     * results on successive runs, even if those runs were initialized identically.
     *
     * If seed is set to 0 (which is the default value assigned), a unique seed is chosen when a Context
     * is created from this Force. This is done to ensure that each Context receives unique random seeds
     * without you needing to set them explicitly.
     */
    void setRandomNumberSeed(int seed) {
        randomNumberSeed = seed;
    }
    /**
     * Set a particle as image of another particle
     * @param image       The index of image particle
     * @param parent      The index of the parent of this image particle
     * @return the number of image particles in the system
     */
    int addImagePair(int image, int parent);
    /**
     * Get all the image pairs
     * @return
     */
    const std::vector<std::pair<int, int> > & getImagePairs() const {
        return imagePairs;
    }
    /**
     * Get the z coordinate of mirror for image charges
     * @param z
     */
    double getMirrorLocation() const {
        return mirrorLocation;
    }
    /**
     * Set the z coordinate of mirror for image charges
     * @param z
     */
    void setMirrorLocation(double z) {
        mirrorLocation = z;
    }
    /**
     * Get the strength of electric field applied on electrolyte particles in z direction (kJ/nm.e)
     * Be very careful about the unit. 1 V/nm = 1.60217662E-22 kJ/nm.e
     */
    double getElectricField() const {
        return electricField;
    }
    /**
     * Set the strength of electric field applied on electrolyte particles in z direction (kJ/nm.e)
     * Be very careful about the unit. 1 V/nm = 1.60217662E-22 kJ/nm.e
     * @param field
     */
    void setElectricField(double field) {
        electricField = field;
    }
    /**
     * Treat this particle as electrolyte so the electric field will applied on it
     * @param particle    The index of particle treated as electrolytes
     * @return the number of electrolyte particles
     */
    int addParticleElectrolyte(int particle) {
        particlesElectrolyte.push_back(particle);
        return particlesElectrolyte.size();
    };
    /**
     * Get the particles treated as electrolytes
     * @return
     */
    const std::vector<int> & getParticlesElectrolyte() const {
        return particlesElectrolyte;
    };
    /**
     * Get the indices of particles thermolized by NH
     * @return
     */
    const std::vector<int> & getParticlesNH() const {
        return particlesNH;
    }
    /**
     * Get the indices of particles thermolized by Langevin dynamics
     * @return
     */
    const std::vector<int> & getParticlesLD() const {
        return particlesLD;
    }
    /**
     * Get the indices of molecules thermolized by NH
     * @return
     */
    const std::vector<int> & getMoleculesNH() const {
        return moleculesNH;
    }
    /**
     * Check if a particle thermolized by NH
     * @return
     */
    bool isParticleNH(int i) const {
        return std::find(particlesNH.begin(), particlesNH.end(), i) != particlesNH.end();
    }
    /**
     * Check if a particle thermolized by Langevin dynamics
     * @return
     */
    bool isParticleLD(int i) const {
        return std::find(particlesLD.begin(), particlesLD.end(), i) != particlesLD.end();
    }
    /**
     * Check if a particle is image particle
     * @return
     */
    bool isParticleImage(int i) const {
        return std::find(particlesImage.begin(), particlesImage.end(), i) != particlesImage.end();
    }
    /**
     * Get the number of molecules in the system
     * @return
     */
    int getNumMolecules() const {
        return moleculeMasses.size();
    }
    /**
     * Get the inverse mass of a residue with residue index
     *
     * @param molid                 the index of the molecule for which to get parameters
     * return resMass               the mass of the molecule with index molid
     */
    double getMoleculeInvMass(int molid) const;
    /**
     * Get the molecule id of a particle with particle index
     *
     * @param particle              the index of the particle for which to get parameters
     * return molid                 the index of the molecule of the particle with index particle
     */
    int getParticleMolId(int particle) const;
    /**
     * Get the strength of cosine acceleration for viscosity calculation
     */
    double getCosAcceleration() const {
        return cosAcceleration;
    }
    /**
     * Set the strength of cosine acceleration for viscosity calculation
     * @param acceleration
     */
    void setCosAcceleration(double acceleration) {
        cosAcceleration = acceleration;
    }
    /**
     * Get the velocity scaling factor by propagating Nose-Hoover chain
     * @param
     */
    void propagateNHChain(std::vector<double> &eta, std::vector<double> &eta_dot,
                          std::vector<double> &eta_dotdot, std::vector<double> &eta_mass,
                          double ke2, double ke2_target, double t_target, double &scale) const;
    /**
     * Get the velocity at z=0 and reciprocal viscosity because of the cos acceleration
     * @param
     */
    std::vector<double> getViscosity();
    /**
     * Advance a simulation through time by taking a series of time steps.
     *
     * @param steps   the number of time steps to take
     */
    void step(int steps);
    /**
     * Get whether to print debug information
     */
    const bool& getDebugEnabled() const{
        return debugEnabled;
    };
    /**
     * Set whether to print debug information
     */
    void setDebugEnabled(bool enabled){
        debugEnabled = enabled;
    };
protected:
    /**
     * This will be called by the Context when it is created.  It informs the Integrator
     * of what context it will be integrating, and gives it a chance to do any necessary initialization.
     * It will also get called again if the application calls reinitialize() on the Context.
     */
    void initialize(ContextImpl& context);
    /**
     * This will be called by the Context when it is destroyed to let the Integrator do any necessary
     * cleanup.  It will also get called again if the application calls reinitialize() on the Context.
     */
    void cleanup();
    /**
     * When the user modifies the state, we need to mark that the forces need to be recalculated.
     */
    void stateChanged(State::DataType changed){
        forcesAreValid = false;
    };
    /**
     * Get the names of all Kernels used by this Integrator.
     */
    std::vector<std::string> getKernelNames();
    /**
     * Compute the kinetic energy of the system at the current time.
     */
    double computeKineticEnergy();
    /**
     * Get whether computeKineticEnergy() expects forces to have been computed.  The default
     * implementation returns true to be safe.  Non-leapfrog integrators can override this to
     * return false, which makes calling getState() to query the energy less expensive.
     */
    bool kineticEnergyRequiresForce() const {
        return false;
    }
private:
    bool debugEnabled;
    double temperature, frequency, drudeTemperature, drudeFrequency, maxDrudeDistance;
    int loopsPerStep, numNHChains;
    bool useCOMTempGroup, autoSetCOMTempGroup;
    std::vector<int> particlesNH;
    std::vector<int> moleculesNH;
    std::vector<int> particleMolId;
    std::vector<double> moleculeMasses;
    std::vector<double> moleculeInvMasses;
    Kernel vvKernel, nhKernel;
    bool forcesAreValid;

    std::vector<int> particlesLD;
    double friction, drudeFriction;
    int randomNumberSeed;
    Kernel ldKernel;

    // for constant voltage simulation with image charge method
    std::vector<std::pair<int, int> > imagePairs;
    std::vector<int> particlesImage;
    double mirrorLocation;
    double electricField;
    std::vector<int> particlesElectrolyte;
    Kernel imgKernel, efKernel;

    // for periodic perturbation viscosity calculation
    double cosAcceleration;
    Kernel ppKernel;
};

} // namespace OpenMM

#endif /*OPENMM_VV_INTEGRATOR_H_*/
