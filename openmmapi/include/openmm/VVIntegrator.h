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
 * This Integrator simulates systems that include Drude particles.  It applies two different NoseHoover
 * thermostats to different parts of the system.  The first is applied to ordinary particles (ones that
 * are not part of a Drude particle pair), as well as to the center of mass of each Drude particle pair.
 * A second thermostat, typically with a much lower temperature, is applied to the relative internal
 * displacement of each pair.
 *
 * This integrator can optionally set an upper limit on how far any Drude particle is ever allowed to
 * get from its parent particle.  This can sometimes help to improve stability.  The limit is enforced
 * with a hard wall constraint.
 * 
 * This Integrator requires the System to include a DrudeForce, which it uses to identify the Drude
 * particles.
 */

class OPENMM_EXPORT_DRUDE VVIntegrator : public Integrator {
public:
    /**
     * Create a DrudeNoseHooverIntegrator.
     *
     * @param temperature    the temperature of the main heat bath (in Kelvin)
     * @param couplingTime  the characteristic time with which couples the system to the main heat bath (in picoseconds)
     * @param drudeTemperature    the temperature of the heat bath applied to internal coordinates of Drude particles (in Kelvin)
     * @param drudeCouplingTime  the characteristic time with which couples the system to the heat bath applied to internal coordinates of Drude particles (in picoseconds)
     * @param stepSize       the step size with which to integrator the system (in picoseconds)
     * @param loopsPerStep   the number of loops of NH velocity scaling per single step (integer)
     * @param numNHChains    the number of Nose-Hoover chains (integer)
     * @param useDrudeNHChains   whether to use the NH-Chain for Drude DOFs (bool)
     * @param useCOMTempGroup    whether to set molecular COM motion as a separate temperature group (bool)
     */
    VVIntegrator(double temperature, double couplingTime, double drudeTemperature, double drudeCouplingTime, double stepSize, int loopsPerStep=1, int numNHChains=3, bool useDrudeNHChains=true, bool useCOMTempGroup=true);

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
     * Get the coupling time t which determines how quickly the system is coupled to
     * the main heat bath (in ps).
     *
     * @return the coupling time, measured in ps
     */
    double getCouplingTime() const {
        return couplingTime;
    }
    /**
     * Set the coupling time which determines how quickly the system is coupled to
     * the main heat bath (in ps).
     *
     * @param tau    the coupling time, measured in ps
     */
    void setCouplingTime(double tau) {
        couplingTime = tau;
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
     * Get the coupling time which determines how quickly the internal coordinates of Drude particles
     * are coupled to the heat bath (in ps).
     *
     * @return the coupling time, measured in ps
     */
    double getDrudeCouplingTime() const {
        return drudeCouplingTime;
    }
    /**
     * Set the coupling time which determines how quickly the internal coordinates of Drude particles
     * are coupled to the heat bath (in ps).
     *
     * @param tau    the coupling time, measured in ps
     */
    void setDrudeCouplingTime(double tau) {
        drudeCouplingTime = tau;
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
     * Get whether to use Nose-Hoover chains for Drude DOFs
     *
     * @return
     */
    bool getUseDrudeNHChains() const {
        return useDrudeNHChains;
    }
    /**
     * Set whether to use Nose-Hoover chains for Drude DOFs
     *
     * @return
     */
    void setUseDrudeNHChains(bool use) {
        useDrudeNHChains = use;
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
     * Set whether to use COM Temperature group or not (one should always use COM temp group)
     *
     * @param useCOMGroup    boolean, whether to use COM temperature group or not
     */
    void setUseCOMTempGroup(bool use) {
        useCOMTempGroup = use;
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
     * Get the number of temperature groups for particles which independent thermal bath is used.
     * @return the number of temperature groups for real d.o.f
     */
    int getNumTempGroups() const {
        return tempGroups.size();
    }
    /**
     * Add a new temperature group
     *
     * @return the index of the new temperature group that was added
     */
    int addTempGroup();
    /**
     * Add the temperature group of a particle to the last index of particleTempGroup.
     * Drude particles and particles within constraint should be assigned to the same temp group
     *
     * @param tempGroup       the index of temperature group to be assigned for the last particle
     * @return the index of the particle that was added
     */
    int addParticleTempGroup(int tempGroup);
    /**
     * Get the temperature group of a real particle.
     *
     * @param particle              the index of the particle for which to get parameters
     * @param[out] tempGroup        the index of the temperature group to which the particle is assigned
     */
    void getParticleTempGroup(int particle, int& tempGroup) const;
    /**
     * Set the temperature group of a particle.
     * Drude particles and particles within constraint should be assigned to the same temp group
     *
     * @param particle        the index within the System of the particle
     * @param tempGroup       the index of temperature group for the partile to be assigned
     */
    void setParticleTempGroup(int particle, int tempGroup);
    /**
     * Thermolize this particle with Langevin thermostat instead of Nose-Hoover
     * @param particle The indice of particle to be themrostated by Langevin dynamics
     * @return the number of atoms that will be thermostated by Langevin dynamics
     */
    int addParticleLangevin(int particle) {
        particlesLD.push_back(particle);
        return particlesLD.size();
    };
    /**
     * Get the friction of Langevin thermostat for real atoms (in /ps).
     *
     * @return the coupling time, measured in ps
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
     * @return the coupling time, measured in ps
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
     * Get the random number seed.  See setRandomNumberSeed() for details.
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
     * @param image the index of image particle
     * @param parent the index of the parent of this iamge particle
     * @return the number of image particles in the system
     */
    int addImagePair(int image, int parent);
    /**
     * Get the indices of particles thermolized by NH
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
     * @param particle
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
     * Get the indices of residues thermolized by NH
     * @return
     */
    const std::vector<int> & getResiduesNH() const {
        return residuesNH;
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
     * Get the number of residues in the system
     * @return the number of residues in the system
     */
    int getNumResidues() const {
        return residueMasses.size();
    }
    /**
     * Get the inverse mass of a residue with residue index
     *
     * @param resid                 the index of the residue for which to get parameters
     * return resMass               the mass of the residue with index resid
     */
    double getResInvMass(int resid) const;
    /**
     * Get the residue id of a particle with particle index
     *
     * @param particle              the index of the particle for which to get parameters
     * return resid                 the index of the residue of the particle with index particle
     */
    int getParticleResId(int particle) const;
    /**
     * Get the strength of periodic perturbation acceleration for viscosity calculation
     */
    double getCosAcceleration() const {
        return cosAcceleration;
    }
    /**
     * Get the strength of periodic perturbation acceleration for viscosity calculation
     * @param acceleration
     */
    void setCosAcceleration(double acceleration) {
        cosAcceleration = acceleration;
    }
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
    double temperature, couplingTime, drudeTemperature, drudeCouplingTime, maxDrudeDistance;
    int loopsPerStep, numNHChains;
    bool useDrudeNHChains, useCOMTempGroup;
    std::vector<int> particlesNH;
    std::vector<int> residuesNH;
    std::vector<int> tempGroups;
    std::vector<int> particleTempGroup;
    std::vector<int> particleResId;
    std::vector<double> residueMasses;
    std::vector<double> residueInvMasses;
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
