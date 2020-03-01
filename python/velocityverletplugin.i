%module velocityverletplugin

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"

/*
 * The following lines are needed to handle std::vector.
 * Similar lines may be needed for vectors of vectors or
 * for other STL types like maps.
 */

%include "std_vector.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%{
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "OpenMMVelocityVerlet.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%pythoncode %{
import simtk.openmm as mm
import simtk.unit as unit
%}

/*
 * Add units to function outputs.
*/
%pythonappend OpenMM::VVIntegrator::getTemperature() const %{
   val=unit.Quantity(val, unit.kelvin)
%}

%pythonappend OpenMM::VVIntegrator::getCouplingTime() const %{
   val=unit.Quantity(val, unit.picosecond)
%}

%pythonappend OpenMM::VVIntegrator::getDrudeTemperature() const %{
   val=unit.Quantity(val, unit.kelvin)
%}

%pythonappend OpenMM::VVIntegrator::getDrudeCouplingTime() const %{
   val=unit.Quantity(val, unit.picosecond)
%}

%pythonappend OpenMM::VVIntegrator::getMaxDrudeDistance() const %{
   val=unit.Quantity(val, unit.nanometer)
%}

%pythonappend OpenMM::VVIntegrator::getFriction() const %{
    val=unit.Quantity(val, 1 / unit.picosecond)
%}

%pythonappend OpenMM::VVIntegrator::getDrudeFriction() const %{
    val=unit.Quantity(val, 1 / unit.picosecond)
%}

%pythonappend OpenMM::VVIntegrator::getMirrorLocation() const %{
    val=unit.Quantity(val, unit.nanometer)
%}

%pythonappend OpenMM::VVIntegrator::getElectricField() const %{
    val=unit.Quantity(val, unit.kilojoule / unit.nanometer / unit.elementary_charge).in_units_of(unit.volt / unit.nanometer)
%}

%pythonappend OpenMM::VVIntegrator::getCosAcceleration() const %{
    val=unit.Quantity(val, unit.nanometer / unit.picosecond / unit.picosecond)
%}

%pythonappend OpenMM::VVIntegrator::getViscosity() %{
    val=(unit.Quantity(val[0], unit.nanometer / unit.picosecond),
         unit.Quantity(val[1], unit.picosecond / (unit.dalton * unit.item) * unit.nanometer).in_units_of((unit.pascal * unit.second)**(-1))
        )
%}

namespace OpenMM {

class VVIntegrator : public Integrator {
public:
   VVIntegrator(double temperature, double couplingTime, double drudeTemperature, double drudeCouplingTime, double stepSize, int loopsPerStep=1, int numNHChains=3, int useDrudeNHChains=True, int useCOMTempGroup=True) ;

   double getTemperature() const ;
   void setTemperature(double temp) ;
   double getCouplingTime() const ;
   void setCouplingTime(double tau) ;
   double getDrudeTemperature() const ;
   void setDrudeTemperature(double temp) ;
   double getDrudeCouplingTime() const ;
   void setDrudeCouplingTime(double tau) ;
   double getMaxDrudeDistance() const ;
   void setMaxDrudeDistance(double distance) ;
   virtual void step(int steps) ;
   int getLoopsPerStep() const ;
   void setLoopsPerStep(int loops) ;
   int getNumNHChains() const ;
   void setNumNHChains(int numChains) ;
   int getUseDrudeNHChains() const ;
   void setUseDrudeNHChains(int useDrudeNHChains) ;
   int getUseCOMTempGroup() const ;
   void setUseCOMTempGroup(int useCOMTempGroup) ;
   int getNumTempGroups() const ;
   int addTempGroup() ;
   int addParticleTempGroup(int tempGroup) ;
   void setParticleTempGroup(int particle, int tempGroup) ;

   int addParticleLangevin(int particle) ;
   int getRandomNumberSeed() const ;
   void setRandomNumberSeed(int seed) ;
   int getFriction() const ;
   void setFriction(double fric) ;
   int getDrudeFriction() const ;
   void setDrudeFriction(int fric) ;

   int addImagePair(int, int) ;
   void setMirrorLocation(double) ;
   double getMirrorLocation() const ;
   void addParticleElectrolyte(int) ;
   void setElectricField(double) ;
   double getElectricField() const ;

   void setCosAcceleration(double) ;
   double getCosAcceleration() const ;
   std::vector<double> getViscosity();

   %apply int& OUTPUT {int& tempGroup};
   void getParticleTempGroup(int particle, int& tempGroup) const;
   %clear int& tempGroup;

};

}
