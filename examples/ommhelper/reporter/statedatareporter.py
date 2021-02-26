"""
statedatareporter.py: Outputs data about a simulation

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2012-2013 Stanford University and the Authors.
Authors: Peter Eastman
Contributors: Robert McGibbon, Zheng Gong

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from __future__ import absolute_import
from __future__ import print_function

__author__ = "Peter Eastman"
__version__ = "1.0"

try:
    import bz2

    have_bz2 = True
except:
    have_bz2 = False

try:
    import gzip

    have_gzip = True
except:
    have_gzip = False

import simtk.openmm as mm
import simtk.unit as unit
import math
import time


class StateDataReporter(object):
    '''
    StateDataReporter outputs information about a simulation, such as energy and temperature, to a file.

    This reporter is modified from the StateDataReporter shipped with OpenMM python API.
    It adds support for reporting box size and collective variables.

    To use it, create a StateDataReporter, then add it to the Simulation's list of reporters.  The set of
    data to write is configurable using boolean flags passed to the constructor.  By default the data is
    written in comma-separated-value (CSV) format, but you can specify a different separator to use.

    Parameters
    ----------
    file : string or file
        The file to write to, specified as a file name or file object
    reportInterval : int
        The interval (in time steps) at which to write frames
    step : bool=False
        Whether to write the current step index to the file
    time : bool=False
        Whether to write the current time to the file
    potentialEnergy : bool=False
        Whether to write the potential energy to the file
    kineticEnergy : bool=False
        Whether to write the kinetic energy to the file
    totalEnergy : bool=False
        Whether to write the total energy to the file
    temperature : bool=False
        Whether to write the instantaneous temperature to the file
    volume : bool=False
        Whether to write the periodic box volume to the file
    box: bool=True
        Whether to write the periodic box size to the file
    density : bool=False
        Whether to write the system density to the file
    progress : bool=False
        Whether to write current progress (percent completion) to the file.
        If this is True, you must also specify totalSteps.
    remainingTime : bool=False
        Whether to write an estimate of the remaining clock time until
        completion to the file.  If this is True, you must also specify
        totalSteps.
    speed : bool=False
        Whether to write an estimate of the simulation speed in ns/day to
        the file
    elapsedTime : bool=False
        Whether to write the elapsed time of the simulation in hours to
        the file.
    separator : string=','
        The separator to use between columns in the file
    systemMass : mass=None
        The total mass to use for the system when reporting density.  If
        this is None (the default), the system mass is computed by summing
        the masses of all particles.  This parameter is useful when the
        particle masses do not reflect their actual physical mass, such as
        when some particles have had their masses set to 0 to immobilize
        them.
    totalSteps : int=None
        The total number of steps that will be included in the simulation.
        This is required if either progress or remainingTime is set to True,
        and defines how many steps will indicate 100% completion.
    cvs : list=[]
        Collective variables defined by CustomCVForce
    '''

    def __init__(self, file, reportInterval, step=True, time=True, potentialEnergy=True,
                 kineticEnergy=False, totalEnergy=False, temperature=True, volume=False, box=True,
                 density=True, progress=False, remainingTime=False, speed=True, elapsedTime=True,
                 separator='\t', systemMass=None, totalSteps=None, append=False, cvs=[]):
        self._reportInterval = reportInterval
        self._openedFile = isinstance(file, str)
        if (progress or remainingTime) and totalSteps is None:
            raise ValueError(
                'Reporting progress or remaining time requires total steps to be specified')
        if self._openedFile:
            if append:
                self._out = open(file, 'a')
            else:
                self._out = open(file, 'w')
        else:
            self._out = file
        self._step = step
        self._time = time
        self._potentialEnergy = potentialEnergy
        self._kineticEnergy = kineticEnergy
        self._totalEnergy = totalEnergy
        self._temperature = temperature
        self._volume = volume
        self._box = box
        self._density = density
        self._progress = progress
        self._remainingTime = remainingTime
        self._speed = speed
        self._elapsedTime = elapsedTime
        self._separator = separator
        self._totalMass = systemMass
        self._totalSteps = totalSteps
        self._hasInitialized = False
        self._needsPositions = False
        self._needsVelocities = False
        self._needsForces = False
        self._needEnergy = potentialEnergy or kineticEnergy or totalEnergy or temperature

        self._boxSizeList = [[], [], []]

        self._cvs = cvs

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        tuple
            A five element tuple. The first element is the number of steps
            until the next report. The remaining elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (
        steps, self._needsPositions, self._needsVelocities, self._needsForces, self._needEnergy)

    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        if not self._hasInitialized:
            self._initializeConstants(simulation)
            headers = self._constructHeaders()
            print('#"%s"' % ('"' + self._separator + '"').join(headers), file=self._out)
            try:
                self._out.flush()
            except AttributeError:
                pass
            self._initialClockTime = time.time()
            self._initialSimulationTime = state.getTime()
            self._initialSteps = simulation.currentStep
            self._hasInitialized = True

        # Check for errors.
        self._checkForErrors(simulation, state)

        # Query for the values
        values = self._constructReportValues(simulation, state)

        # Write the values.
        print(self._separator.join(str(v) for v in values), file=self._out)
        try:
            self._out.flush()
        except AttributeError:
            pass

    def _constructReportValues(self, simulation, state):
        """Query the simulation for the current state of our observables of interest.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation

        Returns
        -------
        A list of values summarizing the current state of
        the simulation, to be printed or saved. Each element in the list
        corresponds to one of the columns in the resulting CSV file.
        """
        values = []
        box = state.getPeriodicBoxVectors()
        volume = box[0][0] * box[1][1] * box[2][2]
        clockTime = time.time()
        if self._progress:
            values.append('%.1f%%' % (100.0 * simulation.currentStep / self._totalSteps))
        if self._step:
            values.append(simulation.currentStep)
        if self._time:
            values.append('%.4f' % state.getTime().value_in_unit(unit.picosecond))
        if self._potentialEnergy:
            values.append(
                '%.4f' % state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
        if self._kineticEnergy:
            values.append('%.4f' % state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole))
        if self._totalEnergy:
            values.append(
                '%.4f' % (state.getKineticEnergy() + state.getPotentialEnergy()).value_in_unit(
                    unit.kilojoules_per_mole))
        if self._temperature:
            values.append('%.2f' % (2 * state.getKineticEnergy() / (
                        self._dof * unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin))
        if self._volume:
            values.append('%.4f' % volume.value_in_unit(unit.nanometer ** 3))
        if self._box:
            values.append('%.4f' % box[0][0].value_in_unit(unit.nanometer))
            values.append('%.4f' % box[1][1].value_in_unit(unit.nanometer))
            values.append('%.4f' % box[2][2].value_in_unit(unit.nanometer))
        if self._density:
            values.append('%.4f' % (self._totalMass / volume).value_in_unit(
                unit.gram / unit.item / unit.milliliter))
        if self._speed:
            elapsedDays = (clockTime - self._initialClockTime) / 86400.0
            elapsedNs = (state.getTime() - self._initialSimulationTime).value_in_unit(
                unit.nanosecond)
            if elapsedDays > 0.0:
                values.append('%.3g' % (elapsedNs / elapsedDays))
            else:
                values.append('--')
        if self._elapsedTime:
            values.append('%.3g' % ((time.time() - self._initialClockTime) / 3600.0))
        if self._remainingTime:
            elapsedSeconds = clockTime - self._initialClockTime
            elapsedSteps = simulation.currentStep - self._initialSteps
            if elapsedSteps == 0:
                value = '--'
            else:
                estimatedTotalSeconds = (
                                                    self._totalSteps - self._initialSteps) * elapsedSeconds / elapsedSteps
                remainingSeconds = int(estimatedTotalSeconds - elapsedSeconds)
                remainingDays = remainingSeconds // 86400
                remainingSeconds -= remainingDays * 86400
                remainingHours = remainingSeconds // 3600
                remainingSeconds -= remainingHours * 3600
                remainingMinutes = remainingSeconds // 60
                remainingSeconds -= remainingMinutes * 60
                if remainingDays > 0:
                    value = "%d:%d:%02d:%02d" % (
                    remainingDays, remainingHours, remainingMinutes, remainingSeconds)
                elif remainingHours > 0:
                    value = "%d:%02d:%02d" % (remainingHours, remainingMinutes, remainingSeconds)
                elif remainingMinutes > 0:
                    value = "%d:%02d" % (remainingMinutes, remainingSeconds)
                else:
                    value = "0:%02d" % remainingSeconds
            values.append(value)
        for cv in self._cvs:
            values.append(cv.getCollectiveVariableValues(simulation.context)[0])

        self._boxSizeList[0].append(box[0][0].value_in_unit(unit.nanometer)),
        self._boxSizeList[1].append(box[1][1].value_in_unit(unit.nanometer)),
        self._boxSizeList[2].append(box[2][2].value_in_unit(unit.nanometer)),

        return values

    def _initializeConstants(self, simulation):
        """Initialize a set of constants required for the reports

        Parameters
        - simulation (Simulation) The simulation to generate a report for
        """
        system = simulation.system
        if self._temperature:
            # Compute the number of degrees of freedom.
            dof = 0
            for i in range(system.getNumParticles()):
                if system.getParticleMass(i) > 0 * unit.dalton:
                    dof += 3
            dof -= system.getNumConstraints()
            if any(type(system.getForce(i)) == mm.CMMotionRemover for i in
                   range(system.getNumForces())):
                dof -= 3
            self._dof = dof
        if self._density:
            if self._totalMass is None:
                # Compute the total system mass.
                self._totalMass = 0 * unit.dalton
                for i in range(system.getNumParticles()):
                    self._totalMass += system.getParticleMass(i)
            elif not unit.is_quantity(self._totalMass):
                self._totalMass = self._totalMass * unit.dalton

    def _constructHeaders(self):
        """Construct the headers for the CSV output

        Returns: a list of strings giving the title of each observable being reported on.
        """
        headers = []
        if self._progress:
            headers.append('Progress')
        if self._step:
            headers.append('Step')
        if self._time:
            headers.append('Time')
        if self._potentialEnergy:
            headers.append('E_potential')
        if self._kineticEnergy:
            headers.append('E_kinetic')
        if self._totalEnergy:
            headers.append('E_total')
        if self._temperature:
            headers.append('T')
        if self._volume:
            headers.append('Vol')
        if self._box:
            headers.append('Lx')
            headers.append('Ly')
            headers.append('Lz')
        if self._density:
            headers.append('Density')
        if self._speed:
            headers.append('Speed')
        if self._elapsedTime:
            headers.append('Elapsed')
        if self._remainingTime:
            headers.append('Remaining')
        for i, cv in enumerate(self._cvs):
            headers.append('CV%i' % i)
        return headers

    def _checkForErrors(self, simulation, state):
        """Check for errors in the current state of the simulation

        Parameters
         - simulation (Simulation) The Simulation to generate a report for
         - state (State) The current state of the simulation
        """
        if self._needEnergy:
            energy = (state.getKineticEnergy() + state.getPotentialEnergy()).value_in_unit(
                unit.kilojoules_per_mole)
            if math.isnan(energy):
                raise ValueError('Energy is NaN')
            if math.isinf(energy):
                raise ValueError('Energy is infinite')

    def __del__(self):
        if self._openedFile:
            self._out.close()

    def getBoxSizeAverage(self, timeFraction=0.5):
        '''
        Get the averaged box size of the simulated system since this reporter was added.

        Parameters
        ----------
        timeFraction : float
            The fraction of data (in the end) used to calculate the average.
            e.g. If set to 0.5, then the last half of the box size data will be used for average.

        Returns
        -------
        lx : float
        ly : float
        lz : float
        '''
        n = math.ceil(len(self._boxSizeList[0]) * timeFraction)
        lx = sum(self._boxSizeList[0][-n:]) / n
        ly = sum(self._boxSizeList[1][-n:]) / n
        lz = sum(self._boxSizeList[2][-n:]) / n
        return (lx, ly, lz)
