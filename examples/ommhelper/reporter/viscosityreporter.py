from simtk.openmm import app
from ..unit import *


class ViscosityReporter(object):
    '''
    ViscosityReporter report the viscosity using cosine periodic perturbation method.
    A integrator supporting this method is required.
    e.g. the VVIntegrator from https://github.com/z-gong/openmm-velocityVerlet.

    Parameters
    ----------
    file : string
        The file to write to
    reportInterval : int
        The interval (in time steps) at which to write frames
    append : bool
        Whether or not append to the existing file.
    '''

    def __init__(self, file, reportInterval, append=False):
        self._reportInterval = reportInterval
        if append:
            self._out = open(file, 'a')
        else:
            self._out = open(file, 'w')
        self._hasInitialized = False

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for

        Returns
        -------
        tuple
            A six element tuple. The first element is the number of steps
            until the next report. The next four elements specify whether
            that report will require positions, velocities, forces, and
            energies respectively.  The final element specifies whether
            positions should be wrapped to lie in a single periodic box.
        """
        try:
            simulation.integrator.getCosAcceleration()
        except AttributeError:
            raise Exception('This integrator does not calculate viscosity')

        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False)

    def report(self, simulation: app.Simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        if not self._hasInitialized:
            self._hasInitialized = True
            print('#"Step"\t"Acceleration (nm/ps^2)"\t"VelocityAmplitude (nm/ps)"\t"1/Viscosity (1/Pa.s)"', file=self._out)

        acceleration = simulation.integrator.getCosAcceleration().value_in_unit(nm / ps ** 2)
        vMax, invVis = simulation.integrator.getViscosity()
        vMax = vMax.value_in_unit(nm / ps)
        invVis = invVis.value_in_unit((unit.pascal * unit.second) ** -1)
        print(simulation.currentStep, acceleration, vMax, invVis, sep='\t', file=self._out)

        if hasattr(self._out, 'flush') and callable(self._out.flush):
            self._out.flush()

    def __del__(self):
        self._out.close()
