import math
from .. import GroFile

class GroReporter(object):
    '''
    GroReporter outputs a series of frames from a Simulation to a GRO file.

    Parameters
    ----------
    file : string
        The file to write to
    reportInterval : int
        The interval (in time steps) at which to write frames
    logarithm : bool
        If set to True, then write trajectory at logarithm interval.
        reportInterval will be the minimum step for reporting.
        e.g. when reportInterval set to 30, then report at [30, 40, 50, ..., 90, 100, 200, ..., 900, 1000, 2000, ...] steps.
    enforcePeriodicBox: bool
        Specifies whether particle positions should be translated
        so the center of every molecule lies in the same periodic box.
        If None (the default), it will automatically decide whether to translate molecules
        based on whether the system being simulated uses periodic boundary conditions.
    subset : list(int)=None
        If not None, only the selected atoms will be written
    reportVelocity: bool
        If set to True, velocities will be reported
    append: bool
        If set to True, will append to file
    '''

    def __init__(self, file, reportInterval, logarithm=False, enforcePeriodicBox=False, subset=None, reportVelocity=False, append=False):
        self._reportInterval = reportInterval
        self._logarithm = logarithm
        self._enforcePeriodicBox = enforcePeriodicBox
        if append:
            self._out = open(file, 'a')
        else:
            self._out = open(file, 'w')
        self._reportVelocity = reportVelocity

        if subset is None:
            self._subset = None
        else:
            self._subset = subset[:]

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
        if self._logarithm:
            if simulation.currentStep < self._reportInterval:
                _base = self._reportInterval
            else:
                _base = 10 ** math.floor(math.log10(simulation.currentStep))
            steps = _base - simulation.currentStep % _base
        else:
            steps = self._reportInterval - simulation.currentStep % self._reportInterval

        return (steps, True, self._reportVelocity, False, False, self._enforcePeriodicBox)

    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        time = state.getTime()
        positions = state.getPositions(asNumpy=True)
        velocities = state.getVelocities(asNumpy=True) if self._reportVelocity else None
        vectors = state.getPeriodicBoxVectors()
        GroFile.writeFile(simulation.topology, positions, vectors, self._out, time, self._subset, velocities)

        if hasattr(self._out, 'flush') and callable(self._out.flush):
            self._out.flush()

    def __del__(self):
        self._out.close()
