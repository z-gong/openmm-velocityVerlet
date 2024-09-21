import os
import openmm.openmm as mm


class CheckpointReporter():
    '''
    CheckpointReporter saves periodic checkpoints of a simulation.
    The checkpoints will overwrite old files -- only the latest three will be kept.
    State XML files can be saved together, in case the checkpoint files are broken.

    Parameters
    ----------
    file : string
        The file to write to.
        Any current contents will be overwritten.
        The latest three checkpoint will be kept with the step appended to the file name.
    reportInterval : int
        The interval (in time steps) at which to write checkpoints.
    xml : string, optional
        If provided, the state will be serialized into XML format and saved together with checkpoint.
        Any current contents will be overwritten.
        The latest three XML files will be kept with the step appended to the file name.
    '''

    def __init__(self, file, reportInterval, xml=None):
        self._reportInterval = reportInterval
        self._file = file
        self._xml = xml

        if type(file) is not str:
            raise Exception('file should be str')

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
        return (steps, True, True, False, False, False)

    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """

        filename = self._file + '_%i' % simulation.currentStep
        with open(filename, 'wb') as out:
            out.write(simulation.context.createCheckpoint())

        file_prev3 = self._file + '_%i' % (simulation.currentStep - 3 * self._reportInterval)
        if os.path.exists(file_prev3):
            os.remove(file_prev3)

        if self._xml is not None:
            xml_name = self._xml + '_%i' % simulation.currentStep
            xml = mm.XmlSerializer.serialize(state)
            with open(xml_name, 'w') as f:
                f.write(xml)

            xml_prev3 = self._xml + '_%i' % (simulation.currentStep - 3 * self._reportInterval)
            if os.path.exists(xml_prev3):
                os.remove(xml_prev3)
