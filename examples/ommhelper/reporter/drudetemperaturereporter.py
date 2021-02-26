import numpy as np
import simtk.openmm as mm
from simtk.openmm.app import Simulation
from ..unit import *

class DrudeTemperatureReporter(object):
    '''
    DrudeTemperatureReporter reports the temperatures of different DOFs in a Drude simulation system.

    The temperatures for three sets of degrees of freedom are reported
    -- molecular center of mass, internal atomic and Drude temperature.
    It's better to set the reportInterval larger than 10000 to avoid performance penalty

    Parameters
    ----------
    file : string
        The file to write to
    reportInterval : int
        The interval (in time steps) at which to write frames
    append : bool
        Whether or not append to existing file
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
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, True, False, False)

    def report(self, simulation, state):
        """Generate a report.

        Parameters
        ----------
        simulation : Simulation
            The Simulation to generate a report for
        state : State
            The current state of the simulation
        """
        system: mm.System = simulation.system
        if not self._hasInitialized:
            self.n_atom = system.getNumParticles()
            self.molecules: [[int]] = [list(atoms) for atoms in simulation.context.getMolecules()]
            self.n_mol = len(self.molecules)
            self.mol_atoms = np.zeros(self.n_atom, dtype=int)  # record which molecule the atoms are in
            self.mass_molecules = np.zeros(self.n_mol) # record the mass of molecules
            for i, atoms in enumerate(self.molecules):
                for atom in atoms:
                    self.mol_atoms[atom] = i
                    self.mass_molecules[i] += system.getParticleMass(atom).value_in_unit(unit.dalton)

            self.dof_com = np.count_nonzero(self.mass_molecules) * 3
            self.dof_atom = self.dof_drude = 0
            for i in range(self.n_atom):
                if system.getParticleMass(i) > 0 * unit.dalton:
                    self.dof_atom += 3
            self.dof_atom -= (self.dof_com + system.getNumConstraints())

            if any(type(f) == mm.CMMotionRemover for f in system.getForces()):
                self.dof_com -= 3

            drude_set = set()
            self.pair_set = set()
            force = next(f for f in system.getForces() if type(f) == mm.DrudeForce)
            self.dof_atom -= 3 * force.getNumParticles()
            self.dof_drude += 3 * force.getNumParticles()
            for i in range(force.getNumParticles()):
                i_drude, i_core = force.getParticleParameters(i)[:2]
                drude_set.add(i_drude)
                self.pair_set.add((i_drude, i_core))
            self.drude_array = np.array(list(drude_set))
            self.atom_array = np.array(list(set(range(self.n_atom)) - drude_set))

            self._hasInitialized = True
            print('#"Step"\t"T_COM"\t"T_Atom"\t"T_Drude"\t"KE_COM"\t"KE_Atom"\t"KE_Drude"', file=self._out)

        velocities = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
        masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(self.n_atom)])

        vel_mol = np.zeros([self.n_mol, 3])
        for i, atoms in enumerate(self.molecules):
            if self.mass_molecules[i] == 0:
                continue
            mv = masses[atoms][:, np.newaxis] * velocities[atoms]
            vel_mol[i] = np.sum(mv, axis=0) / self.mass_molecules[i]
        mvv_com = self.mass_molecules * np.sum(vel_mol ** 2, axis=1)
        ke_com = mvv_com.sum() / 2 * (unit.nanometer / unit.picosecond) ** 2 * unit.dalton
        t_com = (2 * ke_com / (self.dof_com * unit.MOLAR_GAS_CONSTANT_R))

        velocities -= np.array([vel_mol[self.mol_atoms[i]] for i in range(self.n_atom)])
        for i_drude, i_core in self.pair_set:
            v_drude = velocities[i_drude]
            v_core = velocities[i_core]
            m_drude = masses[i_drude]
            m_core = masses[i_core]
            m_com = m_drude + m_core
            m_rel = m_drude * m_core / m_com
            v_com = (m_drude * v_drude + m_core * v_core) / m_com
            v_rel = v_drude - v_core
            velocities[i_drude] = v_rel
            velocities[i_core] = v_com
            masses[i_drude] = m_rel
            masses[i_core] = m_com
        mvv = masses * np.sum(velocities ** 2, axis=1)
        ke = mvv[self.atom_array].sum() / 2 * (unit.nanometer / unit.picosecond) ** 2 * unit.dalton
        ke_drude = mvv[self.drude_array].sum() / 2 * (unit.nanometer / unit.picosecond) ** 2 * unit.dalton
        t = (2 * ke / (self.dof_atom * unit.MOLAR_GAS_CONSTANT_R))
        t_drude = (2 * ke_drude / (self.dof_drude * unit.MOLAR_GAS_CONSTANT_R))
        print(simulation.currentStep,
              t_com.value_in_unit(kelvin), t.value_in_unit(kelvin), t_drude.value_in_unit(kelvin),
              ke_com.value_in_unit(kJ_mol), ke.value_in_unit(kJ_mol), ke_drude.value_in_unit(kJ_mol),
              sep='\t', file=self._out)

        if hasattr(self._out, 'flush') and callable(self._out.flush):
            self._out.flush()

    def __del__(self):
        self._out.close()
