import sys
from simtk.openmm.app.gromacsgrofile import GromacsGroFile
from simtk.unit import nanometer, picosecond, norm, is_quantity


class GroFile(GromacsGroFile):
    '''
    GroFile is a parser and writer for Gromacs .gro file.

    GroFile extends GromacsGroFile from OpenMM python API by adding writing ability.

    Parameters
    ----------
    file : string
        the name of the file to load
    '''

    @staticmethod
    def writeFile(topology, positions, vectors, file, time=None, subset=None, velocities=None):
        '''
        Write positions (and optionally velocities) of atoms into a GRO file

        It is possible to write a subset of atoms into the GRO file by providing `subset` argument.

        Parameters
        ----------
        topology : simtk.openmm.app.Topology
        positions : array_like of shape (n_atom, 3)
            The length of positions should equal to the number of atoms in the topology, even when subset is provided.
        vectors :  array_like of shape (3, 3)
            The full box vectors.
        file : str or FileIO
        time : float
        subset : list of int, optional
            If not provided, then all atoms will be written.
        velocities array_like of shape (n_atom, 3), optional
            The length of velocities should equal to the number of atoms in the topology, even when subset is provided.
            If not provided, then velocities information will not be written.
        '''
        if type(file) is str:
            _file = open(file, 'w')
        else:
            _file = file

        GroFile._writeHeader(time, _file)
        GroFile._writeModel(topology, positions, _file, subset, velocities)
        GroFile._writeFooter(vectors, _file)

        if type(file) is str:
            _file.close()

    @staticmethod
    def _writeHeader(time=None, file=sys.stdout):
        """Write out the header for a PDB file.

        Parameters
        ----------
        topology : Topology
            The Topology defining the molecular system being written
        file : file=stdout
            A file to write the file to
        """
        if time is None:
            time = 0.0
        elif is_quantity(time):
            time = time.value_in_unit(picosecond)

        print("written by openmm t = %.3f ps" % time, file=file)

    @staticmethod
    def _writeModel(topology, positions, file=sys.stdout, subset=None, velocities=None):
        """Write out a model to a PDB file.

        Parameters
        ----------
        topology : Topology
            The Topology defining the model to write
        positions : list
            The list of atomic positions to write
        file : file=stdout
            A file to write the model to
        subset : list(int)=None
            If not None, only the selected atoms will be written
        """

        atoms = list(topology.atoms())
        if len(atoms) != len(positions):
            raise ValueError('The number of positions must match the number of atoms')
        if is_quantity(positions):
            positions = positions.value_in_unit(nanometer)
        if velocities is not None:
            if len(atoms) != len(velocities):
                raise ValueError('The number of velocities must match the number of atoms')
            if is_quantity(velocities):
                velocities = velocities.value_in_unit(nanometer / picosecond)

        if subset is None:
            subset = list(range(len(atoms)))

        print('%i' % len(subset), file=file)
        for ii, i in enumerate(subset):
            atom = atoms[i]
            residue = atom.residue
            coords = positions[i]
            # writing atom symbol instead of name makes visualization easier
            if atom.element is not None:
                name = atom.element.symbol
            else:
                name = ''.join(i for i in atom.name if not i.isdigit())
            line = '%5i%5s%5s%5i%8.3f%8.3f%8.3f' % (
                (residue.index + 1) % 100000, residue.name[:5], name[:5],
                (atom.index + 1) % 100000, coords[0], coords[1], coords[2])
            if velocities is not None:
                vel = velocities[i]
                line += '%8.4f%8.4f%8.4f' % (vel[0], vel[1], vel[2])
            print(line, file=file)

    @staticmethod
    def _writeFooter(periodicBoxVectors, file=sys.stdout):
        """Write out the footer for a PDB file.

        Parameters
        ----------
        topology : Topology
            The Topology defining the molecular system being written
        file : file=stdout
            A file to write the file to
        """
        vectors = periodicBoxVectors.value_in_unit(nanometer)
        xx, xy, xz = vectors[0]
        yx, yy, yz = vectors[1]
        zx, zy, zz = vectors[2]
        print(' %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f' % (
            xx, yy, zz, xy, xz, yx, yz, zx, zy), file=file)
