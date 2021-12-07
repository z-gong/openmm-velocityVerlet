velocity-Verlet plugin for OpenMM
=================================

This plugin works with OpenMM to perform velocity-Verlet integration,
along with various simulation methods that can not be done easily with `CustomIntegrator`.

Currently, this plugin enables
- Nose-Hoover (NH) thermostat
- Temperature-grouped Nose-Hoover (TGNH) thermostat for Drude polarizable model
- Langevin thermostat for both non-polarizable and Drude polarizable model
- Periodic perturbation method for viscosity calculation
- Image charge method for constant voltage simulation
- External electric field
- Middle discretization scheme

### Note: This plugin works with OpenMM 7.4.2. Its compatibility with newer version of OpenMM is not tested.

Installation
============

This project uses [CMake](http://www.cmake.org) for its build system.
Currently, only CUDA platform is implemented. To build it, follow these steps:

1. Create a directory in which to build the plugin.

2. Run the CMake GUI or ccmake, specifying your new directory as the build directory and the top
level directory of this project as the source directory.

3. Press "Configure".

4. Set OPENMM_DIR to point to the directory where OpenMM is installed.  This is needed to locate
the OpenMM header files and libraries.

5. Set CMAKE_INSTALL_PREFIX to the directory where the plugin should be installed.  Usually,
this will be the same as OPENMM_DIR, so the plugin will be added to your OpenMM installation.

7. Make sure that CUDA_TOOLKIT_ROOT_DIR is set correctly and that VELOCITYVERLET_BUILD_CUDA_LIB is selected.

8. Press "Configure" again if necessary, then press "Generate".

9. Use the build system you selected to build and install the plugin.  For example, if you
selected Unix Makefiles, type `make install`.

10. To build and install the Python API, build the "PythonInstall" target, for example by typing
 `make PythonInstall`.


Usage
=====

### Nose-Hoover and temperature-grouped Nose-Hoover thermostat
NH thermostat is implemented in this plugin to enable accurate calculation of dynamic properties,
like viscosity, diffusion coefficient, auto correlation function, etc...
For Drude model integrated with extended-Lagrangian method, however,
the naive dual-NH thermostat presents serious thermo-equipartition problem,
resulting in ridiculously high molecular translational temperature.
Therefore, the TGNH thermostat proposed by [Chang Yun Son et. al.](https://pubs.acs.org/doi/10.1021/acs.jpclett.9b02983)
is also implemented, which thermolizes molecular center of mass (COM) motions separately from atomic motions relative to COMs
to ensure that the average temperature of molecular motion equals to preset value.

When a `VVIntegrator` is bounded to a `System`, the thermostat to use will be determined automatically. 
If there is Drude particles in the system (there is `DrudeForce` and there are particles been added to this `DrudeForce`),
then the TGNH thermostat will be applied.
Otherwise, the NH thermostat will be applied.

```python
from velocityverletplugin import VVIntegrator
from simtk.unit import kelvin as K, picosecond as ps
...
# Integrate the system with velocity-Verlet algorithm at time step of 0.001 ps.
# The system will be thermolized at 300 K with collision frequency of 10 /ps.
# If there are Drude particles, the Drude relative motions will be thermostated at 1 K with collion frequency of 40 /ps.
integrator = VVIntegrator(300 * K, 10 / ps, 1 * K, 40 / ps, 0.001 * ps)
...
```

### Langevin thermostat
OpenMM natively supports Langevin thermostat.
However, it cannot be applied to only a part of the system.
This plugin can apply Langevin thermostat to selected particles in the system,
while the other particles are still thermolized with NH or TGNH thermostat,
which is more suitable for highly heterogeneous system.
For example, in the simulation of solid-liquid interface,
this plugin enables applying NH thermostat on liquid molecules,
whereas Langevin thermostat on solid particles.

When a particle get added to the Langevin thermostat, it will be removed from the NH thermostat automatically.
Note that for Drude polarizable model, one single molecule should be thermolized by either TGNH or Langevin thermostat.
It is invalid to apply Langevin thermostat on part of a single molecule, because TGNH thermostat will thermolize the molecular COM motions.

The friction used for Langevin thermostat will be determined automatically.
If there is Drude particles in the system, the friction for atoms and Drude particles will be 5 /ps and 20 /ps, respectively.
Otherwise, the friction for atoms will be 1 /ps.
If you are not happy with the default value,
the fraction can be set explicitly by calling `integrator.setFriction()` and `integrator.setDrudeFriction()`.

```python
from velocityverletplugin import VVIntegrator
from simtk.unit import kelvin as K, picosecond as ps
...
integrator = VVIntegrator(300 * K, 10 / ps, 1 * K, 40 / ps, 0.001 * ps)
# Thermolize selected particles with Langevin thermostat.
# The remains particles will still be thermolzied by NH or TGNH thermostat.
for i in atoms_solid:
    integrator.addParticleLangevin(i)
...
```

### Periodic perturbation method 
Periodic perturbation method is an efficient approach for viscosity calculation.
A cosine-shaped acceleration is applied to liquid, which will introduce a velocity gradient.
Then the viscosity can be calculated from the ensemble average of the generated velocity profile.
This method should be used together with NH or TGNH thermostat for non-polarizable or Drude polarizable system, respectively.
During each step, the collective velocity will be removed before thermostating, and be added back after the thermostating.
Therefore, the temperature will be correctly maintained.
However, the temperature printed out by `StateDataReporter` may be higher then the correct value.
This is expected, because `StateDataReporter` will treat the collective motions as thermal motion when calculating the temperature.

The calculated viscosity may depend on the value of acceleration.
Therefore, the acceleration strength should be determined carefully.
Refer to [this article](https://aip.scitation.org/doi/10.1063/1.1421362),
[this article](https://doi.org/10.1021/acs.jced.9b00050)
and [this article](https://www.sciencedirect.com/science/article/abs/pii/S0378381219302638)
for more information.

```python
from velocityverletplugin import VVIntegrator
from simtk.unit import kelvin as K, picosecond as ps, nanosecond as ns
...
integrator = VVIntegrator(300 * K, 10 / ps, 1 * K, 40 / ps, 0.001 * ps)
# Apply cosine-shaped acceleration to all particles in the system.
integrator.setCosAcceleration(0.01 * nm/ps**2)
...
# Print the reciprical viscosity and velocity amplitude at current step.
# In order to calculate the ensemble average, its better to write a reporter to save it at fixed time interval.
print(integrator.getViscosity())
```

### Image charge method
Image charge method is an efficient approach to enforce constant voltage drop between two parallel electrodes.
It requires that two electrodes are planar and ideal conductor.
In order to use this method, the image particles should have already been added to the system with correct positions and charges.
The two electrodes are assumed to be in the `xy` plane and their `z` coordinates should be restrained during the simulation.
The length of the simulation box in `z` direction should be twice the distance between two electrodes.
The electrode particle should not carry any charge, because the Coulomb interactions between electrolytes and electrodes will be described by image charges.
Refer to [this article](https://pubs.acs.org/doi/10.1021/acs.jpcc.9b06635) for more details of this method.

```python
from velocityverletplugin import VVIntegrator
from simtk.unit import kelvin as K, picosecond as ps, nanosecond as ns
...
integrator = VVIntegrator(300 * K, 10 / ps, 1 * K, 40 / ps, 0.001 * ps)
# Identify the pairs of image particles and real particles so that the positions of imamge particles can be updated.
for i_image, i_parent in zip(atoms_image, atoms_parent):
    integrator.addImagePair(i_image, i_parent)
# Identify the location of the electrode at the right side.
integrator.setMirrorLocation(length_box / 2)
...
```

### External electric field
An external electric field along `z` direction can be applied to selected particles.
This can be used in combination with the image charge method described above to introduce a constant voltage drop between two electrodes.
Note that external electric field can also be applied by using `CustomExternalForce`.

```python
from velocityverletplugin import VVIntegrator
from simtk.unit import kelvin as K, picosecond as ps, nanosecond as ns, volt
...
integrator = VVIntegrator(300 * K, 10 / ps, 1 * K, 40 / ps, 0.001 * ps)
# Apply extral electric field to selected electrolyte particles.
for i in electrolytes:
    integrator.addParticleElectrolyte(i)
integrator.setElectricField(1.0 * volt / nm)
...
```

_Known issue:_ The external electric field should be applied to all particles or none of them which belongs to same molecules types.
It is a bit confusing. Imaging a system made of 100 A molecules and 100 B molecules.
The electric field can be applied to only 100 A molecules, or to only 100 B molecules, or to all of them.
However, it **CANNOT** be applied to only 50 A molecules.
This has something to do with the mechanism of reordering of atoms.
If you need to handle this kind of situation, you are recommended to use `CustomExternalForce`.

### Middle discretization scheme
Use middle discretization scheme to integrate the position and momentum of particles.
For NH or TGNH thermostat, the middle scheme can provide a performance boost of around 20 %.
Because that with vanilla velocity-Verlet integrator, two NH scaling is required per step,
whereas with middle scheme, only one NH scaling is required at each step.

The middle scheme is enabled by default.
If you prefer to use the original discretization scheme, it can be disabled by `integrator.setUseMiddleScheme(False)`.

```python
from velocityverletplugin import VVIntegrator
from simtk.unit import kelvin as K, picosecond as ps
...
integrator = VVIntegrator(300 * K, 10 / ps, 1 * K, 40 / ps, 0.001 * ps)
integrator.setUseMiddleScheme(True)
...
```

Examples and citation
=====================

Examples from the following work are provided in `examples` to demonstrate the usage of this plugin.
Please cite this article if you find this plugin useful.

[Gong, Z.; Padua, A. A. H. Effect of Side Chain Modifications in Imidazolium Ionic Liquids on the Properties of the Electrical Double Layer at a Molybdenum Disulfide Electrode. J. Chem. Phys. 2021.](https://doi.org/10.1063/5.0040172)


License
=======

Portions copyright (c) 2020 the Authors.

Authors: Zheng Gong

Contributors:

Part of the TGNH code comes from [scychon's openmm_drudeNose](https://github.com/scychon/openmm_drudeNose).

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
