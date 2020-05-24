# openmm-velocityVerlet
Lammps style velocity-Verlet integrator and modifiers plugin for OpenMM

This plugin works with OpenMM to perform velocity-Verlet integration,
along with a series of simulation methods that can not be done easily with `CustomIntegrator`.

Currently, this plugin enables
- Nose-Hoover thermostat
- Temperature-grouped Nose-Hoover thermostat for Drude polarizable model
- Langevin thermostat for both non-polarizable and Drude polarizable model
- Periodic perturbation method for viscosity calculation
- Image charge method for constant voltage simulation
- External electric field
- Middle discretization scheme

### Nose-Hoover thermostat
The temperature-grouped Nose-Hoover (TGNH) thermostat in this plugin is borrowed from
[scychon's openmm_drudeNose](https://github.com/scychon/openmm_drudeNose).
TGNH is suitable for Drude model where dynamic properties are desired.
The naive dual-Nose-Hoover thermostat for Drude model displays serious thermo-equipartition problem,
resulting in ridiculously high molecular translational temperature.
One should **NEVER** use the naive dual-Nose-Hoover thermostat for Drude model because it's simply **WRONG**.
The TGNH thermostat proposed by [Chang Yun Son et. al.](https://pubs.acs.org/doi/10.1021/acs.jpclett.9b02983)
thermolizes molecular translational degrees of freedom separately from atomic motions
to ensure that the average temperature of molecular motion equals to preset value.

#### usage
```python
from velocityverletplugin import VVIntegrator
from simtk.unit import kelvin as K, picosecond as ps
...
integrator = VVIntegrator(300 * K, 10 / ps, 1 * K, 40 / ps, 0.001 * ps)
...
```

### Langevin thermostat
OpenMM natively supports Langevin thermostat.
However, one cannot apply several thermostats in one simulation,
which is not ideal for very heterogeneous system.
For example, one want to simulate solid-liquid interface,
this plugin can apply Nose-Hoover thermostat on liquid molecules,
whereas Langevin thermostat on solid particles.

#### usage
```python
from velocityverletplugin import VVIntegrator
from simtk.unit import kelvin as K, picosecond as ps
...
integrator = VVIntegrator(300 * K, 10 / ps, 1 * K, 40 / ps, 0.001 * ps)
for i in atoms_solid:
    integrator.addParticleLangevin(i)
integrator.setFriction(5.0 / ps)
integrator.setDrudeFriction(20 / ps)
...
```

### Periodic perturbation method 
[Periodic perturbation method](https://aip.scitation.org/doi/10.1063/1.1421362)
is an efficient approach for viscosity calculation.
One applies cosine acceleration to liquid,
the viscosity can be obtained from the generated velocity profile.
The method should be used together with TGNH thermostat.
One should be careful about the acceleration strength and data extraction.
Read [this article](https://doi.org/10.1021/acs.jced.9b00050)
and [this article](https://www.sciencedirect.com/science/article/abs/pii/S0378381219302638)
for more information.

#### usage
```python
from velocityverletplugin import VVIntegrator
from simtk.unit import kelvin as K, picosecond as ps, nanosecond as ns
...
integrator = VVIntegrator(300 * K, 10 / ps, 1 * K, 40 / ps, 0.001 * ps)
integrator.setCosAcceleration(0.01 * nm/ps**2)
...
print(integrator.getViscosity())
```

### Image charge method
[Image charge method](https://pubs.acs.org/doi/10.1021/acs.jpcc.9b06635)
is an efficient approach to enforce constant voltage constraint for planar electrodes.

#### usage
```python
from velocityverletplugin import VVIntegrator
from simtk.unit import kelvin as K, picosecond as ps, nanosecond as ns
...
integrator = VVIntegrator(300 * K, 10 / ps, 1 * K, 40 / ps, 0.001 * ps)
for i_image, i_parent in zip(atoms_image, atoms_parent):
    integrator.addImagePair(i_image, i_parent)
integrator.setMirrorLocation(length_box / 2)
...
```

### External electric field
An external electric field along `z` direction can be applied for selected particles.
Note that external electric field can also be done by using `CustomExternalForce`.

#### usage
```python
from velocityverletplugin import VVIntegrator
from simtk.unit import kelvin as K, picosecond as ps, nanosecond as ns, volt
...
integrator = VVIntegrator(300 * K, 10 / ps, 1 * K, 40 / ps, 0.001 * ps)
for i in electrolytes:
    integrator.addParticleElectrolyte(i)
integrator.setElectricField(1.0 * volt/nm)
...
```

### Middle discretization scheme
Use middle scheme to integrate the position and momentum of particles.
With vanilla velocity-Verlet integrator, two Nose-Hoover coupling is required per step.
With middle discretization scheme, only one NH coupling is required at each step.
Therefore, it is around 20 % faster.

####
```python
from velocityverletplugin import VVIntegrator
from simtk.unit import kelvin as K, picosecond as ps
...
integrator = VVIntegrator(300 * K, 10 / ps, 1 * K, 40 / ps, 0.001 * ps)
integrator.setUseMiddleScheme(True)
...
```

## Known issues

- The external electric field should be used with caution.
One should apply the electric field to all particles or none of them of same molecules types.
It's a bit confusing. Imaging a system made of 100 A molecules and 100 B molecules.
One can apply electric field to only 100 A molecules, or to only 100 B molecules, or to all of them.
But one **CANNOT** apply electric field to only 50 A molecules.
This has something to do with the reordering of atoms, which will not be fixed easily.
So if you need to handle this kind of situation, you are recommended to use `CustomExternalForce`.
