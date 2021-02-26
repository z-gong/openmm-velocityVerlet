Two types of simulations are demonstrated herein.

Files
=====

* `run-bulk.py` -- the script for simulating pure ionic liquids, from which density and viscosity can be calculated.
* `run-edl.py` -- the script for simulating electrical double layers formed at the interfaces of MoS2 electrodes and ionic liquids.
* `ommhelper` -- python library required by `run-bulk.py` and `run-edl.py`.
* `models` -- the topology, force field parameters and initial configurations of different systems.

Simulation of bulk liquids
==========================

1. NPT simulation of \[Im21\]\[DCA\] with langevin thermostat
```
python3 run-bulk.py --gro models/bulk_Im21/conf.gro --psf models/bulk_Im21/topol.psf --prm models/bulk_Im21/ff.prm -t 333 -p 1 --thermostat langevin -n 1_000_000

```
2. NPT simulation of \[Im21\]\[DCA\] with nose-hoover thermostat. Cosine acceleration applied for viscosity calculation
```
python3 run-bulk.py --gro models/bulk_Im21/conf.gro --psf models/bulk_Im21/topol.psf --prm models/bulk_Im21/ff.prm -t 333 -p 1 --thermostat nose-hoover --cos 0.02 -n 10_000_000
```

Simulation of electrical double layers
======================================

1. NVT simulation of \[Im21\]\[DCA\] under voltage drop of 2 V
```
python3 run-edl.py --gro models/edl_Im21/conf.gro --psf models/edl_Im21/topol.psf --prm models/edl_Im21/ff.prm -t 333 -v 2 -n 100_000_000
```
