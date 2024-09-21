#!/usr/bin/env python3

import sys
import argparse
from openmm import openmm as mm, app
from ommhelper.unit import *
import ommhelper as oh

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-n', '--nstep', type=int, default=int(1E6), help='number of steps')
parser.add_argument('-t', '--temp', type=float, default=333, help='temperature in Kelvin')
parser.add_argument('-p', '--press', type=float, default=1, help='pressure in bar')
parser.add_argument('--dt', type=float, default=0.001, help='step size in ps')
parser.add_argument('--thermostat', type=str, default='langevin',
                    choices=['langevin', 'nose-hoover'], help='thermostat')
parser.add_argument('--barostat', type=str, default='iso',
                    choices=['no', 'iso', 'semi-iso', 'xyz', 'xy', 'z'], help='barostat')
parser.add_argument('--cos', type=float, default=0,
                    help='cosine acceleration for viscosity calculation')
parser.add_argument('--gro', type=str, default='conf.gro', help='gro file')
parser.add_argument('--psf', type=str, default='topol.psf', help='psf file')
parser.add_argument('--prm', type=str, default='ff.prm', help='prm file')
parser.add_argument('--cpt', type=str, help='load checkpoint')
parser.add_argument('--min', action='store_true', help='minimize energy before simulation')
args = parser.parse_args()


def gen_simulation(gro_file='conf.gro', psf_file='topol.psf', prm_file='ff.prm',
                   dt=0.001, T=300, P=1, tcoupl='langevin', pcoupl='iso',
                   cos=0, restart=None):
    print('Building system...')
    gro = oh.GroFile(gro_file)
    psf = oh.OplsPsfFile(psf_file, periodicBoxVectors=gro.getPeriodicBoxVectors())
    prm = app.CharmmParameterSet(prm_file)
    system = psf.createSystem(prm, nonbondedMethod=app.PME, nonbondedCutoff=1.2 * nm,
                              constraints=app.HBonds, rigidWater=True, verbose=True)
    is_drude = any(type(f) == mm.DrudeForce for f in system.getForces())

    ### apply TT damping for CLPol force field
    donors = [atom.idx for atom in psf.atom_list if atom.attype == 'HO']
    if is_drude and len(donors) > 0:
        print('Add TT damping between HO and Drude dipoles')
        ttforce = oh.CLPolCoulTT(system, donors)
        print(ttforce.getEnergyFunction())

    print('Initializing simulation...')
    if tcoupl == 'langevin':
        if is_drude:
            print('Drude Langevin thermostat: 5.0 /ps, 20 /ps')
            integrator = mm.DrudeLangevinIntegrator(T * kelvin, 5.0 / ps, 1 * kelvin, 20 / ps,
                                                    dt * ps)
            integrator.setMaxDrudeDistance(0.02 * nm)
        else:
            print('Langevin thermostat: 1.0 /ps')
            integrator = mm.LangevinIntegrator(T * kelvin, 1.0 / ps, dt * ps)
    elif tcoupl == 'nose-hoover':
        if is_drude:
            print('Drude temperature-grouped Nose-Hoover thermostat: 10 /ps, 40 /ps')
        else:
            print('Nose-Hoover thermostat: 10 /ps')
        from velocityverletplugin import VVIntegrator
        integrator = VVIntegrator(T * kelvin, 10 / ps, 1 * kelvin, 40 / ps, dt * ps)
        integrator.setUseMiddleScheme(True)
        integrator.setMaxDrudeDistance(0.02 * nm)
    else:
        raise Exception('Available thermostat: langevin, nose-hoover')

    if pcoupl != 'no':
        oh.apply_mc_barostat(system, pcoupl, P, T)

    if cos != 0:
        try:
            integrator.setCosAcceleration(cos)
        except:
            raise Exception('Cosine acceleration not compatible with this integrator')

    _platform = mm.Platform.getPlatformByName('CUDA')
    _properties = {'CudaPrecision': 'mixed'}
    sim = app.Simulation(psf.topology, system, integrator, _platform, _properties)
    if restart:
        sim.loadCheckpoint(restart)
        sim.currentStep = round(sim.context.getState().getTime().value_in_unit(ps) / dt / 10) * 10
        sim.context.setTime(sim.currentStep * dt)
        append = True
    else:
        sim.context.setPositions(gro.positions)
        sim.context.setVelocitiesToTemperature(T * kelvin)
        append = False

    sim.reporters.append(app.DCDReporter('dump.dcd', 10000, enforcePeriodicBox=False,
                                         append=append))
    sim.reporters.append(oh.CheckpointReporter('cpt.cpt', 10000))
    sim.reporters.append(oh.GroReporter('dump.gro', 1000, logarithm=True, append=append))
    sim.reporters.append(oh.StateDataReporter(sys.stdout, 1000, box=False, volume=True,
                                              append=append))
    if is_drude:
        sim.reporters.append(oh.DrudeTemperatureReporter('T_drude.txt', 10000, append=append))
    if cos != 0:
        sim.reporters.append(oh.ViscosityReporter('viscosity.txt', 1000, append=append))

    return sim


if __name__ == '__main__':
    oh.print_omm_info()
    sim = gen_simulation(gro_file=args.gro, psf_file=args.psf, prm_file=args.prm,
                         dt=args.dt, T=args.temp, P=args.press,
                         tcoupl=args.thermostat, pcoupl=args.barostat,
                         cos=args.cos,
                         restart=args.cpt)

    print('Running...')
    oh.energy_decomposition(sim)
    if args.min:
        oh.minimize(sim, 100, 'em.gro')
    sim.step(args.nstep)
