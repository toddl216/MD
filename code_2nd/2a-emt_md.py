import argparse
import os
import ase
from ase import Atoms, Atom, units
import ase.io
from ase.calculators.emt import EMT
from ase.build import fcc110
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet
from ase.constraints import FixAtoms

def ase_newversion():
    ver = ase.__version__
    ref = '3.21.0'
    if ver >= ref:
        return 1
    else:
        return 0

def generate_data(fin, nstep, dump_freq, trajfile='emtmd.traj'):
    """Generates test or training data with a EMT MD simulation."""
    traj = ase.io.Trajectory(trajfile, 'w')
    atoms = ase.io.read(fin)
    atoms.set_calculator(EMT())
    atoms.get_potential_energy()
    traj.write(atoms)
    
    if ase_newversion() :
        MaxwellBoltzmannDistribution(atoms, temperature_K = 300.)
    else:
        MaxwellBoltzmannDistribution(atoms, 350.*units.kB)

    dyn = VelocityVerlet(atoms, timestep=1.*units.fs )

    ncount = int(nstep/dump_freq)
    print("The number of images for database is ", ncount)
    for step in range(ncount):
        dyn.run(dump_freq)
        traj.write(atoms)
        print("step: ", dump_freq * (step+1), "/", nstep)

def main():
    parser = argparse.ArgumentParser(description='Run EMT')
    parser.add_argument('fin', help='input geometry build by ASE')
    parser.add_argument('-ns', '--nstep', default=1000, type=int, help='total number of time steps')
    parser.add_argument('-df', '--dfreq', default=50, type=int, help='write every dnstep')
    args = parser.parse_args()

    generate_data(args.fin, args.nstep, args.dfreq)

if __name__ == '__main__':
    main()
