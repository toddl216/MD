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
import numpy as np

from amp import Amp


def ase_newversion():
    ver = ase.__version__
    ref = '3.21.0'
    if ver >= ref:
        return 1
    else:
        return 0

def generate_data(fin, amp_pot, nstep, dump_freq, trajfile='test.traj'):
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

    #images = []
    for step in range(ncount):
        dyn.run(dump_freq)
        traj.write(atoms)
        #images.append(atoms)
        #print(f"{atoms.get_potential_energy():10.3f}")
        #print(images[-1].get_positions()[61])
        print("step: ", dump_freq * (step+1), "/", nstep)

    y    = []
    ybar = []
    f    = []
    fbar = []
    calc = Amp.load(amp_pot)
    ### images cannot be saved in the same script: it should be read from outside file
    images = ase.io.Trajectory(trajfile)
    for image in images:
        peemt = image.get_potential_energy()
        femt  = image.get_forces()
        y.append(peemt)
        f.extend(femt)          # as 1D
        image.set_calculator(calc)
        peamp = image.get_potential_energy()
        famp  = image.get_forces()
        ybar.append(peamp)
        fbar.extend(famp)
        #print(image.get_positions()[61])
    
    y     = np.array(y)
    ybar  = np.array(ybar)
    diff  = ybar - y
    Ermse = np.sqrt(np.sum(np.square(diff)))/float(len(y))
    Emax  = np.max(np.absolute(diff))
    res = '\n'.join(map(str, ['{:.3f}'.format(x) for x in diff]))
    print(f"Energy diff :: \n{res}")
    print("Force test")
    f     = np.array(f)
    fbar  = np.array(fbar)
    diff  = fbar - f
    Frmse = np.sqrt(np.sum(np.square(diff)))/float(len(f))
    Fmax  = np.max(np.absolute(diff))
    with open("stat.dat", 'w') as f:
        s1 = f"Ene RMSE: {Ermse:10.3f}\tEne Maxres: {Emax:10.3f}\n"
        s2 = f"For RMSE: {Frmse:10.3f}\tFor Maxres: {Fmax:10.3f}"
        print(s1, s2, sep='')
        f.write(s1)
        f.write(s2)
        #f.write(diff)

    return 0

def main():
    parser = argparse.ArgumentParser(description='Run EMT')
    parser.add_argument('fin', help='input geometry read by ASE')
    parser.add_argument('amppot', default='amp.amp', help="input amp potential")
    parser.add_argument('-ns', '--nstep', default=1000, type=int, help='total number of time steps')
    parser.add_argument('-df', '--dfreq', default=10, type=int, help='write every dnstep')

    args = parser.parse_args()

    generate_data(args.fin, args.amppot, args.nstep, args.dfreq)

if __name__ == '__main__':
    main()
