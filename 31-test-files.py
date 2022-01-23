import argparse
import os
import ase
from ase import Atoms, Atom, units
import ase.io
from ase.calculators.emt import EMT
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

def get_ene_stat(y, ybar):
    y     = np.array(y)
    ybar  = np.array(ybar)
    diff  = ybar - y
    Ermse = np.sqrt(np.sum(np.square(diff))/float(len(y)))
    Emax  = np.max(np.absolute(diff))
    res = '\n'.join(map(str, ['{:.3f}'.format(x) for x in diff]))
    print(f"Energy diff :: \n{res}")
    return Ermse, Emax

def get_force_stat(f, fbar):
    f     = np.array(f).flatten()
    fbar  = np.array(fbar).flatten()
    diff  = fbar - f
    print(f"shape of diff: {diff.shape}, {f.shape}")
    Frmse = np.sqrt(np.sum(np.square(diff))/float(len(f)))
    Fmax  = np.max(np.absolute(diff))
    return Frmse, Fmax

def test_traj(amp_pot, files):

    total_images=[]
    for f in files:
        images = ase.io.Trajectory(f)
        total_images.extend(images)
    print(f"total number of images to be tested: {len(total_images)}")
    
    y    = []
    ybar = []
    f    = []
    fbar = []
    calc = Amp.load(amp_pot)
    ### images cannot be saved in the same script: it should be read from outside file
    for image in total_images:
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

    Ermse, Emax = get_ene_stat(y, ybar)

    print("Force test")

    Frmse, Fmax = get_force_stat(f, fbar)
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
    parser.add_argument('amppot', default='amp.amp', help="input amp potential")
    parser.add_argument('-inf', '--infile', nargs='*', help='read several trajectories for test')

    args = parser.parse_args()

    test_traj(args.amppot, args.infile)

if __name__ == '__main__':
    main()
