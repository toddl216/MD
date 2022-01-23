"""Simple test of the Amp calculator, using Gaussian descriptors and neural
network model. Randomly generates data with the EMT potential in MD
simulations."""

import os
from ase import Atoms, Atom, units
import ase.io
from ase.build import fcc110
from ase.constraints import FixAtoms

''' index for fcc110 size
    7   for 2, 2, 2 w. vac=7
    50  for 4, 5, 3 
    86  for 4, 6, 4
    340 for 8, 12, 4
'''


def build_struct():
    pivot = 50 
    atoms = fcc110('Pt', (4, 5, 3), vacuum=15.)
    atoms.extend(Atoms([Atom('Cu', atoms[pivot].position + (0., 0., 2.5)),
                        Atom('Cu', atoms[pivot].position + (0., 0., 5.))]))
    #atoms.set_constraint(FixAtoms(indices=[0, 2]))
    ase.io.write('POSCAR', atoms, format='vasp')

if __name__ == '__main__':
    build_struct()
