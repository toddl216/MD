import argparse
from amp import Amp
import ase
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import VelocityVerlet

def ase_newversion():
    ver = ase.__version__
    ref = '3.21.0'
    if ver >= ref:
        return 1
    else:
        return 0

def amp_md(amp_pot, atoms, nstep, dump_freq, dt):
    """ Running AIMD using amp.amp """
    traj = ase.io.Trajectory("nn.traj", 'w')
    calc = Amp.load(amp_pot)

    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    if ase_newversion() :
        MaxwellBoltzmannDistribution(atoms, temperature_K = 300.)
    else:
        MaxwellBoltzmannDistribution(atoms, 300.*units.kB)
    traj.write(atoms)
    dyn = VelocityVerlet(atoms, timestep= dt * units.fs)
    f = open("nn-pot.ene", "w")
    f.write(f"{'time':^5s}{'Etot':^15s}{'Epot':^15s}{'Ekin':^10s}\n")
    print(f"\n Dump at every {dump_freq} timestep")
    print(f"   {'time':^5s}{'Etot':^15s}{'Epot':^15s}{'Ekin':^10s}")
    ncount = int(nstep/dump_freq)
    for step in range(ncount):
        pot = atoms.get_potential_energy()  # 
        kin = atoms.get_kinetic_energy()
        tot = pot + kin
        f.write(f"{step:5d}{tot:15.4f}{pot:15.4f}{kin:10.4f}\n")
        print(f"{step:5d}{tot:15.4f}{pot:15.4f}{kin:10.4f}")
        dyn.run(dump_freq)
        traj.write(atoms)                   # write kinetic energy, but pot is not seen in ase
    f.close()        

def main():
    parser = argparse.ArgumentParser(description='run amp with extxyz, OUTCAR: validation is removed ', prefix_chars='-+/')
    parser.add_argument('pot', default='amp.amp', help="input amp potential")
    parser.add_argument('fin', help='input trajectory file which can be read by ase')
    parser.add_argument('-i','--index', default=0, type=int, help='select start configuration from input file')
    parser.add_argument('-ns','--nstep', default=1000, type=int, help='number of step with dt')
    parser.add_argument('-df', '--dfreq', default=20, type=int, help='write every dnstep')
    parser.add_argument('-dt','--dt', default=1.0, type=float, help='time interval in fs')

    args = parser.parse_args()

    atoms = ase.io.read(args.fin, index=args.index)      # index = string
    amp_md(args.pot, atoms, args.nstep, args.dfreq, args.dt)

if __name__ == '__main__':
    main()

