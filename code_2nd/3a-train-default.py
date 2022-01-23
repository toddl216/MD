import argparse
import os
import ase.io

from amp import Amp
from amp.descriptor.gaussian import Gaussian
from amp.model.neuralnetwork import NeuralNetwork
from amp.model import LossFunction
from amp.utilities import Annealer


def run_training(trajfile, Lforce):
    '''
    default energy_rmse     0.001
            energy_coeff    1.0
            force_rmse      0.01
            force_coeff     0.04
    '''            
    images = ase.io.Trajectory(trajfile)
    calc = Amp(descriptor=Gaussian(), model=NeuralNetwork(hiddenlayers=(10, 10, 10)), cores=8)
    Annealer(calc=calc, images=images, Tmax=20, Tmin=1, steps=4000)
    convergence={}
    convergence['energy_rmse'] = 0.001       
    if Lforce:
        convergence['force_rmse'] = 0.02        # between 0.01 ~ 0.02 ~ 0.05 ~ 0.1
        calc.model.lossfunction = LossFunction(convergence=convergence, force_coefficient=0.04)
    else:
        calc.model.lossfunction = LossFunction(convergence=convergence)
    calc.train(images=images)

def main():
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument('fin', help='input trajectory file which can be read by ase')
    parser.add_argument('-f', '--force', action='store_true', help='add force training')
    args = parser.parse_args()

    run_training(args.fin, args.force)

if __name__ == '__main__':
    main()
