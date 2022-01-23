### train descriptor
### Gaussian Descriptor

import numpy as np
import sys
### in case changing amp package name
#import ampm
#sys.modules['amp'] = ampm
from amp.descriptor.gaussian import make_symmetry_functions
Lprint = 1

class GS_param:
    '''
    func_param: 'log10', powNN=N^(m/N)
    Now eta & Rsm is calculated without Rc
    '''
    name='gs'
    def __init__(self, func_param='log10', pmin=0.05, pmax=5, pnum=4, pmod='orig', cutoff = 6.5):
        self.func_param = func_param
        self.pmin   = pmin
        self.pmax   = pmax
        self.nparam = pnum
        self.pmod   = pmod
        self.cutoff = cutoff

    def make_Gs(self, image):
        ### obtain atom symbols
        elements = set([atom.symbol for atom in image])
        elements = sorted(elements)     # because set is not subscriptable
        G = {}
        Np = self.nparam
        Rc = self.cutoff
        #etas = np.logspace(np.log(0.5), np.log(5), num=6)
        if self.func_param == 'log10':
            print(f"parameter function {self.func_param} was chosen in {__name__}")
            etas_orig = np.logspace(np.log10(self.pmin), np.log10(self.pmax), num=Np)
            if Lprint: print("etas_orig = ", " ".join("%.2f" % x for x in etas_orig))
            if self.pmod == 'del':
                etas = np.delete(etas_orig, [1,2])        # [9], [1,2], [1,2,9]
            elif self.pmod == 'couple':
                etas_orig2 = np.logspace(np.log10(self.pmin), np.log10(self.pmax/2.), num=Np)
                etas1 = np.take(etas_orig, [0,4,6,8])
                etas  = np.sort(np.append(etas1, np.take(etas_orig2, [3,5,7,9])))
            elif self.pmod == 'orig':
                etas = etas_orig
            else:
                print("error in input param control")
                sys.exit(0)
            if Lprint: print("   etas   = ", " ".join("%.2f" % x for x in etas))
            Rs = list(np.zeros(len(etas)))
        elif self.func_param == "powNN":
            ### G2a: for N, num(m) = N+1, m={0,1,...,N), Rs = 0
            etas_orig = [ Np**(2*m/Np) for m in range(Np+1)]                  # without Rc
            if self.pmod == 'mod':
                etas = [ Np**(2*m/Np) for m in range(Np+1) if m%2 == 0]           # without Rc
            elif self.pmod == 'orig':
                etas = etas_orig
            else:
                print("error in input param control")
                sys.exit(0)
            Rs = list(np.zeros(len(etas)))
            ### G2b: Rs != 0; Rs_ms keep m=0 for eta, but it will not be used for G2
            Rsms_noRc = [ Np**(-m/Np) for m in range(Np+1)]               # without Rc
            Rsms = [ Rc*Np**(-m/Np) for m in range(Np+1)]               # without Rc
            Rsms_1 = Rsms[1:]
            etasms=[ (Rsms_noRc[m+1]-Rsms_noRc[m])**(-2) for m in range(Np)]
            Rs.extend(Rsms_1)
            etas.extend(etasms)
        else:
            print(f"{self.func_param} is not available")
            sys.exit(100)
        with open("etas.dat",'w') as f:
            f.write("etas_orig: ")
            f.write(" ".join("%.1f" % x for x in etas_orig))
            f.write("\netas     : " )
            f.write(" ".join("%.1f" % x for x in etas))
        for element in elements:
            _G = make_symmetry_functions(type='G2', etas=etas, offsets=Rs, elements=elements)       # Rc is out of eta
            #_G += make_symmetry_functions(type='G4', etas=[0.005], zetas=[1., 2., 4., 8.], gammas=[+1., -1.], elements=elements)
            _G += make_symmetry_functions(type='G4', etas=[0.005], zetas=[1., 4.], gammas=[+1., -1.], elements=elements)
            G[element] = _G
        #nflow.ncount += 1
        return G

class ZN_param:
    name='zn'

class BS_param:
    name='bs'


