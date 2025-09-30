#!/usr/bin/env python3
"""
  Copyright (c) 2015-2025, AMULET Development Team.
  Distributed under the terms of the FreeBSD License.

  This file contains different utilities for AMULET code.
"""
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

__copyright__ = "Copyright 2020-2025, AMULET Developers Team"
__maintainer__ = "Alexander Poteryaev"
__email__ = "poteryaev.alexander@gmail.com"

def main():
    """
    Program reads amulet.out file and extract energy related data.
    It may also vizualize, plot and analyze them.
    """

    clargs = handle_commandline()  

    with open(clargs.i.name) as f:
        amulet_out = f.readlines()

#   Regular expressions to find
    re_number = r"([-+]?\d*\.?\d*[EeDd]?[-+]?\d*)"
    re_mu = re.compile(r"\s*Fermi level\s+" + re_number)
    re_edmft = re.compile(r"\s*DMFT correction to total energy and error\s+" + re_number)
    re_kin = re.compile(r"  DM?FT kinetic energy  \s+" + re_number)
    re_ntot = re.compile(r"\s*Total number of particles\s+" + re_number)
    re_beta = re.compile(r"\s*Inverse temperature \[beta\]\s+" + re_number)
    re_tinev = re.compile(r"\s*Temperature \[TineV\]\s+" + re_number)
    re_tink = re.compile(r"\s*Temperature \[TinK\]\s+" + re_number)
    re_etol = re.compile(r"\s*Tolerance for DFT\+DMFT self\-consistent "
                         r"calculations \[etol\]\s+" + re_number)

    mu = []
    e_dmft = []
    e_kin = []
    ntot = []
    etol = 0.001

    for line in amulet_out:
        match = re_beta.match(line)
        if match:
            beta = float(match.group(1))

        match = re_etol.match(line)
        if match:
            etol = float(match.group(1))

        match = re_tinev.match(line)
        if match:
            tinev = float(match.group(1))

        match = re_tink.match(line)
        if match:
            tink = float(match.group(1))

        match = re_mu.match(line)
        if match:
            mu.append(float(match.group(1)))

        match = re_edmft.match(line)
        if match:
            e_dmft.append(float(match.group(1)))

        match = re_kin.match(line)
        if match:
            e_kin.append(float(match.group(1)))

        match = re_ntot.match(line)
        if match:
            ntot.append(float(match.group(1)))

    mu_dft = mu.pop(0)
    ntot_dft = ntot.pop(0)

    e_kin_dft = e_kin.pop(0)
    e_kin = e_kin[::2]
    e_dmft = np.array(e_dmft)

#   Calculation of mean and standard deviation
    mu_mean, mu_std = stat(mu, clargs.lastiter)
    ntot_mean, ntot_std = stat(ntot, clargs.lastiter)
    e_kin_mean, e_kin_std = stat(e_kin, clargs.lastiter)
    e_dmft_mean, e_dmft_std = stat(e_dmft, clargs.lastiter)

#   Print out
    print("#" + 60*"-")
    print("#\n#" + 20*" " + "DFT data\n#")
    print(f"#  Fermi level,            mu =  {mu_dft: >12.5f}")
    print(f"#  Inverse temperature,  beta =  {beta: >12.5f}")
    print(f"#  Temperature (in eV), k_B T =  {tinev: >12.5f}")
    print(f"#  Temperature (in K),      T =  {tink: >12.5f}")
    print(f"#  Kinetic energy,          K =  {e_kin_dft: >12.5f}")
    print("#\n#" + 60*"-")
    print("#\n#" + 17*" " + "DFT+DMFT data\n#")
    print("#        \u03BC        N_tot     E_kin     E_DMFT\n#")


    for i in range(len(e_dmft)):
        print(f" {i+1: >3d}  {mu[i]: >.5f}  {ntot[i]: >.5f}  "
              f"{e_kin[i]: >.5f}  {e_dmft[i]: >.5f}")
#        print(i+1, mu[i], ntot[i], e_kin[i], e_dmft[i])

    print("\n\n#" + 60*"-")
    print(f"#  Mean (m) and standard deviation (s) over last"
          f"{np.min([clargs.lastiter,np.size(e_dmft)]): } iterations")
    print(f"#  m  {mu_mean[-1]: >.5f}  {ntot_mean[-1]: >.5f}  "
          f"{e_kin_mean[-1]: >.5f}  {e_dmft_mean[-1]: >.5f}")
    print(f"#  s   {mu_std[-1]: >.5f}   {ntot_std[-1]: >.5f}    "
          f"{e_kin_std[-1]: >.5f}   {e_dmft_std[-1]: >.5f}")

#   Visualization
    if clargs.visualize or clargs.savefig:
        figure, axes = plt.subplots(figsize=(10,7), facecolor='None', 
                                    edgecolor='None')
        axes.set_xlabel('Iterations', fontsize=12)
        a_ = round(np.mean(e_dmft), 3)
        s_ = ' + ' + str(-a_) if a_ < 0 else ' - ' + str(a_)
        axes.set_ylabel('E$_{DMFT}$'+s_+',  eV', fontsize=12)
        axes.axhline(y=0, color='k', linewidth=0.5)
        axes.axhline(y=etol, color='brown')
        axes.axhline(y=-etol, color='brown')
        axes.text(0, 1.05*etol, r'$\varepsilon_{tol}$', color='brown')
        axes.text(-0.3, -1.23*etol, r'$-\varepsilon_{tol}$', color='brown')

        axes.plot(range(1,len(e_dmft)+1), e_dmft-a_, 'o-', markersize=8, 
                  label=r'$E_{DMFT}$')
        axes.plot(range(1,len(e_dmft)+1), e_dmft_mean-a_, 's-', markersize=7, 
                  color='red', 
                  label=r'$\langle E_{{DMFT}} \rangle$'.format(clargs.lastiter))
        axes.plot(range(1,len(e_dmft)+1), e_dmft_std, 's-.', color='green',
                  label=r'$\sigma$')
        axes.plot(range(2,len(e_dmft)+1), e_dmft[1:]-e_dmft[:-1], 'o--', 
                  label=r'$\Delta E_{DMFT}$')

        axes.legend(loc=(0.03, 0.75), fontsize=12)

        if clargs.savefig:
            plt.savefig("e_vs_n.pdf")

        plt.show()

def stat(data, n):
    """
    Moving average.
    Returns arrays of mean and standard deviaton from last n iterations.
    Args:
        data - data array,
        n - number of last iterations to evaluate mean and std.
    Returns:
        mean - array of means,
        std - array of std.
    """

    dim = len(data)
    mean = np.zeros(dim)
    std = np.zeros(dim)
    mean[0] = data[0]

    for i in range(1,dim):
        n_ = n-1 if i >= n else i
        mean[i] = np.mean(data[i-n_:i+1])
        std[i] = np.std(data[i-n_:i+1])

    return mean, std

def handle_commandline():
    """
    Method defines commandline arguments
    """
    parser = argparse.ArgumentParser(description=
            "Program reads amulet.out file and extracts energy related data.")

    parser.add_argument('-i', '-infile', metavar='file', 
            type=argparse.FileType('r'), default='amulet.out',
            help='name (and path) of the AMULET output file '
            '[default: %(default)s]')

    parser.add_argument('-l', '--lastiter', metavar='N', type=int,
            default=10, help='number of last AMULET interations to average '
            '[default: %(default)s]')

    parser.add_argument('-v', '--visualize', action='store_true',
            help='visualize data')

    parser.add_argument('-s', '--savefig', action='store_true',
            help='visualize and save figure to file "e_vs_n.pdf"')

    return parser.parse_args()

if __name__ == "__main__":
    main()

