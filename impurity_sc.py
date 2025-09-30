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

def read_data(fname):
    """
    Program reads amulet.out file and extract data about effective
    quantum impurity. It may also be used to vizualize, plot and analyze.
    """
    with open(fname) as f:
        amulet_out = f.readlines()

#   Regular expressions to find
    re_number = r"([-+]?\d*\.?\d*[EeDd]?[-+]?\d*)"
    re_str = r"(\w*)"
    re_nimp = re.compile(r"\s*Number of different types of impurities \[n_imp_type\]\s+" + re_number)
    re_name = re.compile(r"\s*Impurity name \[name\]\s+" + re_str)
    re_u = re.compile(r"\s*Screened Coulomb interaction \[U\]\s+" + re_number)
    re_j = re.compile(r"\s*Hund exchange coupling       \[J\]\s+" + re_number)
    re_qlm_dft = re.compile(r"\s*Number of electrons per impurity\s+" + re_number)
    re_qlm = re.compile(r"\s*Number of impurity electrons\s+" + re_number)
    re_m = re.compile(r"\s*Magnetization\s*" + re_number)
    re_mz2 = re.compile(r"\s*Instant squared magnetic moment, \< m\_z\^2 \>\s+" + re_number)
    re_mu = re.compile(r"\s*Fermi level\s+" + re_number)
    re_spinpol = re.compile(r"\s*\w*-?\w* regime of calculations \[spinpol\]\s*(\w)")

    names = []
    uvalue = []
    jvalue = []
    qlm_dft = []
    qlm = []
    magmom = []
    mz2 = []
    mu = []

    for line in amulet_out:
        match = re_mu.match(line)
        if match:
            mu.append(float(match.group(1)))

        match = re_spinpol.match(line)
        if match:
            sp = match.group(1)

        match = re_nimp.match(line)
        if match:
            nimp = int(match.group(1))
            continue

        match = re_name.match(line)
        if match:
            names.append(match.group(1))
            continue

        match = re_u.match(line)
        if match:
            uvalue.append(float(match.group(1)))
            continue

        match = re_j.match(line)
        if match:
            jvalue.append(float(match.group(1)))
            continue

        match = re_qlm_dft.match(line)
        if match:
            qlm_dft.append(float(match.group(1)))
            continue

        match = re_qlm.match(line)
        if match:
            qlm.append(float(match.group(1)))
            continue

        match = re_m.match(line)
        if match:
            magmom.append(float(match.group(1)))
            continue

        match = re_mz2.match(line)
        if match:
            mz2.append(float(match.group(1)))
            continue

    nspin = 2 if sp.lower() == 't' else 1        
    qlm_dft = qlm_dft[:len(uvalue)]

    m_dft = magmom[:len(uvalue)]
    magmom = magmom[len(uvalue):]
    mm = []
    for i in range(int(len(magmom)/2/nimp)+1):
        mm = mm + magmom[(2*i+1)*nimp:(2*i+2)*nimp]

    impurity = []
    for i, v in enumerate(names):
        dd = {'name': v}
        dd['U'] = uvalue[i]
        dd['J'] = jvalue[i]
        dd['n_dft'] = qlm_dft[i]
        dd['m_dft'] = m_dft[i]
        dd['n_imp'] = qlm[i::nimp]
        dd['m_imp'] = mm[i::nimp]
        dd['mz2'] = mz2[i::nimp]
        impurity.append(dd)

    return mu, nspin, impurity

def stat4list(lidict, lastiter):
    """
    If dictionary contains a list, the mean and standard deviation are
    calculated and added to dictionary with the 'key'_stat.
    Parameters:
        lidict - list of dictionaries,
        lastiter - how much last items to use for mean and std calculation.
    Returns:
        Updated lidict.
    """
    
    for i in lidict:
        ndict = {}
        for k, v in i.items():
            if isinstance(v, list) and len(v)>0:
                _mean, _std = stat(v, lastiter)
                ndict[k+'_stat'] = [_mean[-1], _std[-1]]

        for k, v in ndict.items():
            i[k] = v

def visualize(impurity, mu, nspin=1, savefig=False):
    """
    """
    markers = ['o', 's', '^', 'v', 'p', '>', '<', '*']
    nrw_ = 2+nspin if len(impurity[0]['mz2']) > 0 else 1+nspin
    figure, axes = plt.subplots(nrows=nrw_, figsize=(10,13), sharex=True,
                                facecolor='None', edgecolor='None')
        
    axes[0].set_ylabel(r'$\mu$  (eV)', fontsize=12)
    axes[0].plot(range(1,len(mu)+1), mu, 'h--', c='black', markersize=8)
    axes[1].set_ylabel(r'$n_{imp}$', fontsize=12)

    if nspin == 2:
        axes[2].set_ylabel(r'$\langle m_z \rangle$', fontsize=12)
        axes[2].axhline(y=0, color='black', linewidth=0.5)

    if len(impurity[0]['mz2']) > 0:
        axes[-1].set_ylabel(r'$\langle m_z^2 \rangle$', fontsize=12)
        axes[-1].set_xlabel('Iterations', fontsize=12)

    for imp, mar in zip(impurity, markers):
        if len(imp['mz2']) > 0:
            n = np.min([len(mu), len(imp['mz2'])])
            axes[-1].plot(range(1,n+1), imp['mz2'], marker=mar, ls='--', markersize=8)
        else:
            n = np.min([len(mu), len(imp['n_imp'])])

        axes[1].plot(range(1,n+1), imp['n_imp'], marker=mar, ls='--', markersize=8, label=imp['name'])
        if nspin == 2:
            axes[2].plot(range(1,n+1), imp['m_imp'], marker=mar, ls='--', markersize=8)

    figure.legend(fontsize=12)
    figure.tight_layout()

    if savefig:
        plt.savefig("imp_vs_n.pdf")

    plt.show()

def main():
    """
    Program reads amulet.out file and extract data about effective
    quantum impurity. It may also be used to vizualize, plot and analyze.
    """
    clargs = handle_commandline()  

    mu, nspin, impurity = read_data(clargs.i.name)

    mu_dft = mu.pop(0)
    mu_mean, mu_std = stat(mu, clargs.lastiter)
    stat4list(impurity, clargs.lastiter)

#   Visualization
    if clargs.visualize or clargs.savefig:
        visualize(impurity, mu, nspin=nspin, savefig=clargs.savefig)

    for v in impurity:
        if len(v['n_imp']) != len(mu): v['n_imp'].append('')
        if len(v['m_imp']) != len(mu): v['m_imp'].append('')
        if len(v['mz2']) != len(mu): v['mz2'].append('')

#   Print out
    print("#" + 71*"-")
    print("#")
    print("#" + 30*" " + "DFT data")
    print(f"#  Fermi level, mu = {mu_dft}")
    print(f"#  There are {len(impurity)} type(s) of impurities")
    print("#")
    for v in impurity:
        print(f"#  Impurity name:               {v['name']}")
        print(f"#    Screened Coulomb interaction, U = {v['U']}")
        print(f"#    Hund's exchange coupling,     J = {v['J']}")
        print(f"#    Occupation,                 qlm = {v['n_dft']}")
        print(f"#    Magnetization,                m = {v['m_dft']}")
    print("#")
    print("#" + 71*"-")
    print("#\n#" + 25*" " + "DFT+DMFT data\n#")
    print("#              ", end="")
    for imp in impurity:
        print(f"|            {imp['name']}             ", end="")
    print("\n#        \u03BC     |  n_imp    m_imp   <m_z^2>  | \n#")

    for i, v in enumerate(mu):
        print(f" {i+1: >3}  {v: <7}", end="")
        for imp in impurity:
#            print(f"  |  {imp['n_imp'][i]: <7}" + 
#                  f" {imp['m_imp'][i]: <7} {imp['mz2'][i]: <7}", end='')
            print(f"  |  {imp['n_imp'][i]: <7} {imp['m_imp'][i]: <7}", end='')
            try:
                print(f" {imp['mz2'][i]: <7}", end='')
            except:
                pass
        print()

    print("\n\n#" + 71*"-")
    print(f"#  Mean (m) and standard deviation (s) over last"
          f"{np.min([clargs.lastiter,np.size(mu)]): } iterations")
    print(f"#  m  {mu_mean[-1]: >.5f}", end="")
    for imp in impurity:
        print(f"  |  {imp['n_imp_stat'][0]: <7.5f} {imp['m_imp_stat'][0]: <7.5f}", end='')
        try:
            print(f" {imp['mz2_stat'][0]: <7.5f}", end='')
        except:
            pass
    print(f"\n#  s  {mu_std[-1]: >.5f}", end="")
    for imp in impurity:
        print(f"  |  {imp['n_imp_stat'][1]: <7.5f} {imp['m_imp_stat'][1]: <7.5f}", end='')
        try:
            print(f" {imp['mz2_stat'][1]: <7.5f}", end='')
        except:
            pass
    print()

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