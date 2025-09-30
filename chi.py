#!/usr/bin/env python3
"""
  Copyright (c) 2015-2025, AMULET Development Team.
  Distributed under the terms of the FreeBSD License.

  This file contains different utilities for AMULET code.
"""
import argparse
import numpy as np
from amulet_tools import *

__copyright__ = "Copyright 2024-2025, AMULET Developers Team"
__maintainer__ = "Alexander Poteryaev"
__email__ = "poteryaev.alexander@gmail.com"

def main():
    """
    Calculates spin or charge susceptibility out of <n_i(t)n_j(t')> 
    correlators. If some of correlators are missed it can be restored
    from symmetry consideration, <n_i(t)n_j(t')> = <n_j(t')n_i(t)>.
    """
    args = handle_commandline()

#   Read ninj*.dat files
    mesh, ninj = init_func([i.name for i in args.filename], dtype_='real')
    ni, nj = np.shape(ninj)

#   Initialize to zero array of susceptibility with proper dimensions
    for i in range(ni):
        for j in range(nj):
            if isinstance(ninj[i,j], np.ndarray):
                chi = np.zeros(np.shape(ninj[i,j])[:2])
                break
        else:
            continue
        break
            
#   Restore off-diagonal elements of <nn> correlators, n1n2(t)=n2n1(-t)
    if args.restore:
        for i in range(ni):
            for j in range(nj):
                if isinstance(ninj[i,j], int) and   \
                   isinstance(ninj[j,i], np.ndarray):
                    ninj[i,j] = ninj[j,i][:,::-1,:]

#   Charge susceptibility, <N(t)N(0)>
    if args.charge:
        for i in range(ni):
            for j in range(nj):
                if isinstance(ninj[i,j], np.ndarray):
                    chi += np.sum(ninj[i,j][:,:,:], axis=2)
#   Spin susceptibility, <Sz(t)Sz(0)>
    else:
        for i in range(ni):
            for j in range(nj):
                if isinstance(ninj[i,j], np.ndarray):
                    chi += ( ninj[i,j][:,:,0] - ninj[i,j][:,:,1] - 
                             ninj[i,j][:,:,2] + ninj[i,j][:,:,3] ) / 4

    with open('chi.dat', 'w') as fo:
        for k in chi:
            for e, v in zip(mesh, k):
                print(e, v, file=fo)
            print('\n', file=fo)

    exit(0)

def handle_commandline():
    """
    Defines and returns commandline arguments
    """
    parser = argparse.ArgumentParser(
        description='Calculates impurity spin (or charge) '
        'susceptibility out of <ninj> correlators.')

    parser.add_argument('filename', nargs='+',
        type=argparse.FileType('r'), help='input files (Required)')

    parser.add_argument('-c', '--charge', action='store_true',
        help='calculate charge susceptibility instead of spin')

    parser.add_argument('-r', '--restore', action='store_true',
        help='restore data for off-diagonal elements if missed')

    return parser.parse_args()
    
if __name__ == "__main__":
    main()