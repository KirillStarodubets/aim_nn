#!/usr/bin/env python3
"""
  Copyright (c) 2015-2025, AMULET Development Team.
  Distributed under the terms of the FreeBSD License.

  This file contains different utilities for AMULET code.
"""
import argparse
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt

from amulet_tools import *

__copyright__ = "Copyright 2024-2025, AMULET Developers Team"
__maintainer__ = "Alexander Poteryaev"
__email__ = "poteryaev.alexander@gmail.com"

#pd.set_option('display.max_columns', 16)
pd.set_option('display.max_rows', None)
pd.set_option("display.precision", 4)

def main():
    """
    Fit low-energy imaginary part of function with power law and
    parabolic functions.
    """
    clargs = handle_commandline()

#   Set boundary conditions for parameters of fit
    if clargs.no_bounds:
        bounds_p = ([-np.inf, -np.inf, 0], [0, 0, np.inf])
        bounds_c = ([-np.inf, -np.inf, -np.inf], [0, 0, np.inf])
    else:
        bounds_p = (-np.inf, np.inf)
        bounds_c = (-np.inf, np.inf)         

#   Read complex (self-energy) function to fit imaginary part at low-energy
    iwn, sigma = read_amulet_dat_file(clargs.infile.name, dtype='complex')
    sigma = np.transpose(sigma, (2, 0, 1))
    x = np.linspace(start=0, stop=2*iwn[clargs.npoints+1], num=100)
    ymin = np.min(np.imag(sigma)) - 0.02

#   Loop over spins
    for sp, sig in enumerate(sigma):
        fig, ax = plt.subplots()
        coef_ = []
        deriv_ = []
#   Loop over number of Matsubara's points to fit and over interations
        for np_ in range(4, clargs.npoints+1):
            for i, s in enumerate(sig[-clargs.last:,:]):
#   Power law fit: alpha + beta*x^k   
                try:
                    power_par, power_std = curve_fit(
                        power_n, iwn[:np_], np.imag(s[:np_]),
#                        p0=[0, -0.1, 0.5], bounds=bounds_p
                        p0=[np.imag(s[0]), -0.1, 0.7], bounds=bounds_p
                        )
                    power_par = power_par.tolist()
                    power_std = np.sqrt(np.diag(power_std)).tolist()
                except RuntimeError as err:
                    print(err)
                    power_par = power_std = [np.nan]*3

                try:
#   Cubic fit: a + b*x + c*x^2 + d*x^3
#                    cubic_par, cubic_std = curve_fit(
#                        cubic, iwn[:np_], np.imag(s[:np_]), 
#                        p0=[-0.1, -0.1, 1.0, 1.0], bounds=bounds_c
#                        )
#                    cubic_par = cubic_par.tolist()
#                    cubic_std = np.sqrt(np.diag(cubic_std)).tolist()

#   Parabolic fit: a + b*x + c*x^2
                    cubic_par, cubic_std = curve_fit(
                        para, iwn[:np_], np.imag(s[:np_]), 
                        p0=[-0.1, -0.1, 1.0], bounds=bounds_c)
                    cubic_par = cubic_par.tolist()
                    cubic_std = np.sqrt(np.diag(cubic_std)).tolist()
                except RuntimeError as err:
                    print(err)
                    cubic_par = cubic_std = [np.nan]*4

                coef_.append([np_, i, *power_par, *power_std,
                              *cubic_par, *cubic_std])
                
                numdiff = np.imag(s[0]) / iwn[0]
                c_d = cubic_par[1] + 2*cubic_par[2]*iwn[0]/2
                p_d = power_par[1]*power_par[2]*np.power(iwn[0]/2, power_par[2]-1)

                deriv_.append([np_, i, cubic_par[1], c_d, p_d, numdiff])

                if clargs.visualize:
                    ax.plot(iwn, np.imag(s), 'o')
                    lastcolor = plt.gca().lines[-1].get_color()
                    ax.plot(x, [para(x_, *cubic_par) for x_ in x],
                             c=lastcolor)
                    ax.plot(x, [power_n(x_, *power_par) for x_ in x],
                             c=lastcolor, ls='dashed')

        if clargs.visualize:
            plt.xlim(0, 5)
            plt.ylim(bottom=ymin, top=0)

            ax.plot([0.66, 0.72], [0.95, 0.95], 'o', c='black',
                    transform=ax.transAxes)
            ax.plot([0.65, 0.73], [0.9, 0.9], c='blue',
                    transform=ax.transAxes)
            ax.plot([0.65, 0.73], [0.85, 0.85], ls='dashed', c='green',
                    transform=ax.transAxes)
            ax.text(0.75, 0.95, r'$data \;\; points$', c='black',
                    transform=ax.transAxes, verticalalignment='center')
            ax.text(0.75, 0.9, r'$a + bx + cx^2$', c='blue',
                    transform=ax.transAxes, verticalalignment='center')
            ax.text(0.75, 0.85, r'$\alpha + \beta x^k$', c='green',
                    transform=ax.transAxes, verticalalignment='center')

            plt.xlabel(r'$i\omega_n$,  eV')
            plt.ylabel(r'$\Im\Sigma(i\omega_n)$,  eV')
            plt.subplots_adjust(left=0.12, right=0.99, top=0.99, bottom=0.1)
            plt.show()

        df_coef = pd.DataFrame(coef_,
                               columns=['# N', 'Iter', 'alpha', 'beta',
                                        'k', 'alpha_std', 'beta_std',
                                        'k_std', 'a', 'b', 'c',
                                        'a_std', 'b_std', 'c_std'])
        
        if clargs.save:
            df_coef.to_csv('fit_coefficients'+str(sp)+'.dat', sep='\t',
                           float_format='%.4f', index=False)
            
        if clargs.info == 'f':
            print(80*'-')
            print(' Coefficients of fit for all requested iterations and points')
            print(df_coef)
            print()
        
#   Quasiparticle residue: Z = 1 / ( 1 - dSigma/dwn )
        df_deriv = pd.DataFrame(deriv_,
                                columns=['# N', 'Iter', 'b', 'cubic',
                                         'power_n', 'numdiff'])
        df_deriv['Z_b'] = 1 / ( 1 - df_deriv['b'] )
        df_deriv['Z_cubic'] = 1 / ( 1 - df_deriv['cubic'] )
        df_deriv['Z_power_n'] = 1 / ( 1 - df_deriv['power_n'] )
        df_deriv['Z_numdiff'] = 1 / ( 1 - df_deriv['numdiff'] )

        if clargs.save:
            df_deriv.to_csv('z'+str(sp)+'.dat', sep='\t',
                            float_format='%.4f', index=False)

        if clargs.info in ['f', 'z']:
            print(80*'-')
            print(' Z for all requested iterations and points')
            print(df_deriv)
            print()

        if clargs.info in ['f', 'z', 'p', 'ip', 'pi']:
            print(80*'-')
            print(' Z mean grouped by points')
            print(df_deriv.drop(columns='Iter').groupby(['# N']).mean())
            print()

        if clargs.info in ['f', 'z', 'i', 'ip', 'pi']:
            print(80*'-')
            print(' Z mean grouped by iterations')
            print(df_deriv.drop(columns='# N').groupby(['Iter']).mean())
            print()

        if clargs.info in ['f', 'z', 'i', 'p', 'ip', 'pi', '0']:
            print(80*'-')
            print(' Z mean over all requested iterations and points')
            a = df_deriv.mean().drop(['# N', 'Iter', 'b', 'cubic',
                                      'power_n', 'numdiff']).to_frame().T
            print(a.to_string(index=False))
            print()
    
    exit(0)

def para(x, *a):
    """
    Calculate quadratic polynomial.
    Parameters:
        x - variable,
        *a - coefficients at corresponding powers.
    Returns:
        value of cubic polynomial at x.
    """
    f = a[0] + a[1]*x + a[2]*x*x
    return f

def cubic(x, *a):
    """
    Calculate cubic polynomial.
    Parameters:
        x - variable,
        *a - coefficients at corresponding powers.
    Returns:
        value of cubic polynomial at x.
    """
    f = a[0] + a[1]*x + a[2]*x*x + a[3]*x*x*x
    return f

def power_n(x, *a):
    """
    Calculate power function.
    Parameters:
        x - variable,
        a[0] - constant,
        a[1] - factor at power function,
        a[2] - power.
    Returns:
        value of power function.
    """
    f = a[0] + a[1]*np.power(x, a[2])
    return f

def handle_commandline():
    """
    Method defines commandline arguments
    """
    parser = argparse.ArgumentParser(
        description=r'Calculate quasiparticle residue of self-energy, $Z^{-1} = 1 - d \Im\Sigma / d i\omega_n$')
    
    parser.add_argument('infile', metavar='file', 
        type=argparse.FileType('r'), help='self-energy file')
    
    parser.add_argument('-n', '--npoints', type=int, metavar='N',
        default=5, choices=range(4,30),
        help='maximal number of Matsubara points to fit [default: %(default)s]')
    
    parser.add_argument('-l', '--last', type=int, default=1, metavar='L',
        help='number of last iterations to use [default: %(default)s]')
    
    parser.add_argument('-nb', '--no-bounds', action='store_false',
        help="don't use bounds for fitting parameters")
    
    parser.add_argument('-s', '--save', action='store_true',
        help='save output to data files')
    
    parser.add_argument('-v', '--visualize', action='store_true',
                    help='visualize input and fit data')
    
    parser.add_argument('-i', '--info', type=str, default='0',
        choices=['f', 'z', 'i', 'p', 'ip', 'pi', '0', 's'],
        help='amount of information to print [default: %(default)s]')

    return parser.parse_args()

if __name__ == "__main__":
    main()
