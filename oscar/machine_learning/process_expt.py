# file to read and process experimentally collected density matrices
import numpy as np
from scipy.optimize import minimize, approx_fprime
from os.path import join, dirname, abspath
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from uncertainties import ufloat
from uncertainties import unumpy as unp

from sample_rho import *
from rho_methods import *
from jones import *

# set path
current_path = dirname(abspath(__file__))
DATA_PATH = join(current_path, '../../framework/decomp_test/')

def get_rho_from_file_depricated(filename, rho_actual):
    '''Function to read in experimental density matrix from file. Depricated since newer experiments will save the target density matrix in the file; for trials <= 14'''
    # read in data
    try:
        rho, unc, Su = np.load(join(DATA_PATH,filename), allow_pickle=True)
    except:
        rho, unc = np.load(join(DATA_PATH,filename), allow_pickle=True)

    # print results
    print('measublue rho\n---')
    print(rho)
    print('uncertainty \n---')
    print(unc)
    print('actual rho\n ---')
    print(rho_actual)
    print('fidelity', get_fidelity(rho, rho_actual))
    print('purity', get_purity(rho))

    print('trace of measublue rho', np.trace(rho))
    print('eigenvalues of measublue rho', np.linalg.eigvals(rho))

def get_rho_from_file(filename, verbose=True, angles=None):
    '''Function to read in experimental density matrix from file. For trials > 14. N.b. up to trial 23, angles were not saved (but recorded in lab_notebook markdown file). Also note that in trials 20 (E0 eta = 45), 21 (blueo of E0 (eta = 45, chi = 0, 18)), 22 (E0 eta = 60), and 23 (E0 eta = 60, chi = -90), there was a sign error in the phi phase in the Jones matrices, so will recalculate the correct density matrix; ** the one saved in the file as the theoretical density matrix is incorrect **
    --
    Parameters
        filename : str, Name of file to read in
        verbose : bool, Whether to print out results
        angles: list, List of angles used in the experiment. If not None, will assume angles provided in the data file.
    '''

    def split_filename():
            ''' Splits up the file name and identifies the trial number, eta, and chi values'''

            # split filename
            split_filename = filename.split('_')
            # get trial number
            trial = int(split_filename[-1].split('.')[0])
            # get eta
            eta = float(split_filename[1].split(',')[1].split('(')[1])
            chi = float(split_filename[1].split(',')[2].split(')')[0].split(' ')[1])

            return trial, eta, chi

    # read in data
    try:

        # rho, unc, Su, rho_actual, angles, fidelity, purity = np.load(join(DATA_PATH,filename), allow_pickle=True)
        rho, unc, Su, un_proj, un_proj_unc, rho_actual, angles, fidelity, purity = np.load(join(DATA_PATH,filename), allow_pickle=True)
    

        ## update df with info about this trial ##
        if "E0" in filename: # if E0, split up into eta and chi
            trial, eta, chi = split_filename()

        # print results
        if verbose:
            print('angles\n---')
            print(angles)
            print('measured rho\n---')
            print(rho)
            print('uncertainty \n---')
            print(unc)
            print('actual rho\n ---')
            print(rho_actual)
            print('fidelity', fidelity)
            print('purity', purity)

            print('trace of measublue rho', np.trace(rho))
            print('eigenvalues of measublue rho', np.linalg.eigvals(rho))

        return trial, rho, unc, Su, rho_actual, fidelity, purity, eta, chi, angles, un_proj, un_proj_unc
    except:
        rho, unc, Su, rho_actual, _, purity = np.load(join(DATA_PATH,filename), allow_pickle=True)
        # print(np.load(join(DATA_PATH,filename), allow_pickle=True))
        
        ## since angles were not saved, this means we also have the phi sign error as described in the comment to the function, so will need to recalculate the target. ##

        def split_filename():
            ''' Splits up the file name and identifies the trial number, eta, and chi values'''

            # split filename
            split_filename = filename.split('_')
            # get trial number
            trial = int(split_filename[-1].split('.')[0])
            # get eta
            eta = float(split_filename[1].split(',')[1].split('(')[1])
            chi = float(split_filename[1].split(',')[2].split(')')[0].split(' ')[1])

            return trial, eta, chi

        if "E0" in filename: # if E0, split up into eta and chi
            trial, eta, chi = split_filename()

            # chi*=-1 # have to switch sign of chi

            # calculate target rho
            targ_rho = get_E0(np.deg2rad(eta), np.deg2rad(chi))
            fidelity = get_fidelity(rho, targ_rho)

            # print results
            if verbose:
                print('trial', trial)
                print('eta', eta)
                print('chi', chi)
                print('measublue rho\n---')
                print(rho)
                print('uncertainty \n---')
                print(unc)

                print('actual rho\n ---')
                print(rho_actual)
                print('fidelity', fidelity)
                print('purity', purity)

            return trial, rho, unc, Su, rho_actual, fidelity, purity, eta, chi, angles

        else: # if not E0, just print results
            trial = int(split_filename('.')[0].split('_')[-1])
            print('measublue rho\n---')
            print(rho)
            print('uncertainty \n---')
            print(unc)
            print('actual rho\n ---')
            print(rho_actual)
            print('fidelity', fidelity)
            print('purity', purity)

            return trial, rho, unc, Su, rho_actual, fidelity, purity, angles

def analyze_rhos(filenames, settings=None, id='id'):
    '''Extending get_rho_from_file to include multiple files; 
    __
    inputs:
        filenames: list of filenames to analyze
        settings: dict of settings for the experiment
        id: str, special identifier of experiment; used for naming the df
    __
    returns: df with:
        - trial number
        - eta (if they exist)
        - chi (if they exist)
        - fidelity
        - purity
        - W theory (Purity Corrected for purity) and W expt and W unc
        - W' theory (Purity Corrected for purity) and W' expt and W' unc
    '''
    # initialize df
    df = pd.DataFrame()

    for i, file in tqdm(enumerate(filenames)):
        if settings is None:
            try:
                trial, rho, unc, Su, rho_actual, fidelity, purity, eta, chi, angles, un_proj, un_proj_unc = get_rho_from_file(file, verbose=False)
            except:
                trial, rho, unc, Su, rho_actual, fidelity, purity, angles = get_rho_from_file(file, verbose=False)
                eta, chi = None, None
        else:
            try:
                trial, rho, _, Su, rho_actual, fidelity, purity, eta, chi, angles = get_rho_from_file(file, angles = settings[i], verbose=False)
            except:
                trial, rho, _, Su, rho_actual, fidelity, purity, angles = get_rho_from_file(file, verbose=False,angles=settings[i] )
                eta, chi = None, None


        # calculate W and W' theory
        W_T_ls = compute_witnesses(rho = rho_actual) # theory
        W_AT_ls = compute_witnesses(rho = rho_actual, expt_purity=purity, angles=[eta, chi]) # Purity Corrected theory

        # calculate W and W' expt
        W_expt_ls = compute_witnesses(rho = rho, expt=True, counts=unp.uarray(un_proj, un_proj_unc))

        # parse lists
        W_min_T = W_T_ls[0]
        Wp_t1_T = W_T_ls[1]
        Wp_t2_T = W_T_ls[2]
        Wp_t3_T = W_T_ls[3]
        # ---- #
        W_min_AT = W_AT_ls[0]
        Wp_t1_AT = W_AT_ls[1]
        Wp_t2_AT = W_AT_ls[2]
        Wp_t3_AT = W_AT_ls[3]
        # ---- #
        # using propogated uncertainty
        try: # handle the observed difference in python 3.9.7 and 3.10
            W_min_expt = unp.nominal_values(W_expt_ls[0][0][0])
            W_min_unc = unp.std_devs(W_expt_ls[0][0][0])
        except: 
            W_min_expt = unp.nominal_values(W_expt_ls[0][0])
            W_min_unc = unp.std_devs(W_expt_ls[0][0])
        Wp_t1_expt = unp.nominal_values(W_expt_ls[1])
        Wp_t1_unc = unp.std_devs(W_expt_ls[1])
        Wp_t2_expt = unp.nominal_values(W_expt_ls[2])
        Wp_t2_unc = unp.std_devs(W_expt_ls[2])
        Wp_t3_expt = unp.nominal_values(W_expt_ls[3])
        Wp_t3_unc = unp.std_devs(W_expt_ls[3])

        if eta is not None and chi is not None:
            adj_fidelity= get_adj_fidelity(rho_actual, angles, purity)

            df = pd.concat([df, pd.DataFrame.from_records([{'trial':trial, 'eta':eta, 'chi':chi, 'fidelity':fidelity, 'purity':purity, 'AT_fidelity':adj_fidelity,
            'W_min_T': W_min_T, 'Wp_t1_T':Wp_t1_T, 'Wp_t2_T':Wp_t2_T, 'Wp_t3_T':Wp_t3_T,'W_min_AT':W_min_AT, 'W_min_expt':W_min_expt, 'W_min_unc':W_min_unc, 'Wp_t1_AT':Wp_t1_AT, 'Wp_t2_AT':Wp_t2_AT, 'Wp_t3_AT':Wp_t3_AT, 'Wp_t1_expt':Wp_t1_expt, 'Wp_t1_unc':Wp_t1_unc, 'Wp_t2_expt':Wp_t2_expt, 'Wp_t2_unc':Wp_t2_unc, 'Wp_t3_expt':Wp_t3_expt, 'Wp_t3_unc':Wp_t3_unc, 'UV_HWP':angles[0], 'QP':angles[1], 'B_HWP':angles[2]}])])

        else:
            df = pd.concat([df, pd.DataFrame.from_records([{'trial':trial, 'fidelity':fidelity, 'purity':purity, 'W_min_AT':W_min_AT, 'W_min_expt':W_min_expt, 'W_min_unc':W_min_unc, 'Wp_t1_AT':Wp_t1_AT, 'Wp_t2_AT':Wp_t2_AT, 'Wp_t3_AT':Wp_t3_AT, 'Wp_t1_expt':Wp_t1_expt, 'Wp_t1_unc':Wp_t1_unc, 'Wp_t2_expt':Wp_t2_expt, 'Wp_t2_unc':Wp_t2_unc, 'Wp_t3_expt':Wp_t3_expt, 'Wp_t3_unc':Wp_t3_unc, 'UV_HWP':angles[0], 'QP':angles[1], 'B_HWP':angles[2]}])])

    # save df
    print('saving!')
    df.to_csv(join(DATA_PATH, f'rho_analysis_{id}.csv'))

def make_plots_E0(dfname):
    '''Reads in df generated by analyze_rhos and plots witness value comparisons as well as fidelity and purity
    __
    dfname: str, name of df to read in
    num_plots: int, number of separate plots to make (based on eta)
    '''

    id = dfname.split('.')[0].split('_')[-1] # extract identifier from dfname

    # read in df
    df = pd.read_csv(join(DATA_PATH, dfname))
    eta_vals = df['eta'].unique()

    # preset plot sizes
    if len(eta_vals) == 2:
        fig, ax = plt.subplots(2, 2, figsize=(20, 10), sharex=True)
    elif len(eta_vals) == 3:
        fig, ax = plt.subplots(2, 3, figsize=(25, 10), sharex=True)

    for i, eta in enumerate(eta_vals):
        # get df for each eta
        df_eta = df[df['eta'] == eta]
        purity_eta = df_eta['purity'].to_numpy()
        fidelity_eta = df_eta['fidelity'].to_numpy()
        chi_eta = df_eta['chi'].to_numpy()
        adj_fidelity = df_eta['AT_fidelity'].to_numpy()

        # do purity and fidelity plots
        ax[1,i].scatter(chi_eta, purity_eta, label='Purity', color='gold')
        ax[1,i].scatter(chi_eta, fidelity_eta, label='Fidelity', color='turquoise')

        # plot Purity Corrected theory purity
        ax[1,i].plot(chi_eta, adj_fidelity, color='turquoise', linestyle='dashed', label='AT Fidelity')

        # extract witness values
        W_min_T = df_eta['W_min_T'].to_numpy()
        W_min_AT = df_eta['W_min_AT'].to_numpy()
        W_min_expt = df_eta['W_min_expt'].to_numpy()
        W_min_unc = df_eta['W_min_unc'].to_numpy()

        Wp_T = df_eta[['Wp_t1_T', 'Wp_t2_T', 'Wp_t3_T']].min(axis=1).to_numpy()
        Wp_AT = df_eta[['Wp_t1_AT', 'Wp_t2_AT', 'Wp_t3_AT']].min(axis=1).to_numpy()
        Wp_expt = df_eta[['Wp_t1_expt', 'Wp_t2_expt', 'Wp_t3_expt']].min(axis=1).to_numpy()
        Wp_expt_min = df_eta[['Wp_t1_expt', 'Wp_t2_expt', 'Wp_t3_expt']].idxmin(axis=1)
        Wp_unc = np.where(Wp_expt_min == 'Wp_t1_expt', df_eta['Wp_t1_unc'], np.where(Wp_expt_min == 'Wp_t2_expt', df_eta['Wp_t2_unc'], df_eta['Wp_t3_unc']))

        # plot curves for T and AT
        def sinsq(x, a, b, c, d):
            return a*np.sin(b*np.deg2rad(x) + c)**2 + d

        def line(x, a, b):
            return a*x + b
        try: # if W is really close to 0, it will have hard time fitting sinusoid, so fit line instead
            popt_W_T_eta, pcov_W_T_eta = curve_fit(sinsq, chi_eta, W_min_T)
            popt_W_AT_eta, pcov_W_AT_eta = curve_fit(sinsq, chi_eta, W_min_AT)
        except:
            popt_W_T_eta, pcov_W_T_eta = curve_fit(line, chi_eta, W_min_T)
            popt_W_AT_eta, pcov_W_AT_eta = curve_fit(line, chi_eta, W_min_AT)

        popt_Wp_T_eta, pcov_Wp_T_eta = curve_fit(sinsq, chi_eta, Wp_T)
        popt_Wp_AT_eta, pcov_Wp_AT_eta = curve_fit(sinsq, chi_eta, Wp_AT)

        chi_eta_ls = np.linspace(min(chi_eta), max(chi_eta), 1000)

        try:
            ax[0,i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W_T_eta), label='$W_T$', color='navy')
            ax[0,i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_W_AT_eta), label='$W_{AT}$', linestyle='dashed', color='blue')
        except:
            ax[0,i].plot(chi_eta_ls, line(chi_eta_ls, *popt_W_T_eta), label='$W_T$', color='navy')
            ax[0,i].plot(chi_eta_ls, line(chi_eta_ls, *popt_W_AT_eta), label='$W_{AT}$', linestyle='dashed', color='blue')
        ax[0,i].errorbar(chi_eta, W_min_expt, yerr=W_min_unc, fmt='o', color='slateblue', label='$W_{expt}$')


        ax[0,i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_Wp_T_eta), label="$W_{T}'$", color='crimson')
        ax[0,i].plot(chi_eta_ls, sinsq(chi_eta_ls, *popt_Wp_AT_eta), label="$W_{AT}'$", linestyle='dashed', color='red')
        ax[0,i].errorbar(chi_eta, Wp_expt, yerr=Wp_unc, fmt='o', color='salmon', label="$W_{expt}'$")

        ax[0,i].set_title(f'$\eta = {np.round(eta,3)}$')
        ax[0,i].set_ylabel('Witness value')
        ax[0,i].legend(ncol=3)
        ax[1,i].set_xlabel('$\chi$')
        ax[1,i].set_ylabel('Value')
        ax[1,i].legend()

        
    plt.suptitle('Witnesses for $E_0$ states, $\cos(\eta)|\Psi^+\\rangle + \sin(\eta)e^{i \chi}|\Psi^-\\rangle $')
    plt.tight_layout()
    plt.savefig(join(DATA_PATH, f'exp_witnesses_E0_{id}.pdf'))
    plt.show()

    # plot angle settings as a function of chi
    fig, ax = plt.subplots(len(eta_vals),3, figsize=(25,10))

    for i, eta in enumerate(eta_vals):
        # get df for each eta
        df_eta = df[df['eta'] == eta]
    
        UV_HWP = df_eta['UV_HWP'].to_numpy()
        QP = df_eta['QP'].to_numpy()
        B_HWP = df_eta['B_HWP'].to_numpy()

        # plot
        ax[i, 0].scatter(chi_eta, UV_HWP)
        ax[i, 1].scatter(chi_eta, QP)
        ax[i, 2].scatter(chi_eta, B_HWP)
        ax[i, 0].set_title(f'$\eta = {np.round(eta,3)}$, UV HWP')
        ax[i, 0].set_ylabel('Angle (deg)')
        ax[i, 0].set_xlabel('$\chi$')
        ax[i, 1].set_title(f'$\eta = {np.round(eta,3)}$, QP')
        ax[i, 1].set_ylabel('Angle (deg)')
        ax[i, 1].set_xlabel('$\chi$')
        ax[i, 2].set_title(f'$\eta = {np.round(eta,3)}$, B HWP')
        ax[i, 2].set_ylabel('Angle (deg)')
        ax[i, 2].set_xlabel('$\chi$')
    plt.suptitle('Angle settings for $E_0$ states, $\cos(\eta)|\Psi^+\\rangle + \sin(\eta)e^{i \chi}|\Psi^-\\rangle $')
    plt.tight_layout()
    plt.savefig(join(DATA_PATH, f'exp_angles_E0_{id}.pdf'))
    plt.show()

def analyze_diff(filenames, settings=None):
    '''Compare difference of actual and experimental density matrices for each chi and eta
    '''
    diagonal1_mag = []
    diagonal1_mag_c2 = []
    anti_diagonal_mag = []
    anti_diagonal_phase = []
    anti_diagonal_mag_c = []
    anti_diagonal_phase_c = []
    anti_diagonal_mag_c2 = []
    anti_diagonal_phase_c2 = []
    UV_HWP_ls = []
    QP_ls = []
    B_HWP_ls = []

    for i, file in tqdm(enumerate(filenames)):
        print('----------')
        print(file)
        if settings is None:
            try:
                trial, rho, unc, Su, rho_actual, fidelity, purity, eta, chi, angles, un_proj, un_proj_unc = get_rho_from_file(file, verbose=False)
            except:
                trial, rho, unc, Su, rho_actual, fidelity, purity, angles = get_rho_from_file(file, verbose=False)
                eta, chi = None, None
        else:
            try:
                trial, rho, _, Su, rho_actual, fidelity, purity, eta, chi, angles = get_rho_from_file(file, angles = settings[i], verbose=False)
            except:
                trial, rho, _, Su, rho_actual, fidelity, purity, angles = get_rho_from_file(file, verbose=False,angles=settings[i] )
                eta, chi = None, None

        # get corrected rho
        rho_adj = adjust_rho(rho, [eta, chi], purity)
        # adjust UV HWP by 1 degree
        angles_c = angles.copy()
        angles_c[0] += -.5
        angles_c[1] += -.5
        angles_c = np.deg2rad(angles_c)
        rho_adj2 = get_Jrho(angles_c)
        rho_adj2 = adjust_rho(rho_adj2, [eta, chi], purity)

        if i == 0: # set initial value of eta
            eta_0 = eta
            chi_ls = [chi]
            fidelity_ls = [fidelity]
            purity_ls = [purity]
            fidelity_adj_ls = [get_adj_fidelity(rho_actual, angles, purity)]
            purity_adj_ls = [get_purity(rho_adj)]
            fidelity_adj2_ls = [get_fidelity(rho_adj2, rho)]
            purity_adj2_ls = [get_purity(rho_adj2)]
            UV_HWP_ls = [angles[0]]
            QP_ls = [angles[1]]
            B_HWP_ls = [angles[2]]
        if eta != eta_0 or i==len(filenames)-1: # if different eta, reset chi_ls
            # plot the magnitudes and phase
            fig, ax = plt.subplots(2,4, figsize=(20,10), sharex=True)
            ax[0,0].scatter(chi_ls, diagonal1_mag, label='Actual')
            ax[0,0].scatter(chi_ls, diagonal1_mag_c2, label='UVHWP-0.5 QP-0.5 + Purity')
            ax[0,0].set_title('Diagonal 1 Magnitude')
            ax[0,0].legend()
            ax[0,1].scatter(chi_ls, anti_diagonal_mag, label='Actual')
            ax[0,1].scatter(chi_ls, anti_diagonal_mag_c, label='Purity Corrected')
            ax[0,1].scatter(chi_ls, anti_diagonal_mag_c2, label='UVHWP-0.5 QP-0.5 + Purity')
            ax[0,1].legend()
            ax[0,1].set_title('Anti-Diagonal Magnitude')
            ax[0,2].scatter(chi_ls, anti_diagonal_phase, label='Actual')
            ax[0,2].scatter(chi_ls, anti_diagonal_phase_c, label='Purity Corrected')
            ax[0,2].scatter(chi_ls, anti_diagonal_phase_c2, label='UVHWP-0.5 QP-0.5 + Purity')
            ax[0,2].legend()
            ax[0,2].set_title('Anti-Diagonal Phase')
            ax[0,3].scatter(chi_ls, fidelity_ls, label='Actual')
            ax[0,3].scatter(chi_ls, fidelity_adj_ls, label='Purity Corrected')
            ax[0,3].scatter(chi_ls, fidelity_adj2_ls, label='UVHWP-0.5 QP-0.5 + Purity')
            ax[0,3].legend()
            ax[0,3].set_title('Fidelity')
            ax[1,3].scatter(chi_ls, purity_ls, label='Actual')
            ax[1,3].scatter(chi_ls, purity_adj_ls, label='Purity Corrected')
            ax[1,3].scatter(chi_ls, purity_adj2_ls, label='UVHWP-0.5 QP-0.5 + Purity')
            ax[1,3].legend()
            ax[1,3].set_title('Purity')
            ax[1,0].scatter(chi_ls, UV_HWP_ls)
            ax[1,0].set_title('UV HWP')
            ax[1,1].scatter(chi_ls, QP_ls)
            ax[1,1].set_title('QP')
            ax[1,2].scatter(chi_ls, B_HWP_ls)
            ax[1,2].set_title('B HWP')

            ax[0,0].set_ylabel('$\\frac{r_{\\rho_{Th}}}{r_{\\rho_{Expt}}}$')
            ax[0,1].set_ylabel('$\\frac{r_{\\rho_{Th}}}{r_{\\rho_{Expt}}}$')
            ax[0,2].set_ylabel('$\\frac{\\theta_{\\rho_{Th}}} {\\theta_{\\rho_{Expt}}}$')
            ax[1,0].set_xlabel('$\chi$')
            ax[1,1].set_xlabel('$\chi$')
            ax[1,2].set_xlabel('$\chi$')
            ax[1,3].set_xlabel('$\chi$')
            plt.suptitle(f'Differences for $\eta = {np.round(eta_0,3)}$')
            plt.tight_layout()
            plt.savefig(join(DATA_PATH, 'diff_r_phi', f'analysis_{eta_0}.pdf'))
            

            # reset vals
            eta_0 = eta
            chi_ls = [chi]
            fidelity_ls = [fidelity]
            purity_ls = [purity]
            diagonal1_mag = []
            anti_diagonal_mag = []
            anti_diagonal_phase = []
            anti_diagonal_mag_c = []
            anti_diagonal_phase_c = []
            anti_diagonal_mag_c2 = []
            anti_diagonal_phase_c2 = []
            diagonal1_mag_c2 = []
            fidelity_adj_ls = [get_adj_fidelity(rho_actual, angles, purity)]
            purity_adj_ls = [get_purity(rho_adj)]
            fidelity_adj2_ls = [get_fidelity(rho_adj2, rho)]
            purity_adj2_ls = [get_purity(rho_adj2)]
            UV_HWP_ls = [angles[0]]
            QP_ls = [angles[1]]
            B_HWP_ls = [angles[2]]
            
        elif i!=0:
            chi_ls.append(chi)
            fidelity_ls.append(fidelity)
            purity_ls.append(purity)
            fidelity_adj_ls.append(get_adj_fidelity(rho_actual, angles, purity))
            purity_adj_ls.append(get_purity(rho_adj))
            fidelity_adj2_ls.append(get_fidelity(rho_adj2, rho))
            purity_adj2_ls.append(get_purity(rho_adj2))
            UV_HWP_ls.append(angles[0])
            QP_ls.append(angles[1])
            B_HWP_ls.append(angles[2])
        # take difference of actual and experimental density matrices
        phi_act = np.angle(rho_actual, deg=True)
        phi = np.angle(rho, deg=True)
        # get magnitude diff
        rho_actual_mag = np.abs(rho_actual)
        rho_mag = np.abs(rho)
        
        diff_r = rho_actual_mag / rho_mag
        diff_phi = phi_act  / phi

        # log diagonal magntidues and anti-diagonal magnitude
        # log antidiagonal phase
        diagonal1_mag.append(diff_r[1,1])
        anti_diagonal_mag.append(diff_r[1,2])
        anti_diagonal_phase.append(diff_phi[1,2])

        #### correction ####
        phi_adj = np.angle(rho_adj, deg=True)
        rho_adj_mag = np.abs(rho_adj)

        diff_r_adj = rho_adj_mag / rho_mag
        diff_phi_adj = phi_adj  / phi

        anti_diagonal_mag_c.append(diff_r_adj[1,2])
        anti_diagonal_phase_c.append(diff_phi_adj[1,2])

        #### correction 2 ####
        phi_adj2 = np.angle(rho_adj2, deg=True)
        rho_adj_mag2 = np.abs(rho_adj2)

        diff_r_adj2 = rho_adj_mag2 / rho_mag
        diff_phi_adj2 = phi_adj2  / phi

        anti_diagonal_mag_c2.append(diff_r_adj2[1,2])
        anti_diagonal_phase_c2.append(diff_phi_adj2[1,2])
        diagonal1_mag_c2.append(diff_r_adj2[1,1])
        
        fig, ax = plt.subplots(1,2, figsize=(20,10))
        sns.heatmap(diff_r, cmap='coolwarm', annot=True, fmt='.2f', ax=ax[0])
        ax[0].set_title('Magnitude Ratio $\\frac{r_{\\rho_{Th}}}{r_{\\rho_{Expt}}}$')
        sns.heatmap(diff_phi, cmap='coolwarm', annot=True, fmt='.2f', ax=ax[1])

        ax[1].set_title('Phase Difference $\\frac{\\theta_{\\rho_{Th}}} {\\theta_{\\rho_{Expt}}}$')
        plt.suptitle(f'Matrix for $\eta = {np.round(eta,3)}, \chi={np.round(chi,3)}$')
        plt.tight_layout()
        plt.savefig(join(DATA_PATH, 'diff_r_phi', f'diff_{eta}_{chi}.pdf'))

def det_offsets(filenames, N=1000, zeta=1, f=.2, loss_lim = 1e-6):
    '''Determine offsets in UV HWP, QP, and B HWP that minimize the loss function (sum of squares of fidelity differences)
    --
    Params:
    filenames: list of str, filenames of rho files to use
    N: int, number of iterations to run
    zeta: float, learning rate
    f: float, fraction of times to exit gd and do random
    loss_lim: float, loss limit to exit gd
    
    '''
    def get_new_fidelity(x0, angles, purity, eta, chi, rho):
        a, b, c = x0 # offsets for UV HWP, QP, B HWP
        angles_c = angles.copy()
        angles_c[0] += a
        angles_c[1] += b
        angles_c[2] += c
        rho_adj = get_Jrho(np.deg2rad(angles_c))
        rho_adj = adjust_rho(rho_adj, [eta, chi], purity)
        return get_fidelity(rho_adj, rho)

    def loss_func(x0):
        '''Helper function to compute loss between adjusted rho and rho_actual'''
        a, b, c = x0 # offsets for UV HWP, QP, B HWP
        fidelity_ls = []
        # populate lists
        for file in filenames:
            trial, rho, unc, Su, rho_actual, fidelity, purity, eta, chi, angles, un_proj, un_proj_unc = get_rho_from_file(file, verbose=False)

            fidelity_ls.append(abs(fidelity - get_new_fidelity(x0, angles, purity, eta, chi, rho)))

        # calculate loss
        fidelity_ls = np.array(fidelity_ls)
        return np.sqrt(np.sum(fidelity_ls**2))

    # optimize
    def get_random_offset():
        '''Helper function to get random offset, in degrees'''
        return np.random.rand(3)*10 - 5 # random offset between -5 and 5 degrees
    
    # get initial random guess
    x0 = get_random_offset()
    best_loss = loss_func(x0)
    grad_offset = x0
    best_offset = x0

    n = 0
    index_since_improvement = 0
    while n < N and abs(best_loss) > loss_lim:
        # get new x0
        if index_since_improvement == (f*N): # periodic random search (hop)
            x0 = get_random_offset()
            grad_offset = x0
            print('Random search...')
        else:
            gradient = approx_fprime(grad_offset, loss_func, epsilon=1e-8) # epsilon is step size in finite difference
            # if verbose: print(gradient)
            # update angles
            x0 = [best_offset[i] - zeta*gradient[i] for i in range(len(best_offset))]
            grad_offset = x0

        # minimize angles
        soln = minimize(loss_func, x0)

        # update best loss and best x0
        x = soln.x
        loss = soln.fun
        if abs(loss) < abs(best_loss):
            best_loss = loss
            best_offset = x0
            index_since_improvement = 0
        else:
            index_since_improvement += 1
        n += 1
        print(f'Iteration {n}: loss = {loss}, best loss = {best_loss}')

    print('Best loss: ', best_loss)
    print('Best offset: ', best_offset)
            
    
  

if __name__ == '__main__':
    # set filenames for computing W values
    ## new names ##
    filenames_45 = ["rho_('E0', (45.0, 0.0))_33.npy", "rho_('E0', (45.0, 18.0))_33.npy", "rho_('E0', (45.0, 36.0))_33.npy", "rho_('E0', (45.0, 54.0))_33.npy", "rho_('E0', (45.0, 72.0))_33.npy", "rho_('E0', (45.0, 90.0))_33.npy"]
    filenames_30= ["rho_('E0', (29.999999999999996, 0.0))_34.npy", "rho_('E0', (29.999999999999996, 18.0))_34.npy", "rho_('E0', (29.999999999999996, 36.0))_34.npy", "rho_('E0', (29.999999999999996, 54.0))_34.npy", "rho_('E0', (29.999999999999996, 72.0))_34.npy", "rho_('E0', (29.999999999999996, 90.0))_34.npy"]
    filenames_60 = ["rho_('E0', (59.99999999999999, 0.0))_32.npy", "rho_('E0', (59.99999999999999, 18.0))_32.npy", "rho_('E0', (59.99999999999999, 36.0))_32.npy", "rho_('E0', (59.99999999999999, 54.0))_32.npy", "rho_('E0', (59.99999999999999, 72.0))_32.npy", "rho_('E0', (59.99999999999999, 90.0))_32.npy"]
    filenames = filenames_45 + filenames_30 + filenames_60

    ## old ##
    # filenames_45 = ["rho_('E0', (45.0, 0.0))_20.npy", "rho_('E0', (45.0, 18.0))_20.npy", "rho_('E0', (45.0, 36.0))_20.npy", "rho_('E0', (45.0, 54.0))_20.npy", "rho_('E0', (45.0, 72.0))_20.npy", "rho_('E0', (45.0, 90.0))_20.npy"]
    # filenames_60= ["rho_('E0', (59.99999999999999, 0.0))_22.npy", "rho_('E0', (59.99999999999999, 18.0))_22.npy", "rho_('E0', (59.99999999999999, 36.0))_22.npy", "rho_('E0', (59.99999999999999, 54.0))_22.npy", "rho_('E0', (59.99999999999999, 72.0))_22.npy", "rho_('E0', (59.99999999999999, 90.0))_22.npy"]

    # filenames = filenames_45 + filenames_60

    # settings_45 = [[45.0,13.107759739471968,45.0], [40.325617881787,32.45243475604995,45.0], [35.319692011068646,32.80847131578413,45.0], [29.99386625322187,32.59712114540248,45.0], [26.353505137451158,32.91656908476468,44.71253931908844], [20.765759133476752,32.763298596034836,45.0]]
    # settings_60 = [[36.80717351236577,38.298986094951985,45.0], [35.64037134135345,36.377936778443754,44.99999], [32.421520781235735,35.46619180422062,44.99998], [28.842682522467676,34.97796909446873,44.61235], [25.8177216842833,34.72228985431089,44.74163766], [21.614459228879422,34.622127766985436,44.9666]]
    # settings = settings_45 + settings_60
    # analyze rho files
    # id = 'richard'
    
    # id = 'neg3_cor_unc'
    # # analyze_rhos(filenames,id=id)

    # make_plots_E0(f'rho_analysis_{id}.csv')

    # get_rho_from_file_depricated("rho_('PhiP',)_19.npy", PhiP)

    # analyze_diff(filenames)
    det_offsets(filenames)



    '''
        a, b, c = x # offsets for UV HWP, QP, B HWP
        diag_mag_ls = []
        anti_diag_mag_ls = []
        anti_diag_phase_ls = []
        diag_mag_c_ls = []
        anti_diag_mag_c_ls = []
        anti_diag_phase_c_ls = []
        # populate lists
        for file in filenames:
            trial, rho, unc, Su, rho_actual, fidelity, purity, eta, chi, angles, un_proj, un_proj_unc = get_rho_from_file(file, verbose=False)
            angles_c = angles.copy()
            angles_c[0] += a
            angles_c[1] += b
            angles_c[2] += c
            rho_adj = get_Jrho(np.deg2rad(angles_c))
            rho_adj = adjust_rho(rho_adj, [eta, chi], purity)

            # take difference of actual and experimental density matrices

            # get magnitude diff
            rho_actual_mag = np.abs(rho_actual)
            rho_mag = np.abs(rho)
            phi_act = np.angle(rho_actual, deg=True)
            phi = np.angle(rho, deg=True)

            diff_r = rho_actual_mag / rho_mag
            diff_phi = phi_act  / phi

            # log diagonal magntidues and anti-diagonal magnitude
            # log antidiagonal phase
            diag_mag_ls.append(diff_r[1,1])
            anti_diag_mag_ls.append(diff_r[1,2])
            anti_diag_phase_ls.append(diff_phi[1,2])

            #### correction ####
            phi_adj = np.angle(rho_adj, deg=True)
            rho_adj_mag = np.abs(rho_adj)

            diff_r_adj = rho_adj_mag / rho_mag
            diff_phi_adj = phi_adj  / phi

            diag_mag_c_ls.append(diff_r_adj[1,1])
            anti_diag_mag_c_ls.append(diff_r_adj[1,2])
            anti_diag_phase_c_ls.append(diff_phi_adj[1,2])
    '''