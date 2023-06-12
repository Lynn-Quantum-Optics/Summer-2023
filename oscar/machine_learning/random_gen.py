# file to hold methods for random generation of density matrices
import numpy as np

## note: get_random_jones is in jones.py ##
from rho_methods import *
from sample_rho import *

def get_random_simplex():
    '''
    Returns density matrix for random state of form:
    a|HH> + be^(i*beta)|01> + ce^(i*gamma)*|10> + de^(i*delta)*|11>
    along with generation parameters [a, b, c, d, beta, gamma, delta]
    '''

    def get_random_state():
    
        a = np.random.rand()
        b = np.random.rand() *(1-a)
        c = np.random.rand()*(1-a-b)
        d = 1-a-b-c

        real_ls = [] # list to store real coefficients
        real_ls = np.sqrt(np.array([a, b, c, d]))
        np.random.shuffle(real_ls)
        rand_angle=np.random.rand(3)*2*np.pi

        params = np.concatenate((real_ls, rand_angle))

        state_vec = np.multiply(real_ls, np.e**(np.concatenate((np.array([1]), rand_angle))*1j)).reshape((4,1))
        # compute density matrix
        rho = get_rho(state_vec)
        return rho, params
    
    rho, params = get_random_state()
    while not(is_valid_rho(rho)):
        rho, params = get_random_state()

    return [rho, params]

### randomization functions ###
def get_random_roik(purity_cond = 1):
    ''' Function to generate random density matrix with roik method.
    params:
        purity_condition: if not None, will generate random density matrix until purity is less than this value
    '''
    ## method 1: random diagonal elements ##
    def rand_diag():
        # get 4 random params
        def get_rand_elems():
            M11 = np.random.rand()
            M22 = np.random.rand()*(1-M11)
            M33 = np.random.rand()*(1-M11 - M22)
            M44 = np.random.rand()*(1-M11-M22-M33)
            
            # shuffle the entries
            rand_elems = np.array([M11, M22, M33, M44])
            np.random.shuffle(rand_elems)
            return M11+M22+M33+M44, rand_elems[0], rand_elems[1], rand_elems[2], rand_elems[3]
        N = 0 # set to 0 initially
        while N ==0:
            N, M11, M22, M33, M44 = get_rand_elems()
            # define new 4x4 matrix; cast the diagonal array as matrix for more convienient fnuctionality
        M = np.matrix(np.diag([M11/N, M22/N, M33/N, M44/N]))
        return M

    ## method 2: random unitary trans ##
    def rand_unitary():
        # need to first generate 6 other smaller unitaries
        tau = np.pi*2
        e = np.e
        ra = np.pi/2
        def get_rand_elems():
            alpha = np.random.rand()*tau
            psi = np.random.rand()*tau
            chi = np.random.rand()*tau
            phi = np.random.rand()*ra

            return np.matrix([
                [e**(psi*1j)*np.cos(phi), e**(chi*1j)*np.sin(phi)],
                [-e**(-chi*1j)*np.sin(phi), e**(-psi*1j)*np.cos(phi)]
            ])*e**(alpha*1j)

        # loop and create unitaries from blocks
        unitary_final = np.eye(4)
        for k in range(5, -1, -1): # count down to do multiplicatiom from right to left
            sub_unitary_k = get_rand_elems()
            if k==0 or k==3 or k==5:
                unitary_k = np.matrix(np.block([[np.eye(2), np.zeros((2,2))], [np.zeros((2,2)), sub_unitary_k]]))
            elif k==1 or k==4:
                ul = np.matrix([[1,0], [0, sub_unitary_k[0, 0]]])# upper left
                ur = np.matrix([[0,0], [sub_unitary_k[0, 1], 0]])# upper right
                ll = np.matrix([[0,sub_unitary_k[1,0]], [0, 0]])# lower left
                lr = np.matrix([[sub_unitary_k[1,1], 0], [0, 1]])# lower right
                unitary_k = np.matrix(np.block([[ul, ur],[ll, lr]]))
            else: # k==2
                unitary_k = np.matrix(np.block([[sub_unitary_k, np.zeros((2,2))], [np.zeros((2,2)), np.eye(2)]]))
            
            unitary_final = unitary_final @ unitary_k

        return unitary_final

    ## combine M and U as follows: U M U^\dagger
    def combine_rand(): 
        return U @ M @ U.H # .H does conjugate transpose

    M = rand_diag()
    U = rand_unitary()
    M0 = combine_rand()

    while not(is_valid_rho(M0)) or get_purity(M0)>purity_cond: # if not valid density matrix, keep generating
        M = rand_diag()
        U = rand_unitary()
        M0 = combine_rand()

    return M0, get_purity(M0)

def get_random_E0():
    ''' Using the defintion of E0 in sample_rho.py'''

    # get random params
    def get_rand_state():
        eta = np.random.rand()*np.pi/2
        chi = np.random.rand()*2*np.pi

        E0 = get_E0(eta, chi)
        angles = [eta, chi]
        return E0, angles
    E0, angles = get_rand_state()
    while not(is_valid_rho(E0)):
        E0, angles = get_rand_state()

    return [E0,angles]

def get_random_E1():
    ''' Using the defintion of E1 in sample_rho.py'''

    # get random params
    def get_rand_state():
        eta = np.random.rand()*np.pi/2
        chi = np.random.rand()*2*np.pi

        E1 = get_E1(eta, chi)
        angles = [eta, chi]
        return E1, angles
    E1, angles = get_rand_state()
    while not(is_valid_rho(E1)):
        E1, angles = get_rand_state()

    return [E1,angles]
