# file to generate systems by enforcing orthogonality of measured states
import numpy as np
import time
from itertools import combinations

class HyperBell():
    '''HyperBell class to generate systems of measured bell states for a given dimension d and number of states in group k.'''

    def init(self, d, k):
        '''Initialize HyperBell class with dimension d and number of states in group k.'''
        self.d = d
        self.k = k

        print('Initializing HyperBell class with d = {} and k = {}.'.format(d,k))

        self.soln_limit = 2*d # must find solutions for all detectors to fuflill sufficient condition

        self.num_coeff = 4*d # number of coefficients in measurement operator

        self.bounds = [(-1,1) for _ in range(self.num_coeff)] # set bounds for coefficients

        self.num_attempts=2*self.soln_limit # number of attempts to find solutions

        self.precision = 6 # precision for soluttion vec and inner product to be 0
        self.not0_precision = 3 # precision for input vector to not be 0

        self.k_groups_indices = self.get_all_k_indices() # initialize all k-groups of bell states
        self.num_ksys = len(self.k_groups) # number of k-systems

        self.start_time = time.time() # start timer

        self.set_m(0)

    def construct_v(self, coeff):
        '''Returns vector of a_i + i*b_i'''
        v_c = np.zeros((2*self.d, 1), dtype='complex128')
        for o in range(2*self.d):
            v_c[o] = coeff[o] + 1j*coeff[o+2*self.d]
        return v_c

    def get_bell(self, c, p):
        '''Define bell state for given correlation and phase class c, p and dimension d, represented in number basis.
        --
        Params: 
            c (int): correlation class
            p (int): phase class
        --
        Returns:
            bell (np.array): bell state in number basis, 2d x1 vector
        '''
        d = self.d
        bell = np.zeros((2*d,1), dtype='complex128')
        for l in range(d):
            phase = np.exp(1j*2*np.pi * p *l / d)
            numb = np.zeros((2*d,1)) # number basis portion
            numb[l]=1
            numb[d+(l+c)%d]=1
            bell += phase * numb
        return bell * 1/np.sqrt(d) # normalize)

    def get_all_k_indices(self):
        '''Returns all unique k-groups of bell states for a given dimension d.'''
        k_groups_indices = list(combinations(np.arange(0, self.d**2),self.k))
        return k_groups_indices

    def get_m_kgroup(self):
        '''Returns mth k-group of bell states.'''
        k_group_indices = self.k_groups_indices[self.m]
        k_group = []
        for i in k_group_indices:
            c = i // self.d
            p = i % self.d
            k_group.append(self.get_bell(c,p))
        return k_group

    def get_meas(self, bell):
        '''Performs LELM measurement on bell state b.
        --
        Params:
            bell (np.array): bell state in number basis, 2d x1 vector
        --
        Returns:
            meas (func): returns an array taking as input coefficients which in turn return the measured state in number basis, 2d x1 vector; note coefficients are split into real and imaginary parts to make solving process easier
        --
        '''
        d = self.d
        # define measurement coeffients
        def meas(coeff):
            bell_m = np.zeros((2*d,1), dtype='complex128') # initialize measured bell state
            for l in range(2*d):
                if bell[l]!=0: # if state is occupied
                    bell_c = bell.copy()
                    bell_c[l]=0 # annihilate state
                    bell_m += (coeff[l] + 1j*coeff[2*d+l]) * bell_c # mutliply by coefficient
            return bell_m

        return meas

    def get_meas_ip(self, bell_ls, coeff):
        ''' Computes inner product between two measured bell states'''
        # unpack bell states ls
        bell1, bell2 = bell_ls
        # get measurement functions
        bell1_meas_func = self.get_meas(bell1)
        bell2_meas_func = self.get_meas(bell2)
        # get measured states
        bell1_meas = bell1_meas_func(coeff)
        bell2_meas = bell2_meas_func(coeff)
        # print(bell1_meas)
        # print(bell2_meas)
        # compute inner product
        ip = np.conj(bell1_meas).T @ bell2_meas
        ip = ip[0][0] # extract scalar from array
        return ip

    def set_m(self, m):
        '''Sets m to be the number of the k-system under investigation.'''
        self.m = m

    def set_num_attempts(self, num_attempts):
        '''Sets number of attempts to find solutions.'''
        self.num_attempts = num_attempts

    def set_precision(self, precision):
        '''Sets precision as the negative exponent for declaring solutions in terms of 
            - func vector = 0
            - inner product = 0
        '''
        self.precision = precision

    def set_not0_precision(self, not0_precision):
        '''Sets precision as the negative exponent for declaring solutions in terms of 
            - func input != 0
        '''
        self.not0_precision = not0_precision

    def get_ksys(self, coeff):
        '''Function to return the mth k-system as a function of 4*d real coefficients.
        --
        Params:
            i (int): index of system in 
            coeff (np.array, 4*d x 1): coefficients for measurement
        '''
        k_group = self.get_m_kgroup() # get ith k-group
        # take inner product of all pairs of k bell states
        k_sys = [] # list of all inner products for a given group, which make up a system
        for i in range(len(k_group)):
            for j in range(i+1,len(k_group)):
                k_sys.append(self.get_meas_ip(coeff=coeff, bell_ls = [k_group[i], k_group[j]]))
        return np.array(k_sys)

    def get_allsys_func(self):
        '''Returns a vector of all measured states as a function of 4*d real coefficients.
        --
        Params:
            d (int): dimension of system
            k (int): number of states in group
            coeff (np.array, 4*d x 1): coefficients for measurement
        '''
        all_sys = []
        for i in range(len(self.k_groups)):
            all_sys.append(self.get_ksys(i))
        return np.array(all_sys)

    def rand_guess(self):
        '''Returns random simplex guess for coefficients'''
        d = self.d
        # uses stick method
        rand = 2*np.random.rand(4*d-1)
        rand = np.sort(rand)
        guess = np.zeros(4*d)
        guess[0] = rand[0] # set initial guess
        for i in range(1, len(rand), 1):
            guess[i] = rand[i] - rand[i-1] # set remaining guesses by taking difference, i.e. "lengths" of sticks
        guess[-1] = 2 - rand[-1] # set final guess
        guess -= 1 # shift so that guess is centered around 0
        # guess = np.sqrt(guess) # since we want the sum of squares to be 1
        # print('random guess: ', guess)
        return guess

if __name__ == '__main__':
    # test
    d = 2
    k = 3
    hb = HyperBell()
    hb.init(d,k)
    coeff=np.array([-0.9496828 , -0.84160771, -0.53092842, -0.83225099, -0.90072945,
       -0.35618569, -0.86866719, -0.71994774])
    print(hb.get_meas(hb.get_bell(c=0, p=0))(coeff))
    print(hb.get_ksys(coeff))
    print(hb.get_ksys(hb.rand_guess()))
    # print(hb.get_ksys_func(0)())
    # print(hb.get_ksys(0, hb.rand_guess()))

    def meas(bell, coeff):
        bell_m = np.zeros((2*d,1), dtype='complex128') # initialize measured bell state
        print(bell)
        for l in range(2*d):
            if bell[l]!=0: # if state is occupied
                bell_c = bell.copy()
                bell_c[l]=0 # annihilate state
                bell_m += (coeff[l] + 1j*coeff[2*d+l]) * bell_c # mutliply by coefficient
        print(bell_m)
        return bell_m
    meas(hb.get_bell(c=0, p=0), coeff)

