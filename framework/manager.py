import json
import time
import csv
import os
import datetime
import numpy as np
import scipy.stats as stats
from ccu_controller import CCU
from motor_drivers import ElliptecMotor, ThorLabsMotor, COM_PORTS

class Manager:
    ''' Class for managing the automated laboratory equiptment.

    Parameters
    ----------

    '''
    def __init__(self, out_file=None, config='config.json'):
        # load configuration file
        with open(config, 'r') as f:
            self._config = json.load(f)
        
        # check for duplicate output file
        if out_file is not None and os.path.isfile(out_file):
            print(f'File {out_file} already exists. A timestamped csv will be saved instead.')
            out_file = None
        
        # get a safe output file
        if out_file is None:
            out_file = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv")
        
        # open output file and setup output file writer
        self.out_file = open(out_file, 'w+', newline='')
        self.out_writer = csv.writer(self.out_file)
        
        # get motor names
        self.motors = list(self._config['motors'].keys())
        
        # initialize the CCU
        self._ccu = CCU(
            self._config['ccu']['port'],
            self._config['ccu']['baudrate'])

        # write the column headers
        self.out_writer.writerow(\
            [f'start time (s)', 'stop time (s)'] + \
            ['num samples (#)', 'period per sample (s)'] + \
            [f'{m} position (rad)' for m in self.motors] + \
            [f'{k} rate (#/s)' for k in self._ccu.CHANNEL_KEYS] + \
            [f'{k} rate unc (#/s)' for k in self._ccu.CHANNEL_KEYS])
        
        # initialize all of the motors as objects in the manager
        for motor_name in self.motors:
            if self._config['motors'][motor_name]['type'] == 'Elliptec':
                self.__dict__[motor_name] = ElliptecMotor(
                    self._config['motors'][motor_name]['port'],
                    self._config['motors'][motor_name]['address'].encode('utf-8'))
            elif self._config['motors'][motor_name]['type'] == 'ThorLabs':
                self.__dict__[motor_name] = ThorLabsMotor(
                    self._config['motors'][motor_name]['sn'])

    def take_data(self, num_samp:int, samp_period:float) -> None:
        # record all motor positions
        motor_positions = [self.__dict__[m].get_position() for m in self.motors]

        # record start time
        start_time = time.time()

        # run trials
        data = np.row_stack([self._ccu.get_count_rates(samp_period) for _ in range(num_samp)])
        
        # record stop time
        stop_time = time.time()

        # calculate averages and uncertainties
        data_avg = np.mean(data, axis=0)
        data_unc = stats.sem(data, axis=0)

        # record data
        self.out_writer.writerow(\
            [start_time, stop_time] + \
            [num_samp, samp_period] + \
            motor_positions + \
            list(data_avg) + \
            list(data_unc))

    def configure_motors(self, **kwargs) -> None:
        ''' Configure the position of multiple motors at a time

        Parameters
        ----------
        kwargs : <NAME OF MOTOR> = <SET POSITION IN RADIANS>
            Assign each motor name that you wish to move the absolute angle to which you want it to move, in radians.
        '''
        for motor_name, position in kwargs.items():
            if not motor_name in self.motors:
                raise ValueError(f'Attempted to reference unknown motor \"{motor_name}\".')
            self.__dict__[motor_name].rotate_absolute(position)

    def set_meas_basis(self, basis:str) -> None:
        ''' Set the measurement basis for Alice and Bob's half and quarter wave plates. 
        
        Parameters
        ----------
        basis : str
            The measurement basis to set, should have length two. All options are listed in the config.
        '''
        # setup the basis
        A, B = basis
        self.configure_motors(
            A_HWP=self._config['basis_presets']['A_HWP'][A],
            A_QWP=self._config['basis_presets']['A_QWP'][A],
            B_HWP=self._config['basis_presets']['B_HWP'][B],
            B_QWP=self._config['basis_presets']['B_QWP'][B])

    def close(self) -> None:
        # close the output file (writes lines)
        self.out_file.close()
        # close all of the COM_PORT connections
        for port in COM_PORTS.values():
            port.close()
        # close the CCU connection
        self._ccu.close()
