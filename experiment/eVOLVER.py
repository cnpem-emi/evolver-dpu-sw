#!/usr/bin/env python3

import os
import sys
import time
import pickle
import shutil
import logging
import argparse
import numpy as np
import json
import select
import socket
import traceback
import asyncio
import redis
from consts import functions
from threading import Thread, Lock


import custom_script
from custom_script import EXP_NAME
from custom_script import OPERATION_MODE
from custom_script import STIR, TEMP

# Should not be changed
VIALS = [x for x in range(16)]


SAVE_PATH = os.path.dirname(os.path.realpath(__file__))
EXP_DIR = os.path.join(SAVE_PATH, EXP_NAME)
OD_CAL_PATH = os.path.join(SAVE_PATH, 'od_cal.json')
TEMP_CAL_PATH = os.path.join(SAVE_PATH, 'temp_cal.json')
PUMP_CAL_PATH = os.path.join(SAVE_PATH, 'pump_cal.json')
JSON_PARAMS_FILE = os.path.join(SAVE_PATH, 'eVOLVER_parameters.json')
CHANNEL_INDEX_PATH = os.path.join(SAVE_PATH, 'channel_index.json')


SIGMOID = 'sigmoid'
LINEAR = 'linear'
THREE_DIMENSION = '3d'

logger = logging.getLogger('eVOLVER')

paused = False



# ---- eVolver DPU entity, connects to server (hardware access) evolver-sw
global EVOLVER_NS
EVOLVER_NS = None
EVOLVER_IP = '127.0.0.1'
EVOLVER_PORT = 6001

global broadcastSocket
global broadcastReady
global lock
broadcastSocket = None
broadcastReady = False
lock = Lock() # Lock is similar to mutex. Will only 'allow' one task to use same socket at a time


# ----- socketio object
global sio

# ----- Redis database for eVolver unit
global redis_client
redis_client = redis.StrictRedis("127.0.0.1")

# ----- Smart sleeves indexes (hardware differs from sw index)
global channelIdx
with open(CHANNEL_INDEX_PATH) as f:
    channelIdx = json.load(f)



def broadcast():
    '''
    When receive new data from evolver-server:
      - Update Redis key
      - Call broadcast method for eVolver DPU, which will decide next experiment step
    '''
    global broadcastSocket
    global broadcastReady
    global redis_client
    global lock
    global EVOLVER_NS

    while True:
        while broadcastReady:
            
            ready = select.select([broadcastSocket], [], [], 2)
            if ready[0]:
                        data = broadcastSocket.recv(4096)
                        data = json.loads(data)
                        redis_client.set("broadcast", json.dumps(data))
                        EVOLVER_NS.broadcast(data)
        time.sleep(1)



class EvolverDPU():
    global broadcastSocket
    global broadcastReady
    global channelIdx
    global lock

    start_time = None
    use_blank = False
    OD_initial = None
    experiment_params = None
    ip_address = None
    exp_dir = SAVE_PATH

    def __init__(self):
        self.connect()

    def connect(self):
        '''
        Connect to evolver-server, to get/set hardware information
        '''
        global broadcastSocket
        global broadcastReady

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((EVOLVER_IP, EVOLVER_PORT))
        self.s.setblocking(0)

        broadcastSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        broadcastSocket.connect((EVOLVER_IP, EVOLVER_PORT+1000))
        broadcastSocket.setblocking(0)
        broadcastReady = True
        logger.info('connected to eVOLVER as client')

    def disconnect(self):
        self.s.close()
        broadcastSocket.close()

        logger.info('disconnected to eVOLVER as client')


    def broadcast(self, data: dict):
        '''
        This method should be called every time a new broadcast is received from evolver-server (from BBB, which communicates w/ hardware),
        which indicates that new values are available.

        This method converts raw data into real values and ask for new step from custom functions.
        '''
        print('\nBroadcast received')
        elapsed_time = round((time.time() - self.start_time) / 3600, 4)
        print('Elapsed time: %.4f hours' % elapsed_time)

        # Check if calibration files are available
        if not self.check_for_calibrations():
            print('Calibration files still missing, skipping custom '
                           'functions')
            return
        with open(OD_CAL_PATH) as f:
            od_cal = json.load(f)
        with open(TEMP_CAL_PATH) as f:
            temp_cal = json.load(f)


        # Apply calibrations and update temperatures if needed
        data = self.transform_data(data, VIALS, od_cal, temp_cal)



        # Should we "blank" the OD?
        if self.use_blank and self.OD_initial is None:
            logger.info('setting initial OD reading')
            self.OD_initial = data['transformed']['od']
        elif self.OD_initial is None:
            self.OD_initial = np.zeros(len(VIALS))
        data['transformed']['od'] = (data['transformed']['od'] -
                                        self.OD_initial)


        # Show data in screen
        print("OD: ", data["transformed"]["od"][:8])
        print("Temp: ", data["transformed"]["temp"][:8])
        if data is None:
            logger.error('could not tranform raw data, skipping user-'
                         'defined functions')
            return


        # Save data into .csv files
        try:
            # Transformed values, true values
            self.save_data(data['transformed']['od'], elapsed_time,
                            VIALS, 'OD')
            self.save_data(data['transformed']['temp'], elapsed_time,
                            VIALS, 'temp')
            
            # Raw data
            raw_data = [0]*16
            for param in od_cal['params']:
                # Order raw data before saving values
                for ss in range(16):
                    raw_data[ss] = data['data'].get(param, [])[channelIdx[str(ss)]["channel"]]
                self.save_data(raw_data, elapsed_time, VIALS, param + '_raw')

            for param in temp_cal['params']:
                # Order raw data before saving values
                for ss in range(16):
                    raw_data[ss] = data['data'].get(param, [])[channelIdx[str(ss)]["channel"]]
                self.save_data(raw_data, elapsed_time, VIALS, param + '_raw')

        except OSError:
            logger.info("Broadcast received before experiment initialization - skipping custom function...")
            return


        # Run custom functions
        self.custom_functions(data, VIALS, elapsed_time)

        # save variables
        self.save_variables(self.start_time, self.OD_initial)

        # Restart logging for db/gdrive syncing
        logging.shutdown()
        logging.getLogger('eVOLVER')




    def activecalibrations(self, data: dict):
        print('Calibrations received')
        for calibration in data:
            if calibration['calibrationType'] == 'od':
                file_path = OD_CAL_PATH
            elif calibration['calibrationType'] == 'temperature':
                file_path = TEMP_CAL_PATH
            elif calibration['calibrationType'] == 'pump':
                file_path = PUMP_CAL_PATH
            else:
                continue
            for fit in calibration['fits']:
                if fit['active']:
                    with open(file_path, 'w') as f:
                        json.dump(fit, f)
                    # Create raw data directories and files for params needed
                    for param in fit['params']:
                        if not os.path.isdir(os.path.join(EXP_DIR, param + '_raw')) and param != 'pump':
                            os.makedirs(os.path.join(EXP_DIR, param + '_raw'))
                            for x in range(len(fit['coefficients'])):
                                exp_str = "Experiment: {0} vial {1}, {2}".format(EXP_NAME,
                                        x,
                                        time.strftime("%c"))
                                self._create_file(x, param + '_raw', defaults=[exp_str])
                    break

    def request_calibrations(self) -> dict:
        '''
        Request calibrations to evolver-server
        '''
        logger.debug('requesting active calibrations')
        lock.acquire()
        self.s.send(functions['getactivecal']['id'].to_bytes(1,'big') + b'\r\n')
        time.sleep(1)
        for _ in range(3):
            ready = select.select([self.s], [], [], 2)
            if ready[0]:
                info = json.loads(self.s.recv(30000)[:-2])
                break
            else:
                time.sleep(1)
        lock.release()
        return info


    def setrawcalibration(self, data):     
        logger.debug('setrawcalibration')
        lock.acquire()
        self.s.send(functions["setrawcalibration"]["id"].to_bytes(1,'big')+ bytes(json.dumps(data), 'utf-8') + b'\r\n')
        time.sleep(1)
        for _ in range(3):
            ready = select.select([self.s], [], [], 2)
            if ready[0]:
                info = self.s.recv(30000)[:-2]
                break
            else:
                time.sleep(1)
        lock.release()
        return info
    


    def transform_data(self, data, vials, od_cal, temp_cal):
        od_data_2 = None

        if od_cal['type'] == THREE_DIMENSION:
            od_data_2 = data['data'].get(od_cal['params'][1], None)

        od_data = data['data'].get(od_cal['params'][0], None)
        temp_data = data['data'].get(temp_cal['params'][0], None)
        temp_value = [0]*16
        od_value = [0]*16
        set_temp_data = data['config'].get('temp', {}).get('value', None)


        if od_data is None or temp_data is None or set_temp_data is None:
            print('Incomplete data recieved, Error with measurement')
            logger.error('Incomplete data received, error with measurements')
            return None
        if 'NaN' in od_data or 'NaN' in temp_data or 'NaN' in set_temp_data:
            print('NaN recieved, Error with measurement')
            logger.error('NaN received, error with measurements')
            return None

        od_data = np.array([float(x) for x in od_data])
        if od_data_2:
            od_data_2 = np.array([float(x) for x in od_data_2])
        temp_data = np.array([float(x) for x in temp_data])
        set_temp_data = np.array([float(x) for x in set_temp_data])

        temps = []
        for x in vials:
            file_name =  "vial{0}_temp_config.txt".format(x)
            file_path = os.path.join(EXP_DIR, 'temp_config', file_name)
            temp_set_data = np.genfromtxt(file_path, delimiter=',')
            temp_set = temp_set_data[len(temp_set_data)-1][1]
            temps.append(temp_set)
            od_coefficients = od_cal['coefficients'][x]
            temp_coefficients = temp_cal['coefficients'][x]
            index_value = channelIdx[str(x)]["channel"]

            try:
                if od_cal['type'] == SIGMOID:
                    #convert raw photodiode data into ODdata using calibration curve
                    od_value[x] = (od_coefficients[2] -
                                        ((np.log10((od_coefficients[1] -
                                                    od_coefficients[0]) /
                                                    (float(od_data[index_value]) -
                                                    od_coefficients[0])-1)) /
                                                    od_coefficients[3]))
                    if not np.isfinite(od_data[x]):
                        od_value[x] = np.nan
                        logger.debug('OD from vial %d: %s' % (x, od_value[x]))
                    else:
                        logger.debug('OD from vial %d: %.3f' % (x, od_value[x]))
                elif od_cal['type'] == THREE_DIMENSION:
                    od_value[x] = np.real(od_coefficients[0] +
                                        (od_coefficients[1]*od_data[index_value]) +
                                        (od_coefficients[2]*od_data_2[index_value]) +
                                        (od_coefficients[3]*(od_data[index_value]**2)) +
                                        (od_coefficients[4]*od_data[x]*od_data_2[index_value]) +
                                        (od_coefficients[5]*(od_data_2[index_value]**2)))
                else:
                    logger.error('OD calibration not of supported type!')
                    od_value[x] = np.nan
            except ValueError:
                print("OD Read Error")
                logger.error('OD read error for vial %d, setting to NaN' % x)
                od_value[x] = np.nan
            try:
                temp_value[x] = temp_data[channelIdx[str(x)]["channel"]]
                temp_value[x] = (float(temp_value[x]) *
                                temp_coefficients[0]) + temp_coefficients[1]
                #print('temperature from vial %d: %.3f' % (x, temp_value[x]))
            except ValueError:
                print("Temp Read Error")
                logger.error('temperature read error for vial %d, setting to NaN'
                            % x)
                temp_value[x]  = 'NaN'
            try:
                set_temp_data[x] = (float(set_temp_data[x]) *
                                    temp_coefficients[0]) + temp_coefficients[1]
                logger.debug('set_temperature from vial %d: %.3f' % (x,
                                                                set_temp_data[x]))
            except ValueError:
                print("Set Temp Read Error")
                logger.error('set temperature read error for vial %d, setting to NaN'
                            % x)
                set_temp_data[x]  = 'NaN'

        temps = np.array(temps)
        # update temperatures only if difference with expected
        # value is above 0.2 degrees celsius
        delta_t = np.abs(set_temp_data - temps).max()

        if delta_t < 0.2:
            logger.info('updating temperatures (max. deltaT is %.2f)' %
                        delta_t)
            coefficients = temp_cal['coefficients']
            raw_temperatures = [0]*16
            for x in vials:
                index = channelIdx[str(x)]["channel"]
                raw_temperatures[index] = str(int((temps[x] - temp_cal['coefficients'][x][1]) /
                                        temp_cal['coefficients'][x][0]))
            self.update_temperature(raw_temperatures)
        else:
            # config from server agrees with local config
            # report if actual temperature doesn't match
            delta_t = np.abs(temps - temp_data).max()
            if delta_t > 0.2:
                logger.debug('actual temperature doesn\'t match configuration '
                            '(yet? max deltaT is %.2f)' % delta_t)
                logger.debug('temperature config: %s' % temps)
                logger.debug('actual temperatures: %s' % temp_data)

        # add a new field in the data dictionary
        data['transformed'] = {}
        data['transformed']['od'] = od_value
        data['transformed']['temp'] = temp_value
        return data

    def update_stir_rate(self, stir_rates: list, immediate = True):
        '''
        Update stir values
        '''
        data = {'param': 'stir', 'value': stir_rates,
                'immediate': immediate, 'recurring': True}
        logger.debug('stir rate command: %s' % data)
        lock.acquire()
        self.s.send(functions['command']['id'].to_bytes(1,'big') + bytes(json.dumps(data), 'utf-8') + b'\r\n')
        lock.release()

    def update_temperature(self, temperatures: list, immediate = True):
        '''
        Update temperature values. Values in list should be raw values 0-4095
        '''
        data = {'param': 'temp', 'value': temperatures,
                'immediate': immediate, 'recurring': True}
        logger.debug('temperature command: %s' % data)
        lock.acquire()
        self.s.send(functions['command']['id'].to_bytes(1,'big') + bytes(json.dumps(data), 'utf-8') + b'\r\n')
        lock.release()

    def fluid_command(self, MESSAGE: list):
        '''
        Update fluids values
        '''
        logger.debug('fluid command: %s' % MESSAGE)
        command = {'param': 'pump', 'value': MESSAGE,
                   'recurring': False ,'immediate': True}
        lock.acquire()
        self.s.send(functions['command']['id'].to_bytes(1,'big') + bytes(json.dumps(data), 'utf-8') + b'\r\n')
        lock.release()

    def update_chemo(self, data: dict, vials: list, bolus_in_s: list, period_config: list, immediate = True):
        '''
        Update chemostato operations.
        Usually asked to be run in custom_script.
        Currently, only changes pumps !
        '''
        current_pump = data['config']['pump']['value']

        MESSAGE = {'fields_expected_incoming': 49,
                   'fields_expected_outgoing': 49,
                   'recurring': True,
                   'immediate': immediate,
                   'value': ['--'] * 48,
                   'param': 'pump'}

        for x in vials:
            pumpA_idx = channelIdx[str(x)]["A"]
            pumpB_idx = channelIdx[str(x)]["B"]
            pumpC_idx = channelIdx[str(x)]["C"]

            # stop pumps if period is zero
            if period_config[x] == 0:
                # influx
                MESSAGE['value'][pumpA_idx] = '0|0'
                MESSAGE['value'][pumpB_idx] = '0|0'
                # efflux
                MESSAGE['value'][pumpC_idx] = '0|0'

            else:
                # influx 1
                MESSAGE['value'][pumpA_idx] = '%.2f|%.1f' % (bolus_in_s[x], period_config[x])
                # influx 2
                MESSAGE['value'][pumpB_idx] = '%.2f|%.1f' % (bolus_in_s[x], period_config[x])
                # efflux
                MESSAGE['value'][pumpC_idx] = '%.2f|%.1f' % (bolus_in_s[x] * 3,
                                                        period_config[x])
        if True: #MESSAGE['value'] != current_pump:
            lock.acquire()
            self.s.send(functions['command']['id'].to_bytes(1,'big') + bytes(json.dumps(MESSAGE), 'utf-8') + b'\r\n')
            lock.release()

    def stop_all_pumps(self):
        '''
        Stop all pumps
        '''
        data = {'param': 'pump',
                'value': ['0'] * 48,
                'recurring': False,
                'immediate': True}
        logger.info('stopping all pumps')
        lock.acquire()
        self.s.send(functions['command']['id'].to_bytes(1,'big') + bytes(json.dumps(data), 'utf-8') + b'\r\n')
        lock.release()


    def _create_file(self, vial, param, directory=None, defaults=None):
        '''
        Create file for data saving, if needed!
        '''
        if defaults is None:
            defaults = []
        if directory is None:
            directory = param
        file_name =  "vial{0}_{1}.txt".format(vial, param)
        file_path = os.path.join(EXP_DIR, directory, file_name)
        text_file = open(file_path, "w")
        for default in defaults:
            text_file.write(default + '\n')
        text_file.close()

    def initialize_exp(self, vials, experiment_params, log_name, quiet, verbose, always_yes = False):
        self.experiment_params = experiment_params
        logger.info('initializing experiment')

        if os.path.exists(EXP_DIR):
            setup_logging(log_name, quiet, verbose)
            logger.info('found an existing experiment')
            exp_continue = None
            if always_yes:
                exp_continue = 'y'
            else:
                while exp_continue not in ['y', 'n']:
                    exp_continue = 'n' #input('Continue from existing experiment? (y/n): ')
        else:
            exp_continue = 'n'

        if exp_continue == 'n':
            if os.path.exists(EXP_DIR):
                exp_overwrite = None
                if always_yes:
                    exp_overwrite = 'y'
                else:
                    while exp_overwrite not in ['y', 'n']:
                        exp_overwrite = 'y' #input('Directory aleady exists. '
                                            #'Overwrite with new experiment? (y/n): ')
                logger.info('data directory already exists')
                if exp_overwrite == 'y':
                    logger.info('deleting existing data directory')
                    shutil.rmtree(EXP_DIR)
                else:
                    print('Change experiment name in custom_script.py '
                        'and then restart...')
                    logger.warning('not deleting existing data directory, exiting')
                    sys.exit(1)

            start_time = time.time()

            self.request_calibrations()

            logger.debug('creating data directories')
            os.makedirs(os.path.join(EXP_DIR, 'OD'))
            os.makedirs(os.path.join(EXP_DIR, 'od_135_raw'))
            os.makedirs(os.path.join(EXP_DIR, 'temp'))
            os.makedirs(os.path.join(EXP_DIR, 'temp_raw'))
            os.makedirs(os.path.join(EXP_DIR, 'temp_config'))
            os.makedirs(os.path.join(EXP_DIR, 'pump_log'))
            os.makedirs(os.path.join(EXP_DIR, 'ODset'))
            os.makedirs(os.path.join(EXP_DIR, 'growthrate'))
            os.makedirs(os.path.join(EXP_DIR, 'chemo_config'))
            setup_logging(log_name, quiet, verbose)
            for x in vials:
                exp_str = "Experiment: {0} vial {1}, {2}".format(EXP_NAME,
                                                                 x,
                                                           time.strftime("%c"))
                # make OD file
                self._create_file(x, 'OD', defaults=[exp_str])
                self._create_file(x, 'od_135_raw')
                # make temperature data file
                self._create_file(x, 'temp')
                self._create_file(x, 'temp_raw')
                # make temperature configuration file
                self._create_file(x, 'temp_config',
                                  defaults=[exp_str,
                                            "0,{0}".format(TEMP[x])])
                # make pump log file
                self._create_file(x, 'pump_log',
                                  defaults=[exp_str,
                                            "0,0"])
                # make ODset file
                self._create_file(x, 'ODset',
                                  defaults=[exp_str,
                                            "0,0"])
                # make growth rate file
                self._create_file(x, 'gr',
                                  defaults=[exp_str,
                                            "0,0"],
                                  directory='growthrate')
                # make chemostat file
                self._create_file(x, 'chemo_config',
                                  defaults=["0,0,0",
                                            "0,0,0"],
                                  directory='chemo_config')

            stir_rate = STIR
            temp_values = TEMP

            if self.experiment_params:
                stir_rate = list(map(lambda x: x['stir'], self.experiment_params['vial_configuration']))
                temp_values = list(map(lambda x: x['temp'], self.experiment_params['vial_configuration']))
            self.update_stir_rate(stir_rate)
            with open(TEMP_CAL_PATH) as f:
                temp_cal = json.load(f)
                temp_coefficients = temp_cal['coefficients']
                raw_temperatures = [str(int((temp_values[x] - temp_coefficients[x][1]) / temp_coefficients[x][0])) for x in vials]
                raw_temperatures = [0]*16
                for x in vials:
                    index = channelIdx[str(x)]["channel"]
                    raw_temperatures[index] = str(int((temp_values[x] - temp_coefficients[x][1]) / temp_coefficients[x][0]))
                self.update_temperature(raw_temperatures)

            if always_yes:
                exp_blank = 'y'
            else:
                exp_blank = 'n' #input('Calibrate vials to blank? (y/n): ')
            if exp_blank == 'y':
                # will do it with first broadcast
                self.use_blank = True
                logger.info('will use initial OD measurement as blank')
            else:
                self.use_blank = False
                self.OD_initial = np.zeros(len(vials))
        else:
            # load existing experiment
            pickle_name =  "{0}.pickle".format(EXP_NAME)
            pickle_path = os.path.join(EXP_DIR, pickle_name)
            logger.info('loading previous experiment data: %s' % pickle_path)
            with open(pickle_path, 'rb') as f:
                loaded_var  = pickle.load(f)
            x = loaded_var
            start_time = x[0]
            self.OD_initial = x[1]

        # copy current custom script to txt file
        backup_filename = '{0}_{1}.txt'.format(EXP_NAME,
                                            time.strftime('%y%m%d_%H%M'))
        shutil.copy(os.path.join(SAVE_PATH, 'custom_script.py'), os.path.join(EXP_DIR,
                                                    backup_filename))
        logger.info('saved a copy of current custom_script.py as %s' %
                    backup_filename)

        return start_time

    def check_for_calibrations(self):
        '''
        Check whether calibrations are available
        Returns a boolean.
        '''
        result = True
        if not os.path.exists(OD_CAL_PATH) or not os.path.exists(TEMP_CAL_PATH) or not os.path.exists(PUMP_CAL_PATH):
            # log and request again
            logger.warning('Calibrations not received yet, requesting again')
            self.request_calibrations()
            result = False
        return result

    def save_data(self, data: list, elapsed_time: float, vials: list, parameter: str):
        '''
        Save a variable into text file, each smart sleeve has its own file!
        '''
        if len(data) == 0:
            return
        for x in vials:
            file_name =  "vial{0}_{1}.txt".format(x, parameter)
            file_path = os.path.join(EXP_DIR, parameter, file_name)
            text_file = open(file_path, "a+")
            text_file.write("{0},{1}\n".format(elapsed_time, data[x]))
            text_file.close()

    def save_variables(self, start_time, OD_initial):
        # save variables needed for restarting experiment later
        save_path = os.path.dirname(os.path.realpath(__file__))
        pickle_name = "{0}.pickle".format(EXP_NAME)
        pickle_path = os.path.join(EXP_DIR, pickle_name)
        logger.debug('saving all variables: %s' % pickle_path)
        with open(pickle_path, 'wb') as f:
            pickle.dump([start_time, OD_initial], f)


    def get_flow_rate(self) -> dict:
        '''
        Get flow rate (pump calibration!)
        '''
        pump_cal = None
        with open(PUMP_CAL_PATH) as f:
            pump_cal = json.load(f)
        return pump_cal['coefficients']


    """
    def calc_growth_rate(self, vial, gr_start, elapsed_time):
        '''
        Calculates growth rate in order to estimate TURBIDOSTAT flux!
        Note: stats.linregress (lib scipy) IS NOT AVAILABLE IN BBB. It is needed to adapt to a numpy function!!!!
        '''
        ODfile_name =  "vial{0}_OD.txt".format(vial)
        # Grab Data and make setpoint
        OD_path = os.path.join(EXP_DIR, 'OD', ODfile_name)
        OD_data = np.genfromtxt(OD_path, delimiter=',')
        raw_time = OD_data[:, 0]
        raw_OD = OD_data[:, 1]
        raw_time = raw_time[np.isfinite(raw_OD)]
        raw_OD = raw_OD[np.isfinite(raw_OD)]

        # Trim points prior to gr_start
        trim_time = raw_time[np.nonzero(np.where(raw_time > gr_start, 1, 0))]
        trim_OD = raw_OD[np.nonzero(np.where(raw_time > gr_start, 1, 0))]

        # Take natural log, calculate slope
        log_OD = np.log(trim_OD)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            trim_time[np.isfinite(log_OD)],
            log_OD[np.isfinite(log_OD)])
        logger.debug('growth rate for vial %s: %.2f' % (vial, slope))

        # Save slope to file
        file_name =  "vial{0}_gr.txt".format(vial)
        gr_path = os.path.join(EXP_DIR, 'growthrate', file_name)
        text_file = open(gr_path, "a+")
        text_file.write("{0},{1}\n".format(elapsed_time, slope))
        text_file.close()
    """


    def tail_to_np(self, path, window=10, BUFFER_SIZE=512):
        """
        Reads file from the end and returns a numpy array with the data of the last 'window' lines.
        Alternative to np.genfromtxt(path) by loading only the needed lines instead of the whole file.
        """
        f = open(path, 'rb')
        if window == 0:
            return []

        f.seek(0, os.SEEK_END)
        remaining_bytes = f.tell()
        size = window + 1  # Read one more line to avoid broken lines
        block = -1
        data = []

        while size > 0 and remaining_bytes > 0:
            if remaining_bytes - BUFFER_SIZE > 0:
                # Seek back one whole BUFFER_SIZE
                f.seek(block * BUFFER_SIZE, os.SEEK_END)
                # read BUFFER
                bunch = f.read(BUFFER_SIZE)
            else:
                # file too small, start from beginning
                f.seek(0, 0)
                # only read what was not read
                bunch = f.read(remaining_bytes)

            bunch = bunch.decode('utf-8')
            data.append(bunch)
            size -= bunch.count('\n')
            remaining_bytes -= BUFFER_SIZE
            block -= 1

        data = ''.join(reversed(data)).splitlines()[-window:]

        if len(data) < window:
            # Not enough data
            return np.asarray([])

        for c, v in enumerate(data):
            data[c] = v.split(',')

        try:
            data = np.asarray(data, dtype=np.float64)
            return data
        except ValueError:
            # It is reading the header
            return np.asarray([])

    def custom_functions(self, data: dict, vials: list, elapsed_time: float):
        '''
        Load user script from custom_script.py
        Run scripts corresponding to requested operation using new received data
        '''
        mode = self.experiment_params['function'] if self.experiment_params else OPERATION_MODE

        if mode == 'turbidostat':
            custom_script.turbidostat(self, data, vials, elapsed_time)
        elif mode == 'chemostat':
            custom_script.chemostat(self, data, vials, elapsed_time)
        elif mode == 'growthcurve':
            custom_script.growth_curve(self, data, vials, elapsed_time)
        else:
            # try to load the user function
            # if failing report to user
            logger.info('user-defined operation mode %s' % mode)
            try:
                func = getattr(custom_script, mode)
                func(self, data, vials, elapsed_time)
            except AttributeError:
                logger.error('could not find function %s in custom_script.py' %
                            mode)
                print('Could not find function %s in custom_script.py '
                    '- Skipping user defined functions'%
                    mode)

    def stop_exp(self):
        '''
        Stop an experiment means stopping all pumps :D
        '''
        self.stop_all_pumps()



def setup_logging(filename, quiet, verbose):
    if quiet:
        logging.basicConfig(level=logging.CRITICAL + 10)
    else:
        if verbose == 0:
            level = logging.INFO
        elif verbose >= 1:
            level = logging.DEBUG
        logging.basicConfig(format='%(asctime)s - %(name)s - [%(levelname)s] '
                            '- %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename=filename,
                            level=level)

def get_options():
    description = 'Run an eVOLVER experiment from the command line'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-y', '--always-yes', action='store_true',
                        default=False,
                        help='Answer yes to all questions '
                             '(i.e. continues from existing experiment, '
                             'overwrites existing data and blanks OD '
                             'measurements)')
    parser.add_argument('-l', '--log-name',
                        default=os.path.join(EXP_DIR, 'evolver.log'),
                        help='Log file name directory (default: %(default)s)')

    log_nolog = parser.add_mutually_exclusive_group()
    log_nolog.add_argument('-v', '--verbose', action='count',
                           default=0,
                           help='Increase logging verbosity level to DEBUG '
                                '(default: INFO)')
    log_nolog.add_argument('-q', '--quiet', action='store_true',
                           default=False,
                           help='Disable logging to file entirely')
    return parser.parse_args(), parser




if __name__ == '__main__':

    # Connects to local Redis database, which will be used to communicate with SocketIO application (graphical user interface)
    redis_client = redis.StrictRedis("127.0.0.1")

    options, parser = get_options()


    # Get params from JSON file
    experiment_params = None
    if os.path.exists(JSON_PARAMS_FILE):
        with open(JSON_PARAMS_FILE) as f:
            experiment_params = json.load(f)
        

        

    # Creates eVOLVER object
    EVOLVER_NS = EvolverDPU()

    # Start by stopping any existing experiment
    EVOLVER_NS.stop_all_pumps()

    # Config and start an experiment !
    EVOLVER_NS.start_time = EVOLVER_NS.initialize_exp(VIALS,
                                                        experiment_params,
                                                        options.log_name,
                                                        options.quiet,
                                                        options.verbose,
                                                        options.always_yes
                                                        )

    # Is experiment paused ?
    paused = False

    # Creates a broadcast thread, which will receive new data from hardware from evolver-server
    bServer = Thread(target=broadcast)
    bServer.start()

   

     

    while(True):
        try:
            while(True):
                # wait until there is a command in the queue (Redis variable)
                # command = {"payload": bytes, "reply": boolean}
                command = redis_client.brpop("socketio")
                    
                command = json.loads(command[1].decode('UTF-8', errors='ignore').lower())
                print(command)

                if command["command"] == "command":
                    lock.acquire()     
                    EVOLVER_NS.s.send(functions['command']['id'].to_bytes(1,'big') + bytes(json.dumps(command["payload"]), 'utf-8') + b'\r\n')
                    lock.release()
                elif command["command"] == "getactivecal":
                    activelcal = EVOLVER_NS.request_calibrations()
                    redis_client.lpush("socketio_ans", json.dumps(activelcal))
                    
                elif command["command"] == "setrawcalibration":
                    ans = EVOLVER_NS.setrawcalibration(command["payload"])
                    redis_client.lpush("socketio_ans", ans)

                time.sleep(1)


        except KeyboardInterrupt:
            try:
                print('Ctrl-C detected, pausing experiment')
                logger.warning('interrupt received, pausing experiment')
                EVOLVER_NS.stop_exp()
                # stop receiving broadcasts
                EVOLVER_NS.disconnect()
                while True:
                    key = input('Experiment paused. Press enter key to restart '
                                ' or hit Ctrl-C again to terminate experiment')
                    logger.warning('resuming experiment')
                    # no need to have something like "restart_chemo" here
                    # with the new server logic
                    EVOLVER_NS.connect()
                    break
            except KeyboardInterrupt:
                print('Second Ctrl-C detected, shutting down')
                logger.warning('second interrupt received, terminating '
                                'experiment')
                EVOLVER_NS.stop_exp()
                print('Experiment stopped, goodbye!')
                logger.warning('experiment stopped, goodbye!')
                break
        except Exception as e:
            logger.critical('exception %s stopped the experiment' % str(e))
            print('error "%s" stopped the experiment' % str(e))
            traceback.print_exc(file=sys.stdout)
            EVOLVER_NS.stop_exp()
            print('Experiment stopped, goodbye!')
            logger.warning('experiment stopped, goodbye!')
            break

    # stop experiment one last time
    # covers corner case where user presses Ctrl-C twice quickly
    EVOLVER_NS.connect()
    EVOLVER_NS.stop_exp()