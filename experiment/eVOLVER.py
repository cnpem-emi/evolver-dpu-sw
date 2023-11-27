#!/usr/bin/env python3

import argparse
import asyncio
import json
import logging
import os
import pickle
import select
import shutil
import socket
import sys
import time
import traceback
from threading import Lock, Thread
import time

import numpy as np
import redis

import custom_script
from consts import functions
from custom_script import LED, STIR, TEMP

# Should not be changed
VIALS = [x for x in range(16)]

SAVE_PATH = os.path.dirname(os.path.realpath(__file__))
EXPERIMENT_DATA_PATH = os.path.join(SAVE_PATH, "experiment_data")
# EXP_NAME = None
# OPERATION_MODE = None

OD_CAL_PATH = os.path.join(SAVE_PATH, "od_cal.json")
TEMP_CAL_PATH = os.path.join(SAVE_PATH, "temp_cal.json")
PUMP_CAL_PATH = os.path.join(SAVE_PATH, "pump_cal.json")

JSON_PARAMS_FILE = os.path.join(SAVE_PATH, "eVOLVER_parameters.json")
CHANNEL_INDEX_PATH = os.path.join(SAVE_PATH, "channel_index.json")


SIGMOID = "sigmoid"
LINEAR = "linear"
THREE_DIMENSION = "3d"

logger = logging.getLogger("eVOLVER")
paused = False


# ---- eVolver DPU entity, connects to server (hardware access) evolver-sw
global EVOLVER_NS

EVOLVER_NS = None
EVOLVER_IP = "127.0.0.1"
EVOLVER_PORT = 6001

global broadcastSocket
global broadcastReady
global lock

broadcastSocket = None
broadcastReady = False
lock = Lock()


# ----- socketio object
global sio

# ----- Redis database for eVolver unit
global redis_client
redis_client = redis.StrictRedis("127.0.0.1")

# ----- Smart sleeves indexes (hardware differs from sw index)
global channelIdx
with open(CHANNEL_INDEX_PATH) as f:
    channelIdx = json.load(f)


class EvolverDPU:
    global broadcastSocket
    global broadcastReady
    global channelIdx
    global lock

    ip_address = None
    start_time = [None] * 16

    od_cals = [None] * 16
    temp_cal = None
    pump_cal = None
    active_cals = [None] * 16

    exps_ocurring = []
    active_exp = [None] * 16
    exp_configs = [None] * 16
    

    """ Inicializando DPU """

    def __init__(self):
        self.connect()

        """###
        lock.acquire()
        self.s.send(functions["get_active_calibrations"]["id"].to_bytes(1, "big") + b"\r\n"        )
        time.sleep(0.1)

        for _ in range(3):
            ready = select.select([self.s], [], [], 2)
            if ready[0]:
                info = self.s.recv(30000)[:-2]
            else:
                time.sleep(1)
        lock.release()
        self.active_cal = json.loads(info)"""

    def connect(self):
        """
        Connect to evolver-server, to get/set hardware information
        """
        global broadcastSocket
        global broadcastReady

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((EVOLVER_IP, EVOLVER_PORT))
        self.s.setblocking(0)

        broadcastSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        broadcastSocket.connect((EVOLVER_IP, EVOLVER_PORT + 1000))
        broadcastSocket.setblocking(0)
        broadcastReady = True

        logger.info("connected to eVOLVER as client")

    def disconnect(self):
        self.s.close()
        broadcastSocket.close()

        logger.info("disconnected to eVOLVER as client")


    """ Broadcast and broadcast-related """
    def broadcast(self, data: dict):
        """
        This method should be called every time a new broadcast is received from evolver-server
        (from BBB, which communicates w/ hardware), which indicates that new values are available.

        This method converts raw data into real values and ask for new step from custom functions.
        """

        print("\nBroadcast received", data)
        if self.ip_address is None or data["ip"] != self.ip_address:
            self.ip_address = data["ip"]

        if '0' in data["config"]["od_led"]['value']:
            turn_leds_on = 1

        # Check if temp and pump cal are set
        for vial in VIALS:
            # raw_temp = ['--'] * 16
            if self.active_exp[vial] is not None:

                elapsed_time = round((time.time() - self.start_time) / 3600, 4)
                self.save_data(vial, [data["od_135"][vial], data["temp"][vial]], elapsed_time, False)
                    
                try:
                    transformed_data = self.transform_data(data["data"], data["config"]["temp"]["value"][vial], vial)
                    # raw_temp[vial] = transformed_data[2]
                    self.save_data(vial, transformed_data, elapsed_time, True)
                
                except OSError:
                    logger.info("Problem with calibration")
                    return
                    
        self.custom_functions(elapsed_time)

        broadcasting_data = {
            "ip": self.ip_address,
            "timestamp": data['timestamp'],
            "data": data['data'],
            "active_exp": self.active_exp,
            "od_cal": self.od_cals,
            "temp_cal": self.temp_cal,
            "pump_cal": self.pump_cal
        }

        # Restart logging for db/gdrive syncing
        logging.shutdown()
        logging.getLogger("eVOLVER")

        return broadcasting_data


    def transform_data(self, data, setpoint, vial):
        raw_od = float(data["od_135"][vial])
        raw_temp = float(data["temp"][vial])
        setpoint = float(setpoint)

        od_coeff = self.active_cals[vial]["od_135"]
        temp_coeff = self.active_cals[vial]["temp"]

        # Try to apply calibration to od_135
        try: 
            od_value = od_coeff[2] - (np.log10(
                                        (od_coeff[1] - od_coeff[0])/(raw_od - od_coeff[0]) - 1)
                                        / od_coeff[3]
                                    )
            
            if not np.isfinite(od_value):
                od_value = np.nan
                logger.debug("OD from vial %d: %s" % (x+1, od_value[x]))

            else:
                logger.debug("OD from vial %d: %.3f" % (x+1, od_value[x]))

        except ValueError:
                print("OD Read Error")
                logger.error("OD read error for vial %d, setting to NaN" % x)
                od_value = np.nan

        # Try to apply calibration to temperature (read)
        try:
            temp_value = (raw_temp * temp_coeff[0]) + temp_coeff[1]
            print('temperature from vial %d: %.3f' % (x, temp_value[x]))

        except ValueError:
                print("Temp Read Error")
                logger.error("temperature read error for vial %d, setting to NaN" % x)
                temp_value = np.nan

        '''exp_dir = os.path.join(EXPERIMENT_DATA_PATH, self.active_exp[vial])
        file_name = "vial{}_temp_config.txt".format(vial+1)
        file_path = os.path.join(self.exp_dir, "temp_config", file_name)

        temp_set_data = np.genfromtxt(file_path, delimiter=",")
        temp_set = temp_set_data[len(temp_set_data) - 1][1]
        ''' 
        # Try to apply calibration to temperature (setpoint)
        try:
            setpoint_value = (setpoint * temp_coeff[0]) + temp_coeff[1]
        except ValueError:
            print("Set Temp Read Error")
            setpoint = "nan"

        '''# update temperatures only if difference with expected
        # value is above 0.2 degrees celsius
        delta_t = np.abs(setpoint_value - temp_set)

        if delta_t < 0.2:
            logger.info("updating temperatures (max. deltaT is %.2f)" % delta_t)
            raw_set = str(int((temp_set - temp_coeff[1]) / temp_coeff[0]))'''

        return od_value, temp_value, setpoint_value
    

    def save_data(self, vial: int, data: list, elapsed_time: float, is_transformed: bool):
        """
        Save a variable into text file, each smart sleeve has its own file!
        """
        if is_transformed:
            parameters = ['OD', "temp", "temp_config"]
        else:
            parameters = ['od_135_raw', 'temp_raw']

        for i,param in enumerate(parameters):
            file_name = "vial{0}_{1}.txt".format(vial+1, param)
            file_path = os.path.join(EXPERIMENT_DATA_PATH, self.active_exp[vial], param, file_name)
            text_file = open(file_path, "a+")
            text_file.write("{0},{1}\n".format(elapsed_time, data[i]))
            text_file.close()


    def custom_functions(self, elapsed_time: float):
        """
        Load user script from custom_script.py
        Run scripts corresponding to requested operation using new received data
        """

        for experiment in self.exps_ocurring:
            if experiment["mode"] == "turbidostat":
                custom_script.turbidostat(self, experiment["vials"], elapsed_time)

            elif experiment["mode"] == "chemostat":
                custom_script.chemostat(self, experiment["vials"], elapsed_time)

            elif experiment["mode"] == "growthcurve":
                custom_script.growth_curve(self, experiment["vials"], elapsed_time)

            else:
                # try to load the user function
                # if failing report to user
                logger.info("user-defined operation mode %s" % experiment["mode"])

                try:
                    func = getattr(custom_script, experiment["mode"])
                    func(self, experiment["mode"], experiment["vials"], elapsed_time)

                except AttributeError:
                    logger.error("could not find function %s in custom_script.py" % experiment["mode"])
                    print("Could not find function %s in custom_script.py - Skipping user defined functions" % experiment["mode"])


    """# Check if calibration coefficients are available
            if self.active_cals
            if not self.check_calibration(""):
            print("Calibration files still missing, skipping custom functions")
            return

        with open(OD_CAL_PATH) as f:
            od_cal = json.load(f)

        with open(TEMP_CAL_PATH) as f:
            temp_cal = json.load(f)

        # Apply calibrations and update temperatures if needed
        vials = self.active_vials if self.exp_status else VIALS
        data = self.transform_data(data, VIALS, od_cal, temp_cal)

        if data is None:
            return

        # Should we "blank" the OD?
        '''
        if self.use_blank and self.OD_initial is None:
            logger.info("setting initial OD reading")
            self.OD_initial = np.array(data["transformed"]["od"])

        elif self.OD_initial is None:
            self.OD_initial = np.zeros(len(VIALS))

        data["transformed"]["od"] = list(np.array(data["transformed"]["od"]) - self.OD_initial)
        '''
        print("OD: ", data["transformed"]["od"])
        print("Temp: ", data["transformed"]["temp"])
        print()

        if data is None:
            logger.error("could not tranform raw data, skipping user-defined functions")
            return

        if self.exp_status is False:
            return

        elapsed_time = round((time.time() - self.start_time) / 3600, 4)

        # Save data into .csv files
        try:
            # Transformed values, true values
            self.save_data(data["transformed"]["od"], elapsed_time, VIALS, "OD")
            self.save_data(data["transformed"]["temp"], elapsed_time, VIALS, "temp")

            # Raw data
            raw_data = [0] * 16

            for param in od_cal["params"]:
                raw_data = data["data"].get(param, [])
                # Order raw data before saving values
                # for ss in range(16):
                #     raw_data[ss] = data['data'].get(param, [])[channelIdx[str(ss)]["channel"]]

                self.save_data(raw_data, elapsed_time, VIALS, param + "_raw")

            for param in temp_cal["params"]:
                raw_data = data["data"].get(param, [])
                # Order raw data before saving values
                # for ss in range(16):
                #    raw_data[ss] = data['data'].get(param, [])[channelIdx[str(ss)]["channel"]]

                self.save_data(raw_data, elapsed_time, VIALS, param + "_raw")

        except OSError:
            logger.info(
                "Broadcast received before experiment initialization - skipping custom function..."
            )
            return

        # Run custom functions
        self.custom_functions(data, self.active_vials, elapsed_time)

        # save variables
        self.save_variables(self.start_time, self.OD_initial)

        # Restart logging for db/gdrive syncing
        logging.shutdown()
        logging.getLogger("eVOLVER")"""
    #def set_temp_cal()
    #def set_pump_cal()
    #def get_cal_coefficients(vial, experiment_params["od_cal"]):

    
    def check_calibration(self, name: str):
        """
        Check whether calibrations are available
        Returns a boolean.
        """
        result = True

        # Retrieving all calibration names
        lock.acquire()
        self.s.send(functions["get_calibration_names"]["id"].to_bytes(1, "big") + b"\r\n")
        time.sleep(0.1)

        for _ in range(3):
            ready = select.select([self.s], [], [], 2)
            if ready[0]:
                info = self.s.recv(30000)[:-2]
            else:
                time.sleep(1)
        lock.release()

        cal_names = json.loads(info)
        print(cal_names)

        if name not in cal_names:
            result = False

        '''self.active_cal = json.loads(info)

        if (
            not os.path.exists(OD_CAL_PATH)
            or not os.path.exists(TEMP_CAL_PATH)
            or not os.path.exists(PUMP_CAL_PATH)
        ):
            # log and request again
            logger.warning("Calibrations not received yet, requesting again")
            self.request_calibrations()
            result = False'''

        return result

    

    

    

    def save_variables(self, start_time, OD_initial):
        # save variables needed for restarting experiment later
        pickle_name = "{0}.pickle".format(self.exp_name)
        pickle_path = os.path.join(self.exp_dir, pickle_name)
        logger.debug("saving all variables: %s" % pickle_path)

        with open(pickle_path, "wb") as f:
            pickle.dump([start_time, OD_initial], f)

    """ Experiment Related"""

    def config_exp(self, experiment_params, quiet, verbose, always_yes=False):
        logger.info("initializing config\n")
        print("initializing config")
        
        if experiment_params == None:
            logger.info("no configuration sent for experiment, fail to initialize")
            return False

        exp_dir = os.path.join(EXPERIMENT_DATA_PATH, experiment_params["name"])
        if os.path.exists(exp_dir):
            logger.info("change experiment name")
            return False
  
        vials = [conf['vial'] for conf in experiment_params["vial_configuration"]]

        for vial in vials:
            if self.active_exp[vial] != experiment_params["name"]:
                logger.info("vial selected is already in use, stop experiment first")
                return False

        if self.temp_cal is None or self.pump_cal is None: # or (None in self.od_cals):
            logger.info("no calibration set")
            return False
        
        if not self.check_calibration(experiment_params["od_cal"]):
            logger.info("od_cal doesnt exist")
            return False
                
        for vial in vials:
            self.od_cals[vial] = experiment_params["od_cal"]
            self.active_cals[vial] = {
                "od": self.get_cal_coefficients(vial, experiment_params["od_cal"]),
                "temp": self.get_cal_coefficients(vial, self.temp_cal),
                "pump": self.get_cal_coefficients(vial, self.pump_cal)
            }

        exp_dict = {
            "name": experiment_params["name"],
            "mode": experiment_params["funtion"],
            "vials": [conf['vial'] for conf in experiment_params["vial_configuration"]]
        }

        if exp_dict in self.exps_ocurring:
            logger.info("experiment already exists and is running")
            return False
        
        self.exps_ocurring += [exp_dict]

        for i,vial in enumerate(vials):
            self.exp_configs[vial] = {
                "name": experiment_params["name"],
                "mode": experiment_params["function"],
                "config": experiment_params["vial_configuration"][i]
            }

        logger.debug("creating data directories")
        os.makedirs(exp_dir)

        path_config = os.path.join(exp_dir, "exp_config.json")
        experiment_params["temp_cal"] = self.temp_cal
        experiment_params["pump_cal"] = self.pump_cal

        with open(path_config, "w") as file:
            json.dump((experiment_params), file, indent = 4)

        os.makedirs(os.path.join(exp_dir, "OD"))
        os.makedirs(os.path.join(exp_dir, "od_135_raw"))
        os.makedirs(os.path.join(exp_dir, "ODset"))
        os.makedirs(os.path.join(exp_dir, "growthrate"))

        os.makedirs(os.path.join(exp_dir, "temp"))
        os.makedirs(os.path.join(exp_dir, "temp_raw"))
        os.makedirs(os.path.join(exp_dir, "temp_config"))

        os.makedirs(os.path.join(exp_dir, "pump_out_log"))
        os.makedirs(os.path.join(exp_dir, "pump_in_log"))
        os.makedirs(os.path.join(exp_dir, "chemo_config"))
        setup_logging(os.path.join(exp_dir, "evolver.log"), quiet, verbose)

        for i,vial in enumerate(vials):
            exp_str = "Experiment: {0} vial {1}, {2}".format(
                    experiment_params["name"], vial+1, time.strftime("%c")
            )

            # make OD file
            self._create_file(vial+1, "OD", defaults=[exp_str])
            self._create_file(vial+1, "od_135_raw")

            # make temperature data file
            self._create_file(vial+1, "temp")
            self._create_file(vial+1, "temp_raw")

            # make temperature configuration file
            self._create_file(vial+1, "temp_config", defaults=[exp_str, "0,{0}".format(experiment_params["vial_configuration"][i]["temp"])])

            # make pump log file
            self._create_file(vial+1, "pump_in_log", defaults=[exp_str, "0,0"])
            self._create_file(vial+1, "pump_out_log", defaults=[exp_str, "0,0"])

            # make ODset file
            self._create_file(vial+1, "ODset", defaults=[exp_str, "0,0"])

            # make growth rate file
            self._create_file(vial+1, "gr", defaults=[exp_str, "0,0"], directory="growthrate")

            # make chemostat file
            self._create_file(
                vial+1,
                "chemo_config",
                defaults=["0,0,0", "0,0,0"],
                directory="chemo_config",
            )

        return True


    def initialize_exp(self, exp_name):
        logger.info("initializing experiment")
        print("initializing experiment")

        if not os.path.exists(os.path.join(EXPERIMENT_DATA_PATH, exp_name)):
            return False

        if exp_name not in [exp["name"] for exp in self.exps_ocurring]:
            return False

        start_time = time.time()
        start_stir = ['--'] * 16
        start_temp = ['--'] * 16

        for i,vial_conf in enumerate(self.exp_configs):
            if vial_conf["name"] == exp_name:
                start_stir[i] = vial_conf["config"]["stir"]
                self.start_time[i] = start_time

                coeff = self.active_cals[i]["temp"]
                start_temp[i] = str(int((float(vial_conf["config"]["temp"]) - coeff[1]) / coeff[0]))       

        self.update_temperature(start_temp)
        self.update_stir_rate(start_stir)

        '''exp_blank = "y" if always_yes else "n"

        if exp_blank == "y":  # will do it with first broadcast
            self.use_blank = True
            logger.info("will use initial OD measurement as blank")
        else:
            self.use_blank = False
            self.OD_initial = np.zeros(len(self.active_vials))'''

        '''# copy current custom script to txt file
        backup_filename = "{0}_{1}.txt".format(
            self.exp_name, time.strftime("%y%m%d_%H%M")
        )
        shutil.copy(
            os.path.join(SAVE_PATH, "custom_script.py"),
            os.path.join(self.exp_dir, backup_filename),
        )
        logger.info("saved a copy of current custom_script.py as %s" % backup_filename)'''

        for i,vial_conf in enumerate(self.exp_configs):
            if vial_conf["name"] == exp_name:
                self.active_exp[i] = exp_name
                
        print("Started!")
        return start_time

    def _create_file(self, vial, param, directory=None, defaults=None):
        """
        Create file for data saving, if needed!
        """
        if defaults is None:
            defaults = []

        if directory is None:
            directory = param

        file_name = "vial{0}_{1}.txt".format(vial, param)
        file_path = os.path.join(self.exp_dir, directory, file_name)
        text_file = open(file_path, "w")

        for default in defaults:
            text_file.write(default + "\n")
        text_file.close()

    def tail_to_np(self, path, window=10, BUFFER_SIZE=512):
        """
        Reads file from the end and returns a numpy array with the data of the last 'window' lines.
        Alternative to np.genfromtxt(path) by loading only the needed lines instead of the whole file.
        """
        f = open(path, "rb")
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

            bunch = bunch.decode("utf-8")
            data.append(bunch)
            size -= bunch.count("\n")
            remaining_bytes -= BUFFER_SIZE
            block -= 1

        data = "".join(reversed(data)).splitlines()[-window:]

        if len(data) < window:
            # Not enough data
            return np.asarray([])

        for c, v in enumerate(data):
            data[c] = v.split(",")

        try:
            data = np.asarray(data, dtype=np.float64)
            return data

        except ValueError:
            # It is reading the header
            return np.asarray([])

    def update_chemo(
        self,
        data: dict,
        vials: list,
        bolus_in_s: list,
        period_config: list,
        immediate=True,
    ):
        """
        Update chemostato operations.
        Usually asked to be run in custom_script.
        Currently, only changes pumps !
        """
        current_pump = data["config"]["pump"]["value"]

        MESSAGE = {
            "fields_expected_incoming": 49,
            "fields_expected_outgoing": 49,
            "recurring": True,
            "immediate": immediate,
            "value": ["--"] * 48,
            "param": "pump",
        }

        for x in vials:
            pumpA_idx = x  # channelIdx[str(x)]["A"]
            pumpB_idx = x + 16  # channelIdx[str(x)]["B"]
            pumpC_idx = x + 32  # channelIdx[str(x)]["C"]

            # stop pumps if period is zero
            if period_config[x] == 0:
                # influx
                MESSAGE["value"][pumpA_idx] = "0|0"
                MESSAGE["value"][pumpB_idx] = "0|0"
                # efflux
                MESSAGE["value"][pumpC_idx] = "0|0"

            else:
                # influx 1
                MESSAGE["value"][pumpA_idx] = "%.2f|%.1f" % (
                    bolus_in_s[x],
                    period_config[x],
                )
                # influx 2
                MESSAGE["value"][pumpB_idx] = "%.2f|%.1f" % (
                    bolus_in_s[x],
                    period_config[x],
                )
                # efflux
                MESSAGE["value"][pumpC_idx] = "%.2f|%.1f" % (
                    bolus_in_s[x] * 3,
                    period_config[x],
                )

        if True:  # MESSAGE['value'] != current_pump:
            lock.acquire()
            self.s.send(
                functions["command"]["id"].to_bytes(1, "big")
                + bytes(json.dumps(MESSAGE), "utf-8")
                + b"\r\n"
            )
            lock.release()

    def get_flow_rate(self) -> dict:
        """
        Get flow rate (pump calibration!)
        """
        raw_pump_cal = None
        pump_cal = None

        with open(PUMP_CAL_PATH) as f:
            raw_pump_cal = json.load(f)

        # for ss in range(16):
        #    pump_cal[ss] = raw_pump_cal.get("coefficients", [])[channelIdx[str(ss)]["channel"]]
        # return pump_cal

        return raw_pump_cal["coefficients"]

    def calc_growth_rate(self, vial, gr_start, elapsed_time):
        """
        Calculates growth rate in order to estimate TURBIDOSTAT flux!
        """
        ODfile_name = "vial{0}_OD.txt".format(vial)

        # Grab Data and make setpoint
        OD_path = os.path.join(self.exp_dir, "OD", ODfile_name)
        OD_data = np.genfromtxt(OD_path, delimiter=",")

        raw_time = OD_data[:, 0]
        raw_OD = OD_data[:, 1]

        raw_time = raw_time[np.isfinite(raw_OD)]
        raw_OD = raw_OD[np.isfinite(raw_OD)]

        # Trim points prior to gr_start
        trim_time = raw_time[np.nonzero(np.where(raw_time > gr_start, 1, 0))]
        trim_OD = raw_OD[np.nonzero(np.where(raw_time > gr_start, 1, 0))]

        # Take natural log, calculate slope
        log_OD = np.log(trim_OD)
        trim_time = trim_time[np.isfinite(log_OD)]

        A = np.vstack([trim_time, np.ones(len(trim_time))]).T
        slope, intercept = np.linalg.lstsq(A, log_OD[np.isfinite(log_OD)], rcond=None)[
            0
        ]
        logger.debug("growth rate for vial %s: %.2f" % (vial, slope))

        # Save slope to file
        file_name = "vial{0}_gr.txt".format(vial)
        gr_path = os.path.join(self.exp_dir, "growthrate", file_name)
        text_file = open(gr_path, "a+")
        text_file.write("{0},{1}\n".format(elapsed_time, slope))
        text_file.close()

    """ Commands """

    def fluid_command(self, MESSAGE: list):
        """
        Update fluids values
        """
        # Change to correct channel
        logger.debug("fluid command: %s" % MESSAGE)
        data = {
            "param": "pump",
            "value": MESSAGE,
            "recurring": False,
            "immediate": True,
        }

        lock.acquire()
        self.s.send(
            functions["command"]["id"].to_bytes(1, "big")
            + bytes(json.dumps(data), "utf-8")
            + b"\r\n"
        )
        lock.release()

    def update_stir_rate(self, stir_rates: list, immediate=True):
        """
        Update stir values
        """
        # Change to correct channel

        data = {
            "param": "stir",
            "value": stir_rates,
            "immediate": immediate,
            "recurring": True,
        }
        logger.debug("stir rate command: %s" % data)

        lock.acquire()
        self.s.send(
            functions["command"]["id"].to_bytes(1, "big")
            + bytes(json.dumps(data), "utf-8")
            + b"\r\n"
        )
        lock.release()

    def update_temperature(self, temperatures: list, immediate=True):
        """
        Update temperature values. Values in list should be raw values 0-4095
        """

        data = {
            "param": "temp",
            "value": temperatures,
            "immediate": immediate,
            "recurring": True,
        }
        logger.debug("temperature command: %s" % data)

        lock.acquire()
        self.s.send(
            functions["command"]["id"].to_bytes(1, "big")
            + bytes(json.dumps(data), "utf-8")
            + b"\r\n"
        )
        lock.release()

    def update_led(self, leds: list, immediate=True):
        """
        Update Od LED values. Values in list should be in raw values 0-4095
        """
        data = {
            "param": "od_led",
            "value": leds,
            "immediate": immediate,
            "recurring": True,
        }
        logger.debug("OD LED command: %s" % data)

        lock.acquire()
        self.s.send(
            functions["command"]["id"].to_bytes(1, "big")
            + bytes(json.dumps(data), "utf-8")
            + b"\r\n"
        )
        lock.release()

    """ Calibration-related """

    def activecalibrations(self, data: dict):
        print("Calibrations received")
        for calibration in data:
            if calibration["calibrationType"] == "od":
                file_path = OD_CAL_PATH
            elif calibration["calibrationType"] == "temperature":
                file_path = TEMP_CAL_PATH
            elif calibration["calibrationType"] == "pump":
                file_path = PUMP_CAL_PATH
            else:
                continue
            for fit in calibration["fits"]:
                if fit["active"]:
                    with open(file_path, "w") as f:
                        json.dump(fit, f)
                    # Create raw data directories and files for params needed
                    for param in fit["params"]:
                        if (
                            not os.path.isdir(
                                os.path.join(self.exp_dir, param + "_raw")
                            )
                            and param != "pump"
                        ):
                            os.makedirs(os.path.join(self.exp_dir, param + "_raw"))
                            for x in range(len(fit["coefficients"])):
                                exp_str = "Experiment: {0} vial {1}, {2}".format(
                                    self.exp_name, x+1, time.strftime("%c")
                                )
                                self._create_file(x+1, param + "_raw", defaults=[exp_str])
                    break

    # ----- [BEGGINING] Custom functions -----

    def get_all_calibrations(self) -> list:
        """
        Get all calibrations.
        """
        lock.acquire()
        self.s.send(functions["get_all_calibrations"]["id"].to_bytes(1, "big") + b"\r\n")
        time.sleep(0.1)

        for _ in range(3):
            ready = select.select([self.s], [], [], 2)
            if ready[0]:
                server_response = json.loads(self.s.recv(3000000)[:-2])
                lock.release()
                logger.debug("getallcalibrations: %s" % server_response)
                return server_response
            time.sleep(1)

        return []

    def get_update_interval(self) -> str:
        """
        Get update interval.
        """
        lock.acquire()
        self.s.send(functions["getupdateinterval"]["id"].to_bytes(1, "big") + b"\r\n")
        time.sleep(0.1)

        for _ in range(3):
            ready = select.select([self.s], [], [], 2)
            if ready[0]:
                info = json.loads(self.s.recv(30000)[:-2])
                lock.release()
                return info
            time.sleep(0.1)

        return ""

    def setodcalibration(self, data: dict) -> dict:
        """
        Get all calibration from server and write to od_cal.json
        the calibration with name data["name"].
        """
        all_calibrations = self.get_all_calibrations()
        print("\n\n")
        print(len(all_calibrations))
        for calibration in all_calibrations:
            print(calibration)
            print("\n\n")
            if calibration["name"] == data["name"]:
                with open(OD_CAL_PATH, "w") as f:
                    json.dump(calibration["fits"][0], f)
                response = {"message": "temp calibration set"}
                return response

        response = {"message": "temp calibration not set"}
        return response

    def settempcalibration(self, data: dict) -> str:
        """
        Get all calibration from server and write to temp_cal.json
        the calibration with name data["name"].
        """
        all_calibrations = self.get_all_calibrations()
        for calibration in all_calibrations:
            if calibration["name"] == data["name"]:
                with open(TEMP_CAL_PATH, "w") as f:
                    json.dump(calibration, f)
                response = {"message": "calibration set"}
                return response

    def appendcal(self, data: dict):
        """
        Append calibration.
        """
        logger.debug("appendcal")
        lock.acquire()
        self.s.send(
            functions["append_calibration"]["id"].to_bytes(1, "big")
            + bytes(json.dumps(data), "utf-8")
            + b"\r\n"
        )
        time.sleep(0.1)
        for _ in range(3):
            ready = select.select([self.s], [], [], 2)
            if ready[0]:
                info = self.s.recv(30000)[:-2]
                lock.release()
                return info
            else:
                time.sleep(1)
        lock.release()

        return None

    def getcalibrationnames(self):
        logger.debug("getcalibrationnames")

        lock.acquire()
        self.s.send(functions["get_calibration_names"]["id"].to_bytes(1, "big") + b"\r\n")
        time.sleep(0.5)

        for _ in range(3):
            print("Wainting calibration names from server")
            ready = select.select([self.s], [], [], 2)

            if ready[0]:
                msg = self.s.recv(30000)[:-2]
                print("Calibration names from server:", msg)
                info = json.loads(msg)
                lock.release()
                return info
            else:
                time.sleep(0.5)
        lock.release()
        print("failed to get calibration names from server")
        return info

    def getfitnames(self) -> bytes | None:
        """
        Get fit names.
        """
        print
        lock.acquire()
        self.s.send(functions["getfitnames"]["id"].to_bytes(1, "big") + b"\r\n")
        time.sleep(0.1)

        for _ in range(3):
            ready = select.select([self.s], [], [], 2)
            if ready[0]:
                info = json.loads(self.s.recv(30000)[:-2])
                lock.release()
                return info
            time.sleep(1)

        return None

    def getcalibration(self, data: dict) -> bytes | None:
        """
        Get calibration.
        """
        lock.acquire()
        self.s.send(
            functions["getcalibration"]["id"].to_bytes(1, "big")
            + bytes(json.dumps(data), "utf-8")
            + b"\r\n"
        )
        time.sleep(0.5)

        for _ in range(3):
            ready = select.select([self.s], [], [], 2)
            if ready[0]:
                response = self.s.recv(30000)[:-2]
                print("Response: ", response)
                info = json.loads(response)
                lock.release()
                return info
            time.sleep(1)

        return None

    def setfitcalibrations(self, data: dict) -> None:
        logger.debug("setfitcalibrations")
        lock.acquire()
        self.s.send(
            functions["setfitcalibrations"]["id"].to_bytes(1, "big")
            + bytes(json.dumps(data), "utf-8")
            + b"\r\n"
        )
        time.sleep(0.1)
        lock.release()

        return None

    def setactiveodcal(self, data: dict) -> None:
        logger.debug("setactiveodcal")
        lock.acquire()
        self.s.send(
            functions["setactiveodcal"]["id"].to_bytes(1, "big")
            + bytes(json.dumps(data), "utf-8")
            + b"\r\n"
        )
        time.sleep(1)
        lock.release()

        return None

    def get_device_name(self) -> bytes | None:
        """
        Get device name.
        """
        lock.acquire()
        # Check if to_bytes could be replaced by encode
        data = functions["getdevicename"]["id"].to_bytes(1, "big") + b"\r\n"
        self.s.send(data)
        time.sleep(0.1)

        for _ in range(3):
            ready = select.select([self.s], [], [], 2)
            print(ready)
            if ready[0]:
                # info = self.s.recv(30000)[:-2]
                info = self.s.recv(30000)
                lock.release()
                print(info)
                return info
            time.sleep(1)

        return None

    def activecalibrations(self, data: dict):
        print("Calibrations received")
        for calibration in data:
            if calibration["calibrationtype"] == "od":
                file_path = OD_CAL_PATH
            elif calibration["calibrationtype"] == "temperature":
                file_path = TEMP_CAL_PATH
            elif calibration["calibrationtype"] == "pump":
                file_path = PUMP_CAL_PATH
            else:
                continue
            for fit in calibration["fits"]:
                if fit["active"]:
                    with open(file_path, "w") as f:
                        json.dump(fit, f)
                    # Create raw data directories and files for params needed
                    for param in fit["params"]:
                        if (
                            not os.path.isdir(
                                os.path.join(self.exp_dir, param + "_raw")
                            )
                            and param != "pump"
                        ):
                            os.makedirs(os.path.join(self.exp_dir, param + "_raw"))
                            for x in range(len(fit["coefficients"])):
                                exp_str = "Experiment: {0} vial {1}, {2}".format(
                                    self.exp_name, x+1, time.strftime("%c")
                                )
                                self._create_file(x+1, param + "_raw", defaults=[exp_str])
                    break

    def request_calibrations(self) -> dict:
        """
        Request calibrations to evolver-server.
        """
        logger.debug("requesting active calibrations")

        lock.acquire()
        self.s.send(functions["get_active_calibrations"]["id"].to_bytes(1, "big") + b"\r\n")
        time.sleep(1)

        for _ in range(3):
            ready = select.select([self.s], [], [], 2)
            if ready[0]:
                info = json.loads(self.s.recv(30000)[:-2])
                lock.release()
                return info
            else:
                time.sleep(1)

        lock.release()
        return None

    def setrawcalibration(self, data):
        """
        Set raw calibration.
        """
        logger.debug("setrawcalibration")
        lock.acquire()
        self.s.send(
            functions["setrawcalibration"]["id"].to_bytes(1, "big")
            + bytes(json.dumps(data), "utf-8")
            + b"\r\n"
        )
        time.sleep(1)

        for _ in range(3):
            ready = select.select([self.s], [], [], 2)

            if ready[0]:
                info = self.s.recv(30000)[:-2]
                break

            time.sleep(.1)

        lock.release()

        return info

    # ----- [END] Custom functions -----

    """def get_last_commands(self):  # x -> dict
        logger.debug("getlastcommands")
        lock.acquire()
        self.s.send(functions["getlastcommands"]["id"].to_bytes(1, "big") + b"\r\n")
        time.sleep(1)

        for _ in range(3):
            ready = select.select([self.s], [], [], 2)

            if ready[0]:
                info = self.s.recv(30000)[:-2]
                lock.release()
                return info
            else:
                time.sleep(1)

        # print(info)"""

    def get_num_commands(self):  # x -> int
        logger.debug("get_num_commands")
        lock.acquire()
        self.s.send(functions["get_num_commands"]["id"].to_bytes(1, "big") + b"\r\n")
        time.sleep(1)

        for _ in range(3):
            ready = select.select([self.s], [], [], 2)

            if ready[0]:
                value = json.loads(self.s.recv(30000)[:-2])
                lock.release()
                # string_value = value.decode('utf-8')
                return value
            time.sleep(1)

        # print(string_value)
        return None

    '''def get_device_name(self):  # x -> dict
        logger.debug("getdevicename")
        lock.acquire()
        self.s.send(functions["get_device_name"]["id"].to_bytes(1, "big") + b"\r\n")
        time.sleep(1)

        for _ in range(3):
            ready = select.select([self.s], [], [], 2)

            if ready[0]:
                info = self.s.recv(30000)[:-2]
                break
            else:
                time.sleep(1)
        lock.release()
        print(info)
        return info'''

    """ Ending experiment """

    def stop_all_pumps(self):
        """
        Stop all pumps
        """
        logger.info("stopping all pumps")
        data = {
            "param": "pump",
            "value": ["0"] * 48,
            "recurring": False,
            "immediate": True,
        }

        lock.acquire()
        self.s.send(
            functions["command"]["id"].to_bytes(1, "big")
            + bytes(json.dumps(data), "utf-8")
            + b"\r\n"
        )
        lock.release()

    def stop_some_vials(self, vials: list):
        pump = ["--" for i in range(48)]
        temp = ['nan' for i in range(16)]
        stir = ['nan' for i in range(16)]

        for vial in vials:
            pump[vial] = 0
            pump[vial + 16] = 0
            pump[vial + 32] = 0
            stir[vial] = 0
            temp[vial] = 4095

        self.update_temperature(temp)
        self.update_stir_rate(stir)
        self.fluid_command(pump)

    def stop_exp(self):
        """
        Stop an experiment means stopping all pumps :D
        temperature set to "zero" and stir stopped
        """
        print("Stopping experiment")
        self.stop_all_pumps()
        self.update_temperature([4095] * 16)
        self.update_stir_rate([0] * 16)

        self.exp_status = False


# Auxiliary functions
def broadcast():
    """
    When receive new data from evolver-server:
      - Update Redis key
      - Call broadcast method for eVOLVER DPU, which will decide next experiment step
    """
    global broadcastSocket
    global broadcastReady
    global redis_client
    global lock

    while True:
        while broadcastReady:
            ready = select.select([broadcastSocket], [], [], 2)

            if ready[0]:
                data = broadcastSocket.recv(4096)
                data = json.loads(data)
                
                broadcast_gui = EVOLVER_NS.broadcast(data)
                redis_client.set("broadcast", json.dumps(broadcast_gui))

        time.sleep(1)


def saved_exps():
    global EXPERIMENT_DATA_PATH
    dirs = []

    all = os.listdir(EXPERIMENT_DATA_PATH)
    for item in all:
        if not os.path.isfile(item):
            dirs += [item]
    dirs.remove("__pycache__")
    return dirs


def setup_logging(filename, quiet, verbose):
    if quiet:
        logging.basicConfig(level=logging.CRITICAL + 10)
    else:
        if verbose == 0:
            level = logging.INFO
        elif verbose >= 1:
            level = logging.DEBUG
        logging.basicConfig(
            format="%(asctime)s - %(name)s - [%(levelname)s] " "- %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=filename,
            level=level,
        )


if __name__ == "__main__":
    redis_client = redis.StrictRedis("127.0.0.1")
    redis_client.delete("socketio_answer")

    # Creates eVOLVER object and turns on leds
    EVOLVER_NS = EvolverDPU()
    EVOLVER_NS.update_led([2048 for i in range(16)])


    # Start by stopping any existing experiment
    EVOLVER_NS.stop_all_pumps()
    # EVOLVER_NS.fluid_command(['1|2' for i in range(48)])

    # Is experiment paused ?
    paused = False

    # Creates a broadcast thread, which will receive new data from hardware from evolver-server
    bServer = Thread(target=broadcast)
    bServer.start()    
    print("DPU on!")

    while True:
        try:
            # wait until there is a command in the queue (Redis variable)
            # command = {"payload": bytes, "reply": boolean}
            command = redis_client.brpop("socketio")
            command = json.loads(command[1].decode("UTF-8", errors="ignore").lower())

            # When api redis writing was updated, this will no longer be necessary
            if isinstance(command, str):
                command = json.loads(command)

            if command["command"] == "expt-config":
                experiment_params = command["payload"]
                config = EVOLVER_NS.config_exp(
                    experiment_params, 0, False, False
                )

                if config:
                    redis_client.lpush(
                        "socketio_answer", json.dumps({"expt-config-setted": None})
                    )

            elif command["command"] == "expt-start":
                EVOLVER_NS.start_time = EVOLVER_NS.initialize_exp(
                    command["payload"]["name"], False
                )

                redis_client.lpush(
                    "socketio_answer", json.dumps({"expt-started": None})
                )

            elif command["command"] == "expt-stop":
                if "vials" in command["payload"].keys():
                    EVOLVER_NS.stop_some_vials(command["payload"]["vials"])
                else:
                    EVOLVER_NS.stop_exp()

                redis_client.lpush(
                    "socketio_answer", json.dumps({"expt-stopped": None})
                )

            elif command["command"] == "command":
                lock.acquire()
                EVOLVER_NS.s.send(
                    functions["command"]["id"].to_bytes(1, "big")
                    + bytes(json.dumps(command["payload"]), "utf-8")
                    + b"\r\n"
                )
                lock.release()

            elif command["command"] == "get_active_calibrations":
                activelcal = EVOLVER_NS.request_calibrations()
                redis_client.lpush("socketio_answer", json.dumps(activelcal))

            elif command["command"] == "appendcal":
                response = EVOLVER_NS.appendcal(command["payload"])
                redis_client.lpush("socketio_answer", json.dumps(response))

            elif command["command"] == "getfitnames":
                fitnames = EVOLVER_NS.getfitnames()
                redis_client.lpush("socketio_answer", json.dumps(fitnames))

            elif command["command"] == "getcalibrationnames":
                calnames = EVOLVER_NS.getcalibrationnames()
                redis_client.lpush("socketio_answer", json.dumps(calnames))

            elif command["command"] == "getcalibration":
                calibration = EVOLVER_NS.getcalibration(command["payload"])
                redis_client.lpush("socketio_answer", json.dumps(calibration))

            elif command["command"] == "setfitcalibrations":
                EVOLVER_NS.setfitcalibrations(command["payload"])

            # TODO: define default answer to all commands
            elif command["command"] == "setactiveodcal":
                EVOLVER_NS.setactiveodcal(command["payload"])
                redis_client.lpush("socketio_answer", "ok")

            elif command["command"] == "setrawcalibration":
                ans = EVOLVER_NS.setrawcalibration(command["payload"])
                redis_client.lpush("socketio_answer", ans)

            elif command["command"] == "getdevicename":
                ans = EVOLVER_NS.get_device_name()
                redis_client.lpush("socketio_answer", ans)

            elif command["command"] == "getupdateinterval":
                ans = EVOLVER_NS.get_update_interval()
                redis_client.lpush("socktio_answer", ans)

            elif command["command"] == "getallcalibrations":
                ans = EVOLVER_NS.get_all_calibrations()
                redis_client.lpush("socketio_answer", json.dumps(ans))

            elif command["command"] == "setodcalibration":
                ans = EVOLVER_NS.setodcalibration(command["payload"])
                redis_client.lpush("socketio_answer", json.dumps(ans))

            elif command["command"] == "settempcalibration":
                ans = EVOLVER_NS.settempcalibration(command["payload"])
                redis_client.lpush("socketio_answer", json.dumps(ans))

            else:
                print(command)

            time.sleep(0.1)

        except KeyboardInterrupt:
            try:
                print("Ctrl-C detected, pausing experiment")
                logger.warning("interrupt received, pausing experiment")
                EVOLVER_NS.stop_exp()
                # stop receiving broadcasts
                EVOLVER_NS.disconnect()

                while True:
                    key = input(
                        "Experiment paused. Press enter key to restart or hit Ctrl-C again to terminate experiment"
                    )
                    logger.warning("resuming experiment")
                    # no need to have something like "restart_chemo" here
                    # with the new server logic
                    EVOLVER_NS.connect()
                    break

            except KeyboardInterrupt:
                print("Second Ctrl-C detected, shutting down")
                logger.warning("second interrupt received, terminating experiment")
                EVOLVER_NS.stop_exp()
                print("Experiment stopped, goodbye!")
                logger.warning("experiment stopped, goodbye!")
                break

        except Exception as e:
            logger.critical("exception %s stopped the experiment" % str(e))
            print('error "%s" stopped the experiment' % str(e))
            traceback.print_exc(file=sys.stdout)
            EVOLVER_NS.stop_exp()
            print("Experiment stopped, goodbye!")
            logger.warning("experiment stopped, goodbye!")
            break

    # stop experiment one last time
    # covers corner case where user presses Ctrl-C twice quickly
    EVOLVER_NS.connect()
    EVOLVER_NS.stop_exp()
