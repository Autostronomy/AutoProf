import os
import sys

from . import autoprofutils, Pipeline, pipeline_steps

__version__ = "1.0.2"
__author__ = "Connor Stone"
__email__ = "connorstone628@gmail.com"

def run_from_terminal():

    assert (
        len(sys.argv) >= 2
    ), "Please supply a config file to AutoProf. See the examples in the 'test' folder that came with AutoProf"

    config_file = sys.argv[1]

    try:
        if ".log" == sys.argv[2][-4:]:
            logfile = sys.argv[2]
        else:
            logfile = None
    except:
        logfile = None

    PIPELINE = Pipeline.Isophote_Pipeline(loggername=logfile)

    PIPELINE.Process_ConfigFile(config_file)
    
