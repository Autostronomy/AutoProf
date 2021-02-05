#!/usr/bin/python3

import os
import sys
sys.path.append(os.environ['AUTOPROF'])
from Pipeline import Isophote_Pipeline

assert len(sys.argv) >= 2

config_file = sys.argv[1]

try:
    if '.log' == sys.argv[2][-4:]:
        logfile = sys.argv[2]
    else:
        logfile = None
except:
    logfile = None

PIPELINE = Isophote_Pipeline(loggername = logfile)

PIPELINE.Process_ConfigFile(config_file)
