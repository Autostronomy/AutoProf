import os
import sys

from . import autoprofutils, Pipeline, pipeline_steps

try:
    from ._version import version as __version__  # noqa
except ModuleNotFoundError:
    __version__ = "0.0.0"
    import warnings

    warnings.warn(
        "WARNING: AutoProf version number not found. This is likely because you are running Autoprof from a source directory."
    )

__author__ = "Connor Stone"
__email__ = "connorstone628@gmail.com"


def run_from_terminal():

    assert (
        len(sys.argv) >= 2
    ), "Please supply a config file to AutoProf. See the examples in the 'test' folder that came with AutoProf"

    config_file = sys.argv[1]

    if config_file.strip().lower() == "--version":
        print(__version__)
        return
    try:
        if ".log" == sys.argv[2][-4:]:
            logfile = sys.argv[2]
        else:
            logfile = None
    except:
        logfile = None

    PIPELINE = Pipeline.Isophote_Pipeline(loggername=logfile)

    PIPELINE.Process_ConfigFile(config_file)
