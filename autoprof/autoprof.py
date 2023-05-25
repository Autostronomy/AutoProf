#!/home/connor/venvs/PY39/bin/python3

import subprocess

p = subprocess.run("which python3", shell=True, stdout=subprocess.PIPE)
with open(__file__, "r") as f:
    raw = f.readlines()
if p.stdout.decode("UTF-8").strip() != raw[0][2:].strip():
    with open(__file__, "w") as f:
        raw[0] = "#!" + p.stdout.decode("UTF-8").strip() + "\n"
        f.writelines(raw)
    print(
        "Encountered a minor hiccup locating python3 and fixed it. Please just run again and everything should work.\nIf you encounter further issues such as 'numpy not found' even though numpy is installed, see the trouble shooting guide in the installation page of the documentation."
    )

import os
import sys

os.environ["AUTOPROF"] = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.environ["AUTOPROF"])
from Pipeline import Isophote_Pipeline

if __name__ == "__main__":
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

    PIPELINE = Isophote_Pipeline(loggername=logfile)

    PIPELINE.Process_ConfigFile(config_file)
