============
Installation
============

Requirements
------------

numpy, scipy, matplotlib, astropy, photutils, scikit-learn

If you have difficulty running AutoProf, it is possible that one of these dependencies is not in its latest (Python3) version and you should try updating.

Basic Install
-------------

1. Download the package from: `AutoProf <https://github.com/ConnorStoneAstro/AutoProf>`_::
   
     cd /where/you/want/AutoProf/to/live/
     git clone git@github.com:ConnorStoneAstro/AutoProf.git
   
   If you experience difficulty cloning the package, you may download a zip file of the package from the github page.
#. Alias the AutoProf function. To make this permanent, include this lines in your .bashrc file (or equivalent for your OS). ::
   
     alias autoprof='/path/to/AutoProf/autoprof/autoprof.py'
   
#. Run the test cases to see that all is well::
   
     cd /path/to/AutoProf/test/
     autoprof test_config.py
     autoprof test_forced_config.py Forced.log
     autoprof test_batch_config.py Batch.log
     autoprof test_tree_config.py Tree.log
   
   This will test a basic AutoProf run on a single galaxy, forced photometry of the galaxy on itself, and batch photometry for multiple images (which are actually the same in this case), and pipeline decision tree construction, respectively. You can also run all of these tests in one command by executing the *runtests.sh* script in bash.
#. Check the diagnostic plots to see what AutoProf can do! You should get a fit that looks like this one, and a whole lot more images to show what is going on.

.. image:: _static/fit_ellipse_testimage.jpg

Issues
------

- Each analysis run should end with the words "Processing Complete!" if you don't get that, then the pipeline probably crashed at some point, check the log file (probably called *AutoProf.log*) for more information
- If you get *Permission Denied*, it is possible that the file is not listed as executable and you need to run::
  
    cd /path/to/AutoProf/
    chmod 755 autoprof.py
  
- If you have everything set up, but are getting strange errors such as *ImportError: No module named photutils* even when photutils is already installed it is possible that your python3 installation lives somewhere unusual. Try executing::
  
    which python3
  
  to see where your python3 installation lives. If the result is something other than */usr/bin/python3* then you need to make a small edit to the *autoprof.py* file. In the first line make the change from::
  
    #!/bin/bash/python3
  
  to instead be::
  
    #!/wherever/your/python3/is/installed
  
- If you are using python 3.9 you may have difficulties installing photutils. For now you will have to revert to python 3.8

For other issues contact connor.stone@queensu.ca for help. The code has been tested on Linux (mint) and Mac machines.
