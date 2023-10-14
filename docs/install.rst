============
Installation
============

Requirements
------------

numpy, scipy, matplotlib, astropy, photutils, scikit-learn

If you have difficulty running AutoProf, it is possible that one of these dependencies is not in its latest (Python3) version and you should try updating.

Basic Install
-------------

This is now very easy::

    pip install autoprof

Once you have it installed you may want to test that everyhting works on your system.

#. Go to this link to `download the test scripts <https://github.com/Autostronomy/AutoProf/tree/main/test>`_. 
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
  
- If you are using python 3.9 you may have difficulties installing photutils. For now you will have to revert to python 3.8

For other issues contact connorstone628@gmail.com for help. The code has been tested on Linux (mint) and Mac machines.
