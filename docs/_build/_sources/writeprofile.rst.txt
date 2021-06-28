===================
Writing The Profile
===================

Description
-----------

**pipeline label: writeprof**

This step writes the results of the AutoProf pipeline analysis to a file.
There are two files written, a *.prof* file containing the surface brightness profile and acompanying measurements, and a *.aux* file containing global results, messages, and setting used for the pipeline.
The *.prof* file looks for specific keywords in the results dictionary: *prof header*, *prof units*, *prof data*, and *prof format*.
There are the results from the isophotal fitting step.
*prof header* gives the column names for the profile, *prof units* is a dictionary which gives the corresponding units for each column header key, *prof data* is a dictionary containing a list of values for each header key, and *prof format* is a dictionary which gives the python string format for values under each header key (for example '%.4f' gives a number to 4 decimal places).
The profile is written with comma (or a user specified delimiter) separation for each value, where each row corresponds to a given isophote at increasing semi-major axis values.

The *.aux* file has a less strict format than the *.prof* file.
The first line records the date and time that the file was written, the second line gives the name of the object as specified by the user or the filename.
The next lines are taken from the results dictionary, any result key with *auxfile* in the name is taken as a message for the *.aux* file and written (in alphabetical order by key) to the file.
See the pipeline step output formats for the messages that are included in the *.aux* file.
Finally, a record of the user specified options is included for reference.

Output format:
no results added from this step
.. code-block:: python
   
  {}

Config Parameters
-----------------

ap_delimiter
  Delimiter character used to separate values in output profile. Will default to a comma (",") if not given (string)

ap_profile_format
  Choose the output format for the profile, options are ['csv', 'fits']. Default is 'csv' (string)
