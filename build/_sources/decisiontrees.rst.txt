==============
Decision Trees
==============

AutoProf at its core is a pipeline building code, as such it has a
more advanced feature for constructing complex pipelines: decision
trees.  In a decision tree pipeline, one can essentially construct a
flow chart of decisions and corresponding methods to run and options
to use.  The beginning of the tree is always *'head'* and AutoProf
will continue to read those steps until it reaches a step containing
the word *branch* (any other text can be included in the step name so
you can write many different branches).  At a branch step, AutoProf
will provide the usual inputs, but the output should be a string or
*None* and a dictionary of new options (if any).  If *None* is
returned then AutoProf carries on along the same branch it is already
on.  If a string is returned, then that is taken as the key from which
to check for the next step in the pipeline steps object.  An empty
dictionary can be used to change no options.  When switching branches,
AutoProf will start at the beginning of the new branch.  Note, the new
branch can even be the branch you started on so watch out for infinite
loops!

For example, in a large sample, one may wish to process edge-on
galaxies differently than the others, but it may not be clear which
galaxies fit the edge-on criteria until the photometry is done.  In
this example, one could have AutoProf perform photometry up to the
point of the *isophoteinit* step, then the rest of the functions could
be chosen based on the ellipticity of the initialized ellipse.  To
make this work one could add:

.. code-block:: python
   
  def My_Edgeon_Fit_Method(IMG, results, options):
      N = 100
      return IMG, {'fit ellip': np.array([results['init ellip']]*N), 'fit pa': np.array([results['init pa']]*N),
                   'fit ellip_err': np.array([0.05]*N), 'fit pa_err': np.array([5*np.pi/180]*N),
                   'fit R': np.logspace(0,np.log10(results['init R']*2),N)}
      
  ap_new_pipeline_methods = {'branch edgeon': lambda IMG,results,options: ('edgeon' if results['init ellip'] > 0.8 else 'standard', {}),
          		     'edgeonfit': My_Edgeon_Fit_Method}
  
  ap_new_pipeline_steps = {'head': ['background', 'psf', 'center', 'isophoteinit', 'branch edgeon'],
		           'standard': ['isophotefit', 'isophoteextract', 'checkfit', 'writeprof'],
		           'edgeon': ['edgeonfit', 'isophoteextract', 'writeprof', 'axialprofiles', 'radialprofiles']}

in the config file. This config file would apply a standard pipeline
for face-on or moderately inclined galaxies, but a special pipeline
for edge-on galaxies which includes a user defined fitting function
*My_Edgeon_Fit_Method*, axial profiles, and radial profiles. This
example is included in the test folder as the *test_tree_config.py*
example config file.
