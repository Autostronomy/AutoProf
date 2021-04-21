
process_mode = 'image'

image_file = 'test_ESO479-G1_r.fits'
pixscale = 0.262
name = 'testtreeimage'
doplot = True
overflowval = 10.
isoband_width = 0.05
samplegeometricscale = 0.05
truncate_evaluation = True
new_pipeline_methods = {'branch extra': lambda d,r,o: 'extra'}
new_pipeline_steps = {'head': ['background', 'psf', 'center', 'isophoteinit',                                                                                 
                               'isophotefit', 'isophoteextract', 'checkfit',
                               'writeprof', 'branch extra'],
                      'extra': ['ellipsemodel', 'orthsample', 'radsample']}
