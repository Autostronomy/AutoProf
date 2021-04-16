
process_mode = 'image'

image_file = 'test_ESO479-G1_r.fits'
pixscale = 0.262
name = 'testimage'
doplot = True
overflowval = 10.
given_center = {'x': 1400, 'y': 1400}
#fit_limit = 0.1
isoband_width = 0.05
samplegeometricscale = 0.05
truncate_evaluation = True
new_pipeline_steps = ['background', 'psf', 'center', 'isophoteinit',                                                                                 
                      'isophotefit', 'isophoteextract', 'checkfit',
                      'ellipsemodel general', 'orthsample', 'radsample']

orthsample_parallel = True
