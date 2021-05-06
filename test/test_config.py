
ap_process_mode = 'image'

ap_image_file = 'test_ESO479-G1_r.fits'
ap_name = 'testimage'
ap_pixscale = 0.262
ap_doplot = True
ap_isoband_width = 0.05
ap_samplegeometricscale = 0.05
ap_truncate_evaluation = True
ap_isoclip = True

ap_slice_anchor = {'x': 1700., 'y': 1350.}
ap_slice_length = 300.

ap_new_pipeline_steps = ['background', 'psf', 'center', 'isophoteinit',                                                                                 
                         'plot image', 'isophotefit', 'isophoteextract',
                         'checkfit', 'ellipsemodel general', 'ellipsemodel',
                         'axialprofiles', 'radialprofiles', 'sliceprofile',
                         'writeprof']
