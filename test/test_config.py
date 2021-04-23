
ap_process_mode = 'image'

ap_image_file = 'test_ESO479-G1_r.fits'
ap_pixscale = 0.262
ap_name = 'testimage'
ap_doplot = True
ap_isoband_width = 0.05
ap_samplegeometricscale = 0.05
ap_truncate_evaluation = True
ap_badpixel_high = 9.9
ap_new_pipeline_steps = ['mask badpixels', 'background', 'psf', 'center', 'isophoteinit',                                                                                 
                         'isophotefit', 'isophoteextract', 'checkfit',
                         'ellipsemodel', 'orthsample', 'radsample', 'writeprof']
