import numpy as np

ap_process_mode = 'image'

ap_image_file = 'test_ESO479-G1_r.fits'
ap_pixscale = 0.262
ap_name = 'testtreeimage'
ap_doplot = True
ap_isoband_width = 0.05
ap_samplegeometricscale = 0.05
ap_truncate_evaluation = True

ap_fouriermodes = 4
ap_slice_anchor = {'x': 1700., 'y': 1350.}
ap_slice_length = 300.
ap_isoclip = True

def My_Edgon_Fit_Method(IMG, results, options):
    N = 100
    return IMG, {'fit ellip': np.array([results['init ellip']]*N),
                 'fit pa': np.array([results['init pa']]*N),
                 'fit ellip_err': np.array([0.05]*N),
                 'fit pa_err': np.array([5*np.pi/180]*N),
                 'fit R': np.logspace(0,np.log10(results['init R']*2),N)}
ap_new_pipeline_methods = {'branch edgeon': lambda IMG,results,options: ('edgeon' if results['init ellip'] > 0.8 else 'standard',{}),
		           'edgeonfit': My_Edgon_Fit_Method}
ap_new_pipeline_steps = {'head': ['background', 'psf', 'center', 'isophoteinit', 'branch edgeon'],
		         'standard': ['isophotefit', 'isophoteextract', 'checkfit', 'writeprof', 'plot image', 'ellipsemodel general', 'ellipsemodel', 'axialprofiles', 'radialprofiles', 'sliceprofile'],
		         'edgeon': ['edgeonfit', 'isophoteextract', 'writeprof', 'axialprofiles', 'radsample']}
