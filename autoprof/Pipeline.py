import sys
import os
from .pipeline_steps import *
# from .pipeline_steps.Plotting_Steps import Plot_Galaxy_Image
# from .pipeline_steps.Background import Background_Mode, Background_DilatedSources, Background_Unsharp, Background_Basic
# from .pipeline_steps.PSF import PSF_IRAF, PSF_StarFind, PSF_Image, PSF_deconvolve
# from .pipeline_steps.Center import Center_2DGaussian, Center_1DGaussian, Center_OfMass, Center_HillClimb, Center_Forced, Center_HillClimb_mean
# from .pipeline_steps.Isophote_Initialize import Isophote_Initialize, Isophote_Initialize_mean, Isophote_Init_Forced
# from .pipeline_steps.Isophote_Fit import Isophote_Fit_FFT_Robust, Isophote_Fit_Forced, Photutils_Fit, Isophote_Fit_FFT_mean, Isophote_Fit_FixedPhase
# from .pipeline_steps.Mask import Star_Mask_IRAF, Mask_Segmentation_Map, Bad_Pixel_Mask, Star_Mask
# from .pipeline_steps.Isophote_Extract import Isophote_Extract, Isophote_Extract_Forced, Isophote_Extract_Photutils
# from .pipeline_steps.Check_Fit import Check_Fit
# from .pipeline_steps.Write_Prof import WriteProf
# from .pipeline_steps.Write_Fi import WriteFi
# from .pipeline_steps.Ellipse_Model import EllipseModel
# from .pipeline_steps.Radial_Profiles import Radial_Profiles
# from .pipeline_steps.Axial_Profiles import Axial_Profiles
# from .pipeline_steps.Slice_Profiles import Slice_Profile
from .autoprofutils.ImageTransform import Crop
from .autoprofutils.SharedFunctions import GetOptions, Read_Image, PA_shift_convention
from multiprocessing import Pool, current_process
from astropy.io import fits
from scipy.stats import iqr
from itertools import starmap
from functools import partial
import importlib
import numpy as np
from time import time, sleep
import logging
import warnings
import traceback
from astropy.io.fits.verify import VerifyWarning
warnings.simplefilter('ignore', category=VerifyWarning)

class Isophote_Pipeline(object):

    def __init__(self, loggername = None):
        """
        Initialize pipeline object, user can replace functions with their own if they want, otherwise defaults are used.

        loggername: String to use for logging messages
        """

        # Functions avaiable by default for building the pipeline
        self.pipeline_methods = {'background': Background_Mode,
                                 'background dilatedsources': Background_DilatedSources,
                                 'background unsharp': Background_Unsharp,
                                 'background basic': Background_Basic,
                                 'psf': PSF_StarFind,
                                 'psf IRAF': PSF_IRAF,
                                 'psf img': PSF_Image,
                                 'psf deconvolve': PSF_deconvolve,
                                 'center': Center_HillClimb,
                                 'center mean': Center_HillClimb_mean,
                                 'center forced': Center_Forced,
                                 'center 2DGaussian': Center_2DGaussian,
                                 'center 1DGaussian': Center_1DGaussian,
                                 'center OfMass': Center_OfMass,
                                 'crop': Crop,
                                 'isophoteinit': Isophote_Initialize,
                                 'isophoteinit forced': Isophote_Init_Forced,
                                 'isophoteinit mean': Isophote_Initialize_mean,
                                 'plot image': Plot_Galaxy_Image,
                                 'writefi': WriteFi,
                                 'isophotefit': Isophote_Fit_FFT_Robust,
                                 'isophotefit fixed': Isophote_Fit_FixedPhase,
                                 'isophotefit mean': Isophote_Fit_FFT_mean,
                                 'isophotefit forced': Isophote_Fit_Forced,
                                 'isophotefit photutils': Photutils_Fit,
                                 'mask badpixels': Bad_Pixel_Mask,
                                 'starmask': Star_Mask,
                                 'starmask IRAF': Star_Mask_IRAF,
                                 'mask segmentation map': Mask_Segmentation_Map,
                                 'isophoteextract': Isophote_Extract,
                                 'isophoteextract photutils': Isophote_Extract_Photutils,
                                 'isophoteextract forced': Isophote_Extract_Forced,
                                 'checkfit': Check_Fit,
                                 'writeprof': WriteProf,
                                 'ellipsemodel': EllipseModel,
                                 'radialprofiles': Radial_Profiles,
                                 'sliceprofile': Slice_Profile,
                                 'axialprofiles': Axial_Profiles}
        
        # Default pipeline analysis order
        self.pipeline_steps = {'head': ['background', 'psf', 'center', 'isophoteinit',
                                        'isophotefit', 'isophoteextract', 'checkfit', 'writeprof']}
        
        # Start the logger
        logging.basicConfig(level=logging.INFO, filename = 'AutoProf.log' if loggername is None else loggername, filemode = 'w')

    def UpdatePipeline(self, new_pipeline_methods = None, new_pipeline_steps = None):
        """
        modify steps in the AutoProf pipeline.

        new_pipeline_methods: update the dictionary of methods used by the pipeline. This can either add
                                new methods or replace existing ones.
        new_pipeline_steps: update the list of pipeline step strings. These strings refer to keys in
                            pipeline_methods. It is posible to add/remove/rearrange steps here. Alternatively
                            one can supply a dictionary with current pipeline steps as keys and new pipeline
                            steps as values, the corresponding steps will be replaced.
        """
        if new_pipeline_methods:
            logging.info('PIPELINE updating these pipeline methods: %s' % str(new_pipeline_methods.keys()))
            self.pipeline_methods.update(new_pipeline_methods)
        if new_pipeline_steps:
            logging.info('PIPELINE new steps: %s' % (str(new_pipeline_steps)))
            if type(new_pipeline_steps) == list:
                self.pipeline_steps['head'] = new_pipeline_steps
            elif type(new_pipeline_steps) == dict:
                assert 'head' in new_pipeline_steps.keys()
                self.pipeline_steps = new_pipeline_steps
            
    def Process_Image(self, options = {}):
        """
        Function which runs the pipeline for a single image. Each sub-function of the pipeline is run
        in order and the outputs are passed along. If multiple images are given, the pipeline is
        excecuted on the first image and the isophotes are applied to the others.

        returns list of times for each pipeline step if successful. else returns 1
        """

        # Remove any options with None value so they don't interfere with analysis logic
        for key in list(options.keys()):
            if options[key] is None:
                del options[key]
        
        # Seed the random number generator in numpy so each thread gets unique random numbers
        try:
            sleep(0.01)
            np.random.seed(int(np.random.randint(10000)*current_process().pid*(time() % 1) % 2**15))
        except:
            pass
        
        # use filename if no name is given
        if not ('ap_name' in options and type(options['ap_name']) == str):
            base = os.path.split(options['ap_image_file'])[1]
            options['ap_name'] = os.path.splitext(base)[0]

        # Read the primary image
        try:
            dat = Read_Image(options['ap_image_file'], options)
        except:
            logging.error('%s: could not read image %s' % (options['ap_name'], options['ap_image_file']))
            return 1
            
        # Check that image data exists and is not corrupted
        if dat is None or np.all(dat[int(len(dat)/2.)-10:int(len(dat)/2.)+10, int(len(dat[0])/2.)-10:int(len(dat[0])/2.)+10] == 0):
            logging.error('%s Large chunk of data missing, impossible to process image' % options['ap_name'])
            return 1
        
        # Track time to run analysis
        start = time()
        
        # Run the Pipeline
        timers = {}
        results = {}

        key = 'head'
        step = 0
        while step < len(self.pipeline_steps[key]):
            try:
                logging.info('%s: %s %s at: %.1f sec' % (options['ap_name'], key, self.pipeline_steps[key][step], time() - start))
                print('%s: %s %s at: %.1f sec' % (options['ap_name'], key, self.pipeline_steps[key][step], time() - start))
                if 'branch' in self.pipeline_steps[key][step]:
                    decision, newoptions = self.pipeline_methods[self.pipeline_steps[key][step]](dat, results, options)
                    options.update(newoptions)
                    if type(decision) == str:
                        key = decision
                        step = 0
                    else:
                        step += 1
                else:
                    step_start = time()
                    dat, res = self.pipeline_methods[self.pipeline_steps[key][step]](dat, results, options)
                    results.update(res)
                    timers[self.pipeline_steps[key][step]] = time() - step_start
                    step += 1
            except Exception as e:
                logging.error('%s: on step %s got error: %s' % (options['ap_name'], self.pipeline_steps[key][step], str(e)))
                logging.error('%s: with full trace: %s' % (options['ap_name'], traceback.format_exc()))
                return 1
            
        print('%s: Processing Complete! (at %.1f sec)' % (options['ap_name'], time() - start))
        logging.info('%s: Processing Complete! (at %.1f sec)' % (options['ap_name'], time() - start))
        return timers
    
    def Process_List(self, options):
        """
        Wrapper function to run "Process_Image" in parallel for many images.
        """

        assert type(options['ap_image_file']) == list
        
        # Format the inputs so that they can be zipped with the images files
        # and passed to the Process_Image function.
        use_options = []
        for i in range(len(options['ap_image_file'])):
            tmp_options = {}
            for k in options.keys():
                if type(options[k]) == list and not k in ['ap_new_pipeline_steps']:
                    tmp_options[k] = options[k][i]
                else:
                    tmp_options[k] = options[k]
            use_options.append(tmp_options)
        # Track how long it takes to run the analysis
        start = time()
        
        # Create a multiprocessing pool to parallelize image processing
        n_procs = options['ap_n_procs'] if 'ap_n_procs' in options else 1
        if n_procs > 1:
            with Pool(int(n_procs)) as pool:
                res = pool.map(self.Process_Image, use_options,
                               chunksize = 5 if len(options['ap_image_file']) > 100 else 1)
        else:
            res = list(map(self.Process_Image, use_options))
            
        # Report completed processing, and track time used
        logging.info('All Images Finished Processing at %.1f' % (time() - start))
        timers = dict()
        counts = dict()
        for r in res:
            if r == 1:
                continue
            for s in r.keys():
                if s in timers:
                    timers[s] += r[s]
                    counts[s] += 1.
                else:
                    timers[s] = r[s]
                    counts[s] = 1.
        if len(timers) == 0:
            logging.error('All images failed to process!')
            return 
        for s in timers:
            timers[s] /= counts[s]
            logging.info('%s took %.3f seconds on average' % (s, timers[s]))
        
        # Return the success/fail indicators for every Process_Image excecution
        return res
        
    def Process_ConfigFile(self, config_file):
        """
        Reads in a configuration file and sets parameters for the pipeline. The configuration
        file should have variables corresponding to the desired parameters to be set.

        congig_file: string path to configuration file

        returns: timing of each pipeline step if successful. Else returns 1
        """
        
        # Import the config file regardless of where it is from
        if '/' in config_file:
            startat = config_file.rfind('/')+1
        else:
            startat = 0
        if '.' in config_file:
            use_config = config_file[startat:config_file.rfind('.')]
        else:
            use_config = config_file[startat:]
        if '/' in config_file:
            sys.path.append(config_file[:config_file.rfind('/')])
        try:
            c = importlib.import_module(use_config)
        except:
            sys.path.append(os.getcwd())
            c = importlib.import_module(use_config)

        if 'forced' in c.ap_process_mode:
            self.UpdatePipeline(new_pipeline_steps = ['background', 'psf', 'center forced', 'isophoteinit forced',
                                                      'isophoteextract forced', 'writeprof'])
            
        try:
            self.UpdatePipeline(new_pipeline_methods = c.ap_new_pipeline_methods)
        except:
            pass
        try:
            self.UpdatePipeline(new_pipeline_steps = c.ap_new_pipeline_steps)
        except:
            pass
            
        use_options = GetOptions(c)
            
        if c.ap_process_mode in ['image', 'forced image']:
            return self.Process_Image(use_options)
        elif c.ap_process_mode in ['image list', 'forced image list']:
            return self.Process_List(use_options)
        else:
            logging.error('Unrecognized process_mode! Should be in: [image, image list, forced image, forced image list]')
            return 1
        
