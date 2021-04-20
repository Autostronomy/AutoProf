import sys
import os
sys.path.append(os.environ['AUTOPROF'])
from autoprofutils.Background import Background_Mode, Background_DilatedSources, Background_Unsharp
from autoprofutils.PSF import PSF_IRAF, PSF_StarFind
from autoprofutils.Center import Center_2DGaussian, Center_1DGaussian, Center_OfMass, Center_HillClimb, Center_Forced
from autoprofutils.Isophote_Initialize import Isophote_Initialize
from autoprofutils.Isophote_Fit import Isophote_Fit_FFT_Robust, Isophote_Fit_Forced, Photutils_Fit
from autoprofutils.Mask import Star_Mask_IRAF, NoMask, Mask_Segmentation_Map
from autoprofutils.Isophote_Extract import Isophote_Extract, Isophote_Extract_Forced
from autoprofutils.Check_Fit import Check_Fit
from autoprofutils.Write_Prof import WriteProf
from autoprofutils.Ellipse_Model import EllipseModel_Fix, EllipseModel_General
from autoprofutils.Radial_Sample import Radial_Sample
from autoprofutils.Orthogonal_Sample import Orthogonal_Sample
from autoprofutils.SharedFunctions import GetKwargs, Read_Image, PA_shift_convention
from multiprocessing import Pool, current_process
from astropy.io import fits
from scipy.stats import iqr
from itertools import starmap
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
        self.pipeline_functions = {'background': Background_Mode,
                                   'background dilatedsources': Background_DilatedSources,
                                   'background unsharp': Background_Unsharp,
                                   'psf': PSF_StarFind,
                                   'psf IRAF': PSF_IRAF,
                                   'center': Center_HillClimb,
                                   'center forced': Center_Forced,
                                   'center 2DGaussian': Center_2DGaussian,
                                   'center 1DGaussian': Center_1DGaussian,
                                   'center OfMass': Center_OfMass,
                                   'isophoteinit': Isophote_Initialize,
                                   'isophotefit': Isophote_Fit_FFT_Robust,
                                   'isophotefit forced': Isophote_Fit_Forced,
                                   'isophotefit photutils': Photutils_Fit,
                                   'starmask': Star_Mask_IRAF,
                                   'starmask overflowonly': NoMask,
                                   'mask segmentation map': Mask_Segmentation_Map,
                                   'isophoteextract': Isophote_Extract,
                                   'isophoteextract forced': Isophote_Extract_Forced,
                                   'checkfit': Check_Fit,
                                   'writeprof': WriteProf,
                                   'ellipsemodel': EllipseModel_Fix,
                                   'ellipsemodel general': EllipseModel_General,
                                   'radsample': Radial_Sample,
                                   'orthsample': Orthogonal_Sample}
        
        # Default pipeline analysis order
        self.pipeline_steps = {'head': ['background', 'psf', 'center', 'isophoteinit',
                                        'isophotefit', 'isophoteextract', 'checkfit', 'writeprof']}

        # Start the logger
        logging.basicConfig(level=logging.INFO, filename = 'AutoProf.log' if loggername is None else loggername, filemode = 'w')

    def UpdatePipeline(self, new_pipeline_functions = None, new_pipeline_steps = None):
        """
        modify steps in the AutoProf pipeline.

        new_pipeline_functions: update the dictionary of functions used by the pipeline. This can either add
                                new functions or replace existing ones.
        new_pipeline_steps: update the list of pipeline step strings. These strings refer to keys in
                            pipeline_functions. It is posible to add/remove/rearrange steps here. Alternatively
                            one can supply a dictionary with current pipeline steps as keys and new pipeline
                            steps as values, the corresponding steps will be replaced.
        preprocess: Function to process image before AutoProf analysis begins.
        """
        if new_pipeline_functions:
            logging.info('PIPELINE updating these pipeline functions: %s' % str(new_pipeline_functions.keys()))
            self.pipeline_functions.update(new_pipeline_functions)
        if new_pipeline_steps:
            if type(new_pipeline_steps) == list:
                logging.info('PIPELINE new steps: %s' % (str(new_pipeline_steps)))
                self.pipeline_steps['head'] = new_pipeline_steps
            elif type(new_pipeline_steps) == dict:
                logging.info('PIPELINE new steps: %s' % (str(new_pipeline_steps)))
                assert 'head' in new_pipeline_steps.keys()
                self.pipeline_steps = new_pipeline_steps
            
    def Process_Image(self, kwargs_internal = {}, **kwargs):
        """
        Function which runs the pipeline for a single image. Each sub-function of the pipeline is run
        in order and the outputs are passed along. If multiple images are given, the pipeline is
        excecuted on the first image and the isophotes are applied to the others.

        returns list of times for each pipeline step if successful. else returns 1
        """

        kwargs.update(kwargs_internal)

        # Seed the random number generator in numpy so each thread gets unique random numbers
        try:
            sleep(0.01)
            np.random.seed(int(np.random.randint(10000)*current_process().pid*(time() % 1) % 2**15))
        except:
            pass
        
        # use filename if no name is given
        if not ('name' in kwargs and type(kwargs['name']) == str):
            kwargs['name'] = IMG[(IMG.rfind('/') if '/' in IMG else 0):IMG.find('.', (IMG.rfind('/') if '/' in IMG else 0))]

        # Read the primary image
        try:
            dat = Read_Image(IMG, **kwargs)
        except:
            logging.error('%s: could not read image %s' % (kwargs['name'], str(IMG)))
            return 1
            
        # Check that image data exists and is not corrupted
        if dat is None or np.all(dat[int(len(dat)/2.)-10:int(len(dat)/2.)+10, int(len(dat[0])/2.)-10:int(len(dat[0])/2.)+10] == 0):
            logging.error('%s Large chunk of data missing, impossible to process image' % kwargs['name'])
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
                logging.info('%s: %s %s at: %.1f sec' % (kwargs['name'], key, self.pipeline_steps[key][step], time() - start))
                print('%s: %s %s at: %.1f sec' % (kwargs['name'], key, self.pipeline_steps[key][step], time() - start))
                if 'branch' in self.pipeline_steps[key][step]:
                    decision = self.pipeline_functions[self.pipeline_steps[key][step]](dat, results, **kwargs)
                    if type(decision) == str:
                        key = decision
                        step = 0
                    else:
                        step += 1
                else:
                    step_start = time()
                    dat, res = self.pipeline_functions[self.pipeline_steps[key][step]](dat, results, **kwargs)
                    results.update(res)
                    timers[self.pipeline_steps[key][step]] = time() - step_start
                    step += 1
            except Exception as e:
                logging.error('%s: on step %s got error: %s' % (kwargs['name'], self.pipeline_steps[step], str(e)))
                logging.error('%s: with full trace: %s' % (kwargs['name'], traceback.format_exc()))
                return 1
            
        print('%s: Processing Complete! (at %.1f sec)' % (kwargs['name'], time() - start))
        logging.info('%s: Processing Complete! (at %.1f sec)' % (kwargs['name'], time() - start))
        return timers
    
    def Process_List(self, **kwargs):
        """
        Wrapper function to run "Process_Image" in parallel for many images.
        
        IMG: list of strings containing image file paths
        pixscale: angular pixel size in arcsec/pixel
        n_procs: number of processors to use
        """

        assert type(kwargs['image_file']) == list
        
        # Format the inputs so that they can be zipped with the images files
        # and passed to the Process_Image function.
        if all(type(v) != list for v in kwargs.values()):
            use_kwargs = [kwargs]*len(kwargs['image_file'])
        else:
            use_kwargs = []
            for i in range(len(kwargs['image_file'])):
                tmp_kwargs = {}
                for k in kwargs.keys():
                    if type(kwargs[k]) == list:
                        tmp_kwargs[k] = kwargs[k][i]
                    else:
                        tmp_kwargs[k] = kwargs[k]
                use_kwargs.append(tmp_kwargs)
        # Track how long it takes to run the analysis
        start = time()
        
        # Create a multiprocessing pool to parallelize image processing
        if kwargs['n_procs'] > 1:
            with Pool(int(kwargs['n_procs'])) as pool:
                res = pool.map(self.Process_Image, use_kwargs,
                               chunksize = 5 if len(kwargs['image_file']) > 100 else 1)
        else:
            res = list(map(self.Process_Image, use_kwargs))
            
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
            use_config = config_file[startat:config_file.find('.', startat)]
        else:
            use_config = config_file[startat:]
        if '/' in config_file:
            sys.path.append(config_file[:config_file.rfind('/')])
        try:
            c = importlib.import_module(use_config)
        except:
            sys.path.append(os.getcwd())
            c = importlib.import_module(use_config)

        if 'forced' in c.process_mode:
            self.UpdatePipeline(new_pipeline_steps = ['background', 'psf', 'center forced', 'isophoteinit',
                                                      'isophotefit forced', 'isophoteextract forced', 'writeprof'])
            
        try:
            self.UpdatePipeline(new_pipeline_functions = c.new_pipeline_functions)
        except:
            pass
        try:
            self.UpdatePipeline(new_pipeline_steps = c.new_pipeline_steps)
        except:
            pass
            
        use_kwargs = GetKwargs(c)
            
        if c.process_mode in ['image', 'forced image']:
            return self.Process_Image(use_kwargs)
        elif c.process_mode in ['image list', 'forced image list']:
            return self.Process_List(**use_kwargs)
        else:
            logging.error('Unrecognized process_mode! Should be in: [image, image list, forced image, forced image list]')
            return 1
        
