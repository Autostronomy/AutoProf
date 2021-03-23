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
from autoprofutils.Ellipse_Model import EllipseModel_Fix, EllipseModel_General
from autoprofutils.Radial_Sample import Radial_Sample
from autoprofutils.SharedFunctions import GetKwargs, Read_Image
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
                                   'ellipsemodel': EllipseModel_Fix,
                                   'ellipsemodel general': EllipseModel_General,
                                   'radsample': Radial_Sample}
        
        # Default pipeline analysis order
        self.pipeline_steps = ['background', 'psf', 'center', 'isophoteinit', 'isophotefit',
                               'mask segmentation map', 'isophoteextract', 'checkfit', 'radsample', 'ellipsemodel']

        # Holder for any preprocessing function the user may want to apply
        self.preprocess = None

        # Start the logger
        logging.basicConfig(level=logging.INFO, filename = 'AutoProf.log' if loggername is None else loggername, filemode = 'w')

    def UpdatePipeline(self, new_pipeline_functions = None, new_pipeline_steps = None, preprocess = None):
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
                logging.info('PIPELINE new steps: %s' % str(new_pipeline_steps))
                self.pipeline_steps = new_pipeline_steps
            elif type(new_pipeline_steps) == dict:
                for k in new_pipeline_steps.keys():
                    logging.info('PIPELINE replacing "%s" pipeline step with "%s"' % (k, new_pipeline_steps[k]))
                    self.pipeline_steps[self.pipeline_steps.index(k)] = new_pipeline_steps[k]
        if preprocess:
            self.preprocess = preprocess

    def WriteProf(self, results, saveto, pixscale, name = None, **kwargs):
        """
        Writes the photometry information for disk given a photutils isolist object

        extractedprofile: surface brightness profile and all other data
        saveto: Full path string indicating where to save the profile
        pixscale: conversion factor between pixels and arcseconds (arcsec / pixel)
        starmask: Optional, a star mask to save along with the profile
        background: if saving the star mask, the background can also be saved
                    for full reproducability
        """

        with open(saveto + name + '.aux', 'w') as f:
            # write profile info
            f.write('name: %s\n' % str(name))
            f.write('pixel scale: %.3e arcsec/pix\n' % pixscale)
            if 'checkfit' in results:
                for k in results['checkfit'].keys():
                    f.write('check fit %s: %s\n' % (k, 'pass' if results['checkfit'][k] else 'fail'))
            f.write('psf fwhm: %.3f pix\n' % (results['psf fwhm']))
            try:
                f.write('background: %.5e +- %.2e flux/pix, noise: %.5e flux/pix\n' % (results['background'], results['background uncertainty'], results['background noise']))
            except:
                pass
            use_center = results['center']
            f.write('center x: %.2f pix, y: %.2f pix\n' % (use_center['x'], use_center['y']))
            if 'init ellip_err' in results and 'init pa_err' in results:
                f.write('global ellipticity: %.3f +- %.3f, pa: %.3f +- %.3f deg\n' % (results['init ellip'], results['init ellip_err'],
                                                                                      results['init pa']*180/np.pi, results['init pa_err']*180/np.pi))
            else:
                f.write('global ellipticity: %.3f, pa: %.3f deg\n' % (results['init ellip'], results['init pa']*180/np.pi))
            if len(kwargs) > 0:
                for k in kwargs.keys():
                    f.write('settings %s: %s\n' % (k,str(kwargs[k])))
            
        # Write the profile
        delim = kwargs['delimiter'] if 'delimiter' in kwargs else ','
        with open(saveto + name + '.prof', 'w') as f:
            # Write profile header
            f.write(delim.join(results['prof header']) + '\n')
            if 'prof units' in results:
                 f.write(delim.join(results['prof units'][h] for h in results['prof header']) + '\n')
            for i in range(len(results['prof data'][results['prof header'][0]])):
                line = list((results['prof format'][h] % results['prof data'][h][i]) for h in results['prof header'])
                f.write(delim.join(line) + '\n')
                
        # Write the mask data, if provided
        if 'mask' in results and (not results['mask'] is None) and 'savemask' in kwargs and kwargs['savemask']:
            header = fits.Header()
            header['IMAGE 1'] = 'star mask'
            header['IMAGE 2'] = 'overflow values mask'
            hdul = fits.HDUList([fits.PrimaryHDU(header=header),
                                 fits.ImageHDU(results['mask'].astype(int)),
                                 fits.ImageHDU(results['overflow mask'].astype(int))])
            hdul.writeto(saveto + name + '_mask.fits', overwrite = True)
            sleep(1)
            # Zip the mask file because it can be large and take a lot of memory, but in principle
            # is very easy to compress
            os.system('gzip -fq '+ saveto + name + '_mask.fits')
            
    def Process_Image(self, IMG, pixscale, saveto = None, name = None, kwargs_internal = {}, **kwargs):
        """
        Function which runs the pipeline for a single image. Each sub-function of the pipeline is run
        in order and the outputs are passed along. If multiple images are given, the pipeline is
        excecuted on the first image and the isophotes are applied to the others.
        
        IMG: string or list of strings providing the path to an image file
        pixscale: angular size of the pixels in arcsec/pixel
        saveto: string or list of strings indicating where to save profiles
        name: string name of galaxy in image, used for log files to make searching easier

        returns list of times for each pipeline step if successful. else returns 1
        """

        kwargs.update(kwargs_internal)

        try:
            sleep(0.01)
            np.random.seed(int(np.random.randint(10000)*current_process().pid*(time() % 1) % 2**15))
        except:
            pass
        
        # use filename if no name is given
        if name is None:
            name = IMG[(IMG.rfind('/') if '/' in IMG else 0):IMG.find('.', (IMG.rfind('/') if '/' in IMG else 0))]

        # Read the primary image
        try:
            dat = Read_Image(IMG, **kwargs)
        except:
            logging.error('%s: could not read image %s' % (name, str(IMG)))
            return 1
            
        # Check that image data exists and is not corrupted
        if dat is None or np.all(dat[int(len(dat)/2.)-10:int(len(dat)/2.)+10, int(len(dat[0])/2.)-10:int(len(dat[0])/2.)+10] == 0):
            logging.error('%s Large chunk of data missing, impossible to process image' % name)
            return 1
        
        # Save profile to the same folder as the image if no path is provided
        if saveto is None:
            saveto = './'
            
        # Track time to run analysis
        start = time()
        
        # Run the Pipeline
        timers = {}
        results = {}

        # Preprocess the image if needed
        if self.preprocess:
            logging.info('%s: Preprocessing Image' % name)
            print('%s: Preprocessing Image' % name)
            kwargs.update({'preprocess': self.preprocess})
            dat = self.preprocess(dat)
            timers['preprocess'] = time() - start
            
        for step in range(len(self.pipeline_steps)):
            try:
                step_start = time()
                logging.info('%s: %s at: %.1f sec' % (name, self.pipeline_steps[step], time() - start))
                print('%s: %s at: %.1f sec' % (name, self.pipeline_steps[step], time() - start))
                results.update(self.pipeline_functions[self.pipeline_steps[step]](dat, pixscale, name, results, **kwargs))
                timers[self.pipeline_steps[step]] = time() - step_start
            except Exception as e:
                logging.error('%s: on step %s got error: %s' % (name, self.pipeline_steps[step], str(e)))
                logging.error('%s: with full trace: %s' % (name, traceback.format_exc()))
                return 1

        # Save the profile
        print('%s: saving at: %.1f sec' % (name, time() - start))
        logging.info('%s: saving at: %.1f sec' % (name, time() - start))
        self.WriteProf(results, saveto, pixscale, name = name, **kwargs)
                
        print('%s: Processing Complete! (at %.1f sec)' % (name, time() - start))
        logging.info('%s: Processing Complete! (at %.1f sec)' % (name, time() - start))
        return timers
    
    def Process_List(self, IMG, pixscale, n_procs = 4, saveto = None, name = None, **kwargs):
        """
        Wrapper function to run "Process_Image" in parallel for many images.
        
        IMG: list of strings containing image file paths
        pixscale: angular pixel size in arcsec/pixel
        n_procs: number of processors to use
        saveto: list of strings containing file paths to save profiles
        name: names of the galaxies, used for logging
        """

        assert type(IMG) == list
        
        # Format the inputs so that they can be zipped with the images files
        # and passed to the Process_Image function.
        if type(pixscale) in [float, int]:
            use_pixscale = [float(pixscale)]*len(IMG)
        else:
            use_pixscale = pixscale
        if saveto is None:
            use_saveto = [None]*len(IMG)
        elif type(saveto) == list:
            use_saveto = saveto
        elif type(saveto) == str:
            use_saveto = [saveto]*len(IMG)
        else:
            raise TypeError('saveto should be a list or string')
        if name is None:
            use_name = [None]*len(IMG)
        else:
            use_name = name

        if all(type(v) != list for v in kwargs.values()):
            use_kwargs = [kwargs]*len(IMG)
        else:
            use_kwargs = []
            for i in range(len(IMG)):
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
        imagedata = list(zip(IMG, use_pixscale, use_saveto,
                             use_name, use_kwargs))
        if n_procs > 1:
            with Pool(n_procs) as pool:
                res = pool.starmap(self.Process_Image,
                                   imagedata,
                                   chunksize = 5 if len(IMG) > 100 else 1)
        else:
            res = list(starmap(self.Process_Image, imagedata))
            
        # Report completed processing, and track time used
        logging.info('All Images Finished Processing at %.1f' % (time() - start))
        timers = dict((s,0) for s in self.pipeline_steps)
        count_success = 0.
        for r in res:
            if r == 1:
                continue
            count_success += 1.
            for s in self.pipeline_steps:
                timers[s] += r[s]
        for s in self.pipeline_steps:
            timers[s] /= count_success
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
            use_config = config_file[startat:] # config_file[:-3] if config_file[-3:] == '.py' else config_file
        if '/' in config_file:
            sys.path.append(config_file[:config_file.rfind('/')])
        try:
            c = importlib.import_module(use_config)
        except:
            sys.path.append(os.getcwd())
            c = importlib.import_module(use_config)

        if 'forced' in c.process_mode:
            self.UpdatePipeline(new_pipeline_steps = ['background', 'psf', 'center forced', 'isophoteinit', 'isophotefit forced',
                                                      'starmask forced', 'isophoteextract forced', 'checkfit', 'radsample', 'ellipsemodel'])
            
        try:
            self.UpdatePipeline(new_pipeline_functions = c.new_pipeline_functions)
        except:
            pass
        try:
            self.UpdatePipeline(new_pipeline_steps = c.new_pipeline_steps)
        except:
            pass
        try:
            self.UpdatePipeline(preprocess = c.preprocess)
        except:
            pass
            
        use_kwargs = GetKwargs(c)
            
        if c.process_mode in ['image', 'forced image']:
            return self.Process_Image(IMG = c.image_file, pixscale = c.pixscale, **use_kwargs)
        elif c.process_mode in ['image list', 'forced image list']:
            return self.Process_List(IMG = c.image_file, pixscale = c.pixscale, **use_kwargs)
        else:
            logging.error('Unrecognized process_mode! Should be in: [image, image list, forced image, forced image list]')
            return 1
        
