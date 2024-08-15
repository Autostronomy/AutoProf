ap_process_mode = "image list"

ap_n_procs = 4
ap_image_file = [
    "ESO479-G1_r.fits",
    "ESO479-G1_r.fits",
    "ESO479-G1_r.fits",
    "ESO479-G1_r.fits",
]
ap_pixscale = 0.262
ap_name = ["testbatchimage1", "testbatchimage2", "testbatchimage3", "testbatchimage4"]
ap_doplot = True

# In batch processing a single option value will be used for all images, while a list can give custom values for each item in the batch.
# In this case, the first two runs in the batch perform a normal fit while the 3rd and 4th fit ellipses with Fourier mode perturbations.
ap_isofit_fitcoefs = [None, None, (1,), (1, 3, 4)]
# In this case the 1st and 2nd runs will measure the Fourier mode amplitudes along the ellipses, but the 3rd and 4th wont.
ap_iso_measurecoefs = [(1,), (1, 3, 4), None, None]
