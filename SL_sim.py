#general imports
import numpy as np
import random as rand
import glob
from astropy.io import fits
import math
import scipy
import scipy.ndimage
from astropy.nddata.utils import block_reduce
from astropy.nddata.utils import block_replicate
from photutils import aperture_photometry
from photutils import EllipticalAperture
import random
import matplotlib.pyplot as plt
import time
import pandas as pd 
#import psfex
from scipy import signal
import sys
from importlib import reload
import re
from astropy.visualization import lupton_rgb
from astropy.visualization.lupton_rgb import AsinhZScaleMapping
#import DES_mod as DES
from math import isnan
from scipy import interpolate
import os
from scipy.interpolate import griddata
#lenstronomy imports
from lenstronomy.Data.psf import PSF
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import lenstronomy.Util.param_util as param_util
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.image_util as image_util
import lenstronomy.Util.util as util
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet
import lenstronomy.Util.class_creator as class_creator
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.Cosmo.lens_cosmo import LensCosmo

#NEW imports
from lenstronomy.SimulationAPI.sim_api import SimAPI
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Plots import chain_plot
import lenstronomy.Util.constants as const
from astropy.table import Table
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver


def ap_phot(xcen,ycen,image,e1,e2,theta,zp):
    apertures = EllipticalAperture([(xcen,ycen)], e1, e2,theta)
    phot_table = aperture_photometry(image, apertures)
    phot_table['aperture_sum'].info.format = '%.8g' 
    mag = -2.5*np.log10(phot_table['aperture_sum'][0])+zp
    return phot_table['aperture_sum'][0],mag

def ap_phot_plot(xcen,ycen,image,e1,e2,theta,zp):
    apertures = EllipticalAperture([(xcen,ycen)], e1, e2,theta)
    phot_table = aperture_photometry(image, apertures)
    phot_table['aperture_sum'].info.format = '%.8g'
    fig1 = plt.figure()
    apertures.plot(color='white', lw=2)
    plt.imshow(image)
    plt.show()
    plt.close()
    mag = -2.5*np.log10(phot_table['aperture_sum'][0])+zp
    return phot_table['aperture_sum'][0],mag



def background_rms_image(cb,image):
    xg,yg = np.shape(image)
    cut0  = image[0:cb,0:cb]
    cut1  = image[xg-cb:xg,0:cb]
    cut2  = image[0:cb,yg-cb:yg]
    cut3  = image[xg-cb:xg,yg-cb:yg]
    l = [cut0,cut1,cut2,cut3]
    m=np.mean(np.mean(l,axis=1),axis=1)
    ml=min(m)
    mm=max(m)
    if mm > 2*ml:
        s=np.sort(l,axis=0)
        nl=s[:-1]
        std = np.std(nl)
        rms = np.sqrt(np.mean(np.asarray([nl])**2))
    else:
        std = np.std([cut0,cut1,cut2,cut3])
        rms = np.sqrt(np.mean(np.asarray([cut0,cut1,cut2,cut3])**2))
    return std


def make_lensmodel(lens_info,theta_E,source_info,box_f): 
	# lens data specifics
	lens_image = lens_info['image']
	psf_lens = lens_info['psf']
	background_rms = background_rms_image(5,lens_image)
	exposure_time = 100
	kwargs_data_lens = sim_util.data_configure_simple(len(lens_image), lens_info['deltapix'],exposure_time, background_rms)
	kwargs_data_lens['image_data'] = lens_image
	data_class_lens = ImageData(**kwargs_data_lens)
	#PSF
	kwargs_psf_lens = {'psf_type': 'PIXEL', 'pixel_size': lens_info['deltapix'], 'kernel_point_source': psf_lens}
	psf_class_lens = PSF(**kwargs_psf_lens)
	# lens light model
	lens_light_model_list = ['SERSIC_ELLIPSE']
	lens_light_model_class = LightModel(light_model_list=lens_light_model_list)
	kwargs_model = {'lens_light_model_list': lens_light_model_list}
	kwargs_numerics_galfit = {'supersampling_factor': 1}
	kwargs_constraints = {}
	kwargs_likelihood = {'check_bounds': True}
	image_band = [kwargs_data_lens, kwargs_psf_lens,kwargs_numerics_galfit]
	multi_band_list = [image_band]
	kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}
	# Sersic component
	fixed_lens_light = [{}]
	kwargs_lens_light_init=[{'R_sersic': .1, 'n_sersic': 4, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]
	kwargs_lens_light_sigma=[{'n_sersic': 0.5, 'R_sersic': 0.2, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1}]
	kwargs_lower_lens_light=[{'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.01, 'n_sersic': 0.5, 'center_x': -10, 'center_y': -10}]
	kwargs_upper_lens_light=[{'e1': 0.5, 'e2': 0.5, 'R_sersic': 10, 'n_sersic': 8, 'center_x': 10, 'center_y': 10}]
	lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, fixed_lens_light, kwargs_lower_lens_light, kwargs_upper_lens_light]
	kwargs_params = {'lens_light_model': lens_light_params}
	fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model, kwargs_constraints, kwargs_likelihood, kwargs_params)
	fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 50, 'n_iterations': 50}]]
	chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
	kwargs_result = fitting_seq.best_fit()
	modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result)
	# Lens light best result
	kwargs_light_lens = kwargs_result['kwargs_lens_light'][0]
	#Lens model
	kwargs_lens_list = [{'theta_E': theta_E, 'e1': kwargs_light_lens['e1'], 'e2':kwargs_light_lens['e2'],'center_x': kwargs_light_lens['center_x'], 'center_y':kwargs_light_lens['center_y']}]
	lensModel = LensModel(['SIE'])
	lme = LensModelExtensions(lensModel)
	#random position for the source
	x_crit_list, y_crit_list = lme.critical_curve_tiling(kwargs_lens_list, compute_window=(len(source_info['image']))* (source_info['deltapix']),start_scale=source_info['deltapix'], max_order=10)
	if len(x_crit_list)>2 and len(y_crit_list)>2:
		x_caustic_list, y_caustic_list = lensModel.ray_shooting(x_crit_list, y_crit_list, kwargs_lens_list)	   
		xsamp0=np.arange(min(x_caustic_list)-min(x_caustic_list)*box_f[0],max(x_caustic_list)+max(x_caustic_list)*box_f[1],0.1)
		xsamp=xsamp0[abs(xsamp0.round(1))!=0.1]
		ysamp0=np.arange(min(y_caustic_list)-min(y_caustic_list)*box_f[0],max(y_caustic_list)+max(y_caustic_list)*box_f[1],0.1)
		ysamp=ysamp0[abs(ysamp0.round(1))!=0.1]
		if len(xsamp) == 0 or len(ysamp) == 0:
			x_shift,y_shift = 0.15,0.15	#arcseconds
		else:
			y_shift = rand.sample(list(ysamp),1)[0]
			x_shift = rand.sample(list(xsamp),1)[0]
	else:
		x_shift,y_shift = -0.15,0.15	#arcseconds
		x_caustic_list = [0]
		y_caustic_list = [0]
	solver = LensEquationSolver(lensModel)
	theta_ra, theta_dec = solver.image_position_from_source(x_shift,y_shift, kwargs_lens_list)
	if len(theta_ra) <= 1:
		x_shift,y_shift = -0.2,-0.2 #arcseconds1
	if abs(x_shift) >= int(theta_E) or abs(y_shift) >= int(theta_E):
		x_shift,y_shift = 0.3,-0.3
		print('BLABLA')
	print('HERE',min(x_caustic_list)-min(x_caustic_list)*box_f[0],max(x_caustic_list)+max(x_caustic_list)*box_f[1],min(y_caustic_list)-min(y_caustic_list)*box_f[0],max(y_caustic_list)+max(y_caustic_list)*box_f[1])
	return {'lens_light_model_list' : ['SERSIC_ELLIPSE'],'kwargs_light_lens':[kwargs_light_lens],'lens_light_model_class':lens_light_model_class,'kwargs_lens_list':kwargs_lens_list,'kwargs_data_lens':kwargs_data_lens,'source_shift':[x_shift,y_shift]}


def make_lens_sys(lensmodel,src_camera_kwargs,source_info,kwargs_band_src,lens_info):
	#Model
	kwargs_model_postit = {'lens_model_list': ['SIE'],'source_light_model_list': ['INTERPOL']} 
	kwargs_lens = lensmodel['kwargs_lens_list'] # SIE model
	
	#data
	numpix = len(source_info['image'])*source_info['HR_factor']
	kwargs_source_mag = [{'magnitude': source_info['magnitude'], 'image': source_info['image'], 'scale': source_info['deltapix']/source_info['HR_factor'], 'phi_G': 0, 'center_x': lensmodel['source_shift'][0], 'center_y': lensmodel['source_shift'][1]}] #phi_G is to rotate, centers are to shift in arcsecs 
	sim = SimAPI(numpix=numpix,kwargs_single_band=kwargs_band_src, kwargs_model=kwargs_model_postit)
	kwargs_numerics = {'supersampling_factor': source_info['HR_factor']}
	imSim = sim.image_model_class(kwargs_numerics)
	_,kwargs_source,_ =sim.magnitude2amplitude(kwargs_source_mag=kwargs_source_mag)
	
	#simulation
	image_HD = imSim.image(kwargs_lens=kwargs_lens,kwargs_source=kwargs_source)

	mag_LS = -2.5*np.log10(sum(sum(image_HD))) + source_info['zero_point']
	magnification = source_info['magnitude']-mag_LS #rough estimation in the magnitude change
	
	#DES - PSF convolution
	npsf=inter_psf(lens_info['psf'],lens_info['deltapix'],source_info['deltapix'])
	source_conv = signal.fftconvolve(image_HD,npsf,mode='same') 

	#resize in deltapix
	source_lensed_res = block_reduce(source_conv, source_info['HR_factor']) #in case an HR_factor was used
	eq_pa = int(lens_info['deltapix']*len(lens_info['image'])*len(source_lensed_res)/(source_info['deltapix']*len(source_info['image']))) #this value is to consider a source image of the same size in arcseconds as the lens 
	bs=np.zeros([eq_pa,eq_pa])
	val=int((len(bs)-len(source_lensed_res))/2)
	bs[val:val+len(source_lensed_res),val:val+len(source_lensed_res)]=source_lensed_res
	source_size_scale=block_reduce(bs, int(len(bs)/len(lens_info['image'])))

	#flux rescale
	flux_img=sum(image_HD.flatten())*(10**(0.4*(lens_info['zero_point']-source_info['zero_point'])))
	sc=sum(source_conv.flatten())/flux_img
	source_scaled =source_size_scale/sc

	#cut right pixels size
	lens_size   = min(np.shape(lens_info['image']))
	source_size = min(np.shape(source_scaled))


	if lens_size > source_size:
		xin = yin = int((lens_size - source_size)/2) 
		lens_final = lens_info['image'][xin:xin+ source_size,yin:yin+source_size]
		source_final = source_scaled
	
	if lens_size < source_size:
		xin = yin = math.ceil((source_size - lens_size)/2)
		source_final = source_scaled[xin:xin+ lens_size,yin:yin+lens_size]
		lens_final = lens_info['image']
	else:
		lens_final = lens_info['image']
		source_final = source_scaled
	final = lens_final + source_final

	phot_ap = 2
	_,mag_sim=ap_phot(len(final)/2,len(final)/2,final,phot_ap/(lens_info['deltapix']),phot_ap/(lens_info['deltapix']),np.pi/2.,lens_info['zero_point'])

	#in the old code some images cause problems with some inf pixels... this will discard that system	
	if magnification == np.float('-inf') or magnification==np.float('inf'):
		print('INFINITE MAGNIFICATION')
		magnification = 10000

	return {'simulation':final,'src_image':source_info['image'],'mag_sim':mag_sim,'mag_lensed_src':mag_LS,'image_HD':image_HD,'resize':source_size_scale,'conv':source_conv,'magnification':magnification}


def save_sim_mband(name_var,dictionary,image_list,path,name): 
	i,j,k = 0,0,0
	hdr = fits.Header()
	new_hdul = fits.HDUList()
	for i in range(len(name_var)):
		hdr[name_var[i]] = dictionary[name_var[i]]
	for j in range(len(image_list)):
		new_hdul.append(fits.ImageHDU(image_list[j],header=hdr))
	new_hdul.writeto(str(path)+str(name)+'.fits',overwrite=True)



def scale_val(image_array):
    if len(np.shape(image_array)) == 2:
        image_array = [image_array]
    vmin = np.min([background_rms_image(5, image_array[i]) for i in range(len(image_array))])
    xl, yl = np.shape(image_array[0])
    box_size = 14  # in pixel
    xmin = int((xl) / 2 - (box_size / 2))
    xmax = int((xl) / 2 + (box_size / 2))
    vmax = np.max([image_array[i][xmin:xmax, xmin:xmax] for i in range(len(image_array))])
    return vmin, vmax

def showplot_rgb(rimage, gimage, bimage):
    vmin, vmax = scale_val([rimage, gimage, bimage])
    img = np.zeros((rimage.shape[0], rimage.shape[1], 3), dtype=float)
    img[:, :, 0] = sqrt_sc(rimage, scale_min=vmin, scale_max=vmax)
    img[:, :, 1] = sqrt_sc(gimage, scale_min=vmin, scale_max=vmax)
    img[:, :, 2] = sqrt_sc(bimage, scale_min=vmin, scale_max=vmax)
    return img


def background_rms_image(cb,image):
    xg,yg = np.shape(image)
    cut0  = image[0:cb,0:cb]
    cut1  = image[xg-cb:xg,0:cb]
    cut2  = image[0:cb,yg-cb:yg]
    cut3  = image[xg-cb:xg,yg-cb:yg]
    l = [cut0,cut1,cut2,cut3]
    m=np.mean(np.mean(l,axis=1),axis=1)
    ml=min(m)
    mm=max(m)
    if mm > 5*ml:
        s=np.sort(l,axis=0)
        nl=s[:-1]
        std = np.std(nl)
    else:
        std = np.std([cut0,cut1,cut2,cut3])
    return std

def sqrt_sc(inputArray, scale_min=None, scale_max=None):
    #this definition was taken from lenstronomy
    imageData = np.array(inputArray, copy=True)
    if scale_min is None:
        scale_min = imageData.min()
    if scale_max is None:
        scale_max = imageData.max()
    imageData = imageData.clip(min=scale_min, max=scale_max)
    imageData = imageData - scale_min
    indices = np.where(imageData < 0)
    imageData[indices] = 0.00001
    imageData = np.sqrt(imageData)
    imageData = imageData / np.sqrt(scale_max - scale_min)
    return imageData


def inter_psf(psf,old_px,new_px):
	Xnew=Ynew = np.linspace(-len(psf)/2,len(psf)/2,len(psf)*(old_px/new_px))
	xnew,ynew = np.meshgrid(Xnew,Ynew)
	X = Y = np.linspace(-len(psf)/2,len(psf)/2,len(psf))
	x,y = np.meshgrid(X,Y)
	p0=np.asarray([x,y]).T
	nx,ny,nz=np.shape(p0)
	p=p0.reshape(nx*ny,2)
	val=psf.reshape(nx*ny)
	grid_z0 = griddata(p, val, (xnew, ynew), method='linear')
	return grid_z0


