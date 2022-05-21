import os
import time
import pickle
import numpy as np
from astropy.io import ascii
from astropy.table import Table, vstack
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid, quad
from scipy.special import erf
import multiprocessing
import matplotlib.pyplot as plt

# initial masss function
def imf(Mini, alpha=-2.35):
	return Mini**alpha

# magnitude of composite flux
def addmag(mag1, mag2):
	return mag1-2.5*np.log10(1+10**((mag1-mag2)/2.5))

# calculate the likelihood for a detection or non-detection in a filter
def like1(synmag, obsmag, err, det):
	if det==1: LL=1.0/np.sqrt(2*np.pi)/err*np.exp(-0.5*((synmag-obsmag)/err)**2) # detection
	elif det==0: LL=0.5+0.5*erf((synmag-obsmag)/np.sqrt(2)/err) # non-detection
	else: LL=1.0 # lacks observation
	return LL

# prepare the stellar isochrones
def get_isochrones(iso, LAs, bands=None, DM=None, gal_Av=None, gal_coeff=None, magcuts=None):
#{{{#
	pts=[]
	interps=[]
	for iLA in range(len(LAs)):

		# select isochrone for this age
		pt0=iso[np.abs(iso['logAge']-LAs[iLA])<1e-4][['Mini']+[bd+'mag' for bd in bands]]
		if len(pt0)==0:
			print('isochrone of log(t/yr) =', LAs[iLA], 'is not provided in the isochrone table')
			return -1

		# add galactic extinction and distance modulus to the absolute magnitudes
		for bd in bands: pt0[bd+'mag']=np.around(pt0[bd+'mag']+gal_coeff[bd]*gal_Av+DM, 3)

		# mass-mag interpolation for this age
		interp0={}
		for bd in bands:
			interp0[bd]=interp1d(pt0['Mini'], pt0[bd+'mag'], fill_value=99.999, bounds_error=False)
		interps.append(interp0)

		# cut the lower isochrone that is too faint
		Mlow=999.0
		for bd in bands:
			Mlow_tmp=np.amin(pt0[pt0[bd+'mag']<magcuts[bd]]['Mini'])
			if Mlow_tmp<Mlow: Mlow=Mlow_tmp
		pt0=pt0[pt0['Mini']>=Mlow]

		pts.append(pt0)
#}}}#
	return pts, interps

# get the maximum stellar mass and AA, BB, CC for the stellar isochrones
def get_ABC_maxMs(pts, alpha=-2.35, minStellarMass=0.08, maxStellarMass=300.0):
#{{{#
	maxMs=[np.amax(pt0['Mini']) for pt0 in pts]

	CC0, err_CC0 = quad(imf, minStellarMass, maxStellarMass, args=(alpha, ))

	AAs=[]
	BBs=[]
	for maxM0 in maxMs:
		AA, err_AA = quad(imf, minStellarMass, maxM0, args=(alpha, ))
		AAs.append(AA)
		BBs.append(CC0-AA)
#}}}#
	return AAs, BBs, CC0, maxMs

# interpolate the stellar isochrone to increase sampling in Mini
def interp_isochrone(pt0, interp0, bands=None, obs=None, dmag_min=0.02):
#{{{#
	# minimum sampling
	dmag={bd: np.maximum(dmag_min, obs[bd+'err']) for bd in bands}
	# new data points
	Ms_new=[]
	for iM in range(len(pt0)-1):
		npts=1 # number of new data points (+1)
		for bd in bands:
			delta_mag=np.abs(pt0[iM][bd+'mag']-pt0[iM+1][bd+'mag'])
			npts_tmp=int(np.ceil(delta_mag/dmag[bd]))
			if npts_tmp>npts: npts=npts_tmp
		if npts>1:
			# too sparsely sampled
			dmass=(pt0[iM+1]['Mini']-pt0[iM]['Mini'])/npts
			Ms_new.append(pt0[iM]['Mini']+np.arange(1, npts)*dmass)
	# increase sampling?
	if len(Ms_new)>0:
		# add new data points with linear interpolation
		Ms_new=np.hstack(Ms_new)
		pt0_new=Table([Ms_new], names=['Mini'])
		for bd in bands: pt0_new[bd+'mag']=np.around(interp0[bd](Ms_new), 3)
		newpt0=vstack([pt0, pt0_new])
		newpt0=newpt0[np.argsort(newpt0['Mini'])]
	else:
		newpt0=pt0.copy()
#}}}#
	return newpt0

# get the important parameter vectors and the magnitude vector/array
def get_MST_synmags(pt0, bands, maxM0, AA0, BB0, CC0, Pbin, alpha, minStellarMass):
#{{{#
	# get vectors for the initial masses and the S and T parameters
	MMvec = np.array(pt0['Mini'])
	SSvec = (1-Pbin)*(MMvec**alpha)/AA0 + Pbin*BB0/CC0/(maxM0-minStellarMass)
	TTvec = Pbin*(MMvec**alpha)/CC0/(MMvec-minStellarMass)

	# get synthetic magnitudes for single and binary stars
	synmags_sin={bd: np.array(pt0[bd+'mag']) for bd in bands}
	synmags_bin={bd: addmag(*np.meshgrid(synmags_sin[bd], synmags_sin[bd], sparse=True)) for bd in bands}
#}}}#
	res={'MMvec': MMvec, 'SSvec': SSvec, 'TTvec': TTvec,
		'synmags_sin': synmags_sin, 'synmags_bin': synmags_bin}
	return res

# calculate the marginalized likelihood for a given age and extinction
def get_LH(obs, bands, int_coeff, Av0, MMvec, SSvec, TTvec, synmags_sin, synmags_bin):
#{{{#
	# magnitude check
	for bd in bands:
		if obs[bd+'det']==1:
			mag_tmp=obs[bd+'mag']-int_coeff[bd]*Av0
			err_tmp=obs[bd+'err']
			if mag_tmp+5.0*err_tmp<np.amin(synmags_sin[bd])-2.5*np.log10(2):
				# this source is too bright than any possible magnitudes
				return 0.0

	# color check
	for ibd1 in range(len(bands)-1):
		bd1=bands[ibd1]
		mag1=obs[bd1+'mag']-int_coeff[bd1]*Av0
		err1=obs[bd1+'err']
		for ibd2 in range(ibd1+1, len(bands)):
			bd2=bands[ibd2]
			mag2=obs[bd2+'mag']-int_coeff[bd2]*Av0
			err2=obs[bd2+'err']
			if obs[bd1+'det']==1:
				if mag1-mag2+5.0*np.sqrt(err1**2+err2**2)<np.amin(synmags_sin[bd1]-synmags_sin[bd2]):
					# this source is too blue than any possible colors
					return 0.0
			if obs[bd2+'det']==1:
				if mag1-mag2-5.0*np.sqrt(err1**2+err2**2)>np.amax(synmags_sin[bd1]-synmags_sin[bd2]):
					# this source is too red than any possible colors
					return 0.0

	# calculate likelihood
	LH_sin=1.0
	LH_bin=1.0
	for bd in bands:
		LH_sin = LH_sin*like1(synmags_sin[bd],
			obs[bd+'mag']-int_coeff[bd]*Av0, obs[bd+'err'], obs[bd+'det'])
		LH_bin = LH_bin*like1(synmags_bin[bd],
			obs[bd+'mag']-int_coeff[bd]*Av0, obs[bd+'err'], obs[bd+'det'])
	LH_tot = trapezoid(SSvec*LH_sin+TTvec*0.5*trapezoid(LH_bin, x=MMvec, axis=1), x=MMvec)
#}}}#
	return LH_tot

def calc_LH_array(obs, LAs, Avs, pts, interps, AAs, BBs, CC0, maxMs, bands=None, int_coeff=None,
	Pbin=0.5, alpha=-2.35, minStellarMass=0.08, num_core=1, dmag_min=0.02):
#{{{#
	# isochrone interpolation
	newpts=[interp_isochrone(pts[iLA], interps[iLA], bands=bands, obs=obs,
		dmag_min=dmag_min) for iLA in range(len(LAs))]

	# calculate the parameter vectors and synthetic magnitudes
	pool=multiprocessing.Pool(num_core)
	res = pool.starmap(get_MST_synmags, [(newpts[iLA], bands, maxMs[iLA], AAs[iLA], BBs[iLA], CC0,
		Pbin, alpha, minStellarMass) for iLA in range(len(LAs))])
	pool.close()
	pool.join()

	# calculate likelihood
	pool=multiprocessing.Pool(num_core)
	LHs = pool.starmap(get_LH, [(obs, bands, int_coeff, Avs[iAv],
		res[iLA]['MMvec'], res[iLA]['SSvec'], res[iLA]['TTvec'],
		res[iLA]['synmags_sin'], res[iLA]['synmags_bin'])
			for iLA in range(len(LAs)) for iAv in range(len(Avs))]) # LHs[iLA, iAv]
	pool.close()
	pool.join()

	# convert the result into a numpy array
	LHs=np.array(LHs, dtype=np.float32).reshape(len(LAs), len(Avs))
#}}}#
	return LHs

def starfit(cat_filename=None, iso_filename=None, index=None, bands=None,
	logAge_min=6.60, logAge_max=7.60, logAge_bin=0.01,
	intAv_min=0.0, intAv_max=2.0, intAv_bin=0.02,
	DM=None, gal_Av=None, gal_coeff=None, int_coeff=None,
	Pbin=0.5, alpha=-2.35, minStellarMass=0.08, maxStellarMass=300.0,
	magcuts=None, dmag_min=0.02, num_core=1, path='./', overwrite=False):

	# check arguments and keywords{{{
	if (cat_filename is None): print('cat_filename not provided')
	if (iso_filename is None): print('iso_filename not provided')
	if (bands is None): print('bands not provided')
	if (DM is None): print('DM not provided')
	if (gal_Av is None) : print('gal_Av not provided')
	if (gal_coeff is None): print('gal_coeff not provided')
	if (int_coeff is None) : print('int_coeff not provided')
	if (magcuts is None): print('magcuts not provided')
	# check magnitude cuts
	for bd in bands:
		if bd not in magcuts.keys():
			print('magcuts not provided')
			return -1
	# check extinction coefficients
	for bd in bands:
		if bd not in gal_coeff.keys():
			print('gal_coeff not provided')
			return -1
		if bd not in int_coeff.keys():
			print('int_coeff not provided')
			return -1
	# check arguments and keywords}}}

	# read and check stellar catalog{{{
	cat=ascii.read(cat_filename)
	for bd in bands:
		if bd+'mag' not in cat.colnames:
			print(); print(bd+'mag is not provided in the catalog')
			return -1
		if bd+'err' not in cat.colnames:
			print(); print(bd+'err is not provided in the catalog')
			return -1
		if bd+'det' not in cat.colnames:
			print(); print(bd+'det is not provided in the catalog')
			return -1
	# read and check stellar catalog}}}

	# read and check isochrone table{{{
	iso=ascii.read(iso_filename)
	if 'logAge' not in iso.colnames:
		print('logAge is not provided in the isochrone table')
		return -1
	if 'Mini' not in iso.colnames:
		print('Mini is not provided in the isochrone table')
		return -1
	for bd in bands:
		if bd+'mag' not in iso.colnames:
			print(bd+'mag is not provided in the isochrone table')
			return -1
	# read and check isochrone table}}}

	if path[-1]!='/': path=path+'/'
	if os.path.exists(path)==False: os.system('mkdir '+path)

	LAs=np.around(np.arange(logAge_min, logAge_max, logAge_bin), 3)
	Avs=np.around(np.arange(intAv_min, intAv_max, intAv_bin), 3)

	# print out information{{{
	print()
	print('Basic parameters:')
	print('----------------')
	print('Distance modulus:', DM)
	print('Galactic extinction:', gal_Av)
	print('Binary fraction:', Pbin)
	print('IMF power-law index:', alpha)
	print('Minimum stellar mass:', minStellarMass)
	print('Maximum stellar mass:', maxStellarMass)
	print()
	print('Extinction coefficients:')
	print('-----------------------')
	print('Galactic:', end='')
	for bd in bands: print('  '+bd+': '+str(gal_coeff[bd]), end='')
	print()
	print('Internal:', end='')
	for bd in bands: print('  '+bd+': '+str(int_coeff[bd]), end='')
	print()
	print()
	print('Settings:')
	print('--------')
	print('Magnitude cuts:', end='')
	for bd in bands: print('  '+bd+': '+str(magcuts[bd]), end='')
	print()
	print('Minimum magnitude resolution:', dmag_min)
	print('Number of cores:', num_core)
	print('Where to save the results:', path)
	print('Overwrite existing files:', overwrite)
	print()
	print('Grids:')
	print('-----')
	print('log(t/yr) =', LAs[0], LAs[1], LAs[2], '...', LAs[-2], LAs[-1], end=' ')
	print('('+str(len(LAs))+' ages in total)')
	print('Av =', Avs[0], Avs[1], Avs[2], '...', Avs[-2], Avs[-1], end=' ')
	print('('+str(len(Avs))+' extinctions in total)')
	print()
	# print out information}}}

	pts, interps = get_isochrones(iso, LAs, bands=bands,
		DM=DM, gal_Av=gal_Av, gal_coeff=gal_coeff, magcuts=magcuts)
	AAs, BBs, CC0, maxMs = get_ABC_maxMs(pts, alpha=alpha,
		minStellarMass=minStellarMass, maxStellarMass=maxStellarMass)

	# perform star fit{{{
	ndigit=int(np.ceil(np.log10(len(cat))))
	print('>>> Calculating the likelihood L(D|t,Av) for the individual stars ...'); print()
	print(('%'+str(2*ndigit+1)+'s')%'ID', end=' ')
	for bd in bands: print('%14s'%bd, end='')
	print('     Time')
	print('-'*(2*ndigit+1), end=' ')
	for bd in bands: print('  '+'-'*12, end='')
	print('  -------')
	if index is None: index=range(len(cat))
	for ii in index:
		print(('%'+str(ndigit)+'i')%ii+'/'+str(len(cat)), end='  ');
		if ii>=len(cat) or ii<0:
			print('not found')
			continue
		resfilename=path+str(ii)+'.dat'
		if (overwrite==False) and os.path.exists(resfilename):
			print('file already exists')
			continue
		for bd in bands:
			if cat[ii][bd+'det']==1:
				print(' '+'%6.3f'%cat[ii][bd+'mag']+'±%5.3f'%cat[ii][bd+'err'], end=' ')
			elif cat[ii][bd+'det']==0:
				print('>%6.3f'%cat[ii][bd+'mag']+'±%5.3f'%cat[ii][bd+'err'], end=' ')
			else:
				print('%13s'%'-', end=' ')
		time1=time.time()
		# calculation
		LHs=calc_LH_array(cat[ii], LAs, Avs, pts, interps, AAs, BBs, CC0, maxMs,
			bands=bands, int_coeff=int_coeff,
			Pbin=Pbin, alpha=alpha, minStellarMass=minStellarMass,
			num_core=num_core, dmag_min=dmag_min)
		# save results
		with open(resfilename, 'wb') as ff:
			pickle.dump(LHs, ff)
		# plot results
		fig=plt.figure(figsize=[3.5, 3])
		ax=fig.add_subplot(111)
		ax.set_title('Star '+str(ii))
		ax.set_xlabel('log(t/yr)')
		ax.set_ylabel('Av (mag)')
		p=ax.imshow(np.transpose(LHs),
			extent=[np.amin(LAs), np.amax(LAs), np.amin(Avs), np.amax(Avs)],
			origin='lower', cmap='rainbow', aspect='auto')
		fig.colorbar(p, ax=ax, aspect=30)
		fig.tight_layout()
		fig.savefig(resfilename.replace('.dat', '.pdf'))
		plt.close(fig)
		print(' %7.1f'%(time.time()-time1))
	print(); print('>>> Complete!')
	# perform star fit}}}

# End of file #
