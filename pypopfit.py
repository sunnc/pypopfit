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
def get_isochrones(iso, LAs, bands=None, DM=None, gal_Av=None, gal_coeff=None):
#{{{#
	pts=[]
	interps=[]
	for iLA in range(len(LAs)):

		# select isochrone for this age
		pt0=iso[np.abs(iso['logAge']-LAs[iLA])<1e-4][['logAge', 'Mini']+[bd+'mag' for bd in bands]]
		if len(pt0)==0:
			print('isochrone of log(t/yr) =', LAs[iLA], 'is not provided in the isochrone table')
			return -1
		for bd in bands: pt0[bd+'mag']=np.around(pt0[bd+'mag']+gal_coeff[bd]*gal_Av+DM, 3)
		pts.append(pt0)

		# mass-mag interpolation for this age
		interp0={}
		for bd in bands:
			interp0[bd]=interp1d(pt0['Mini'], pt0[bd+'mag'], fill_value=99.999, bounds_error=False)
		interps.append(interp0)
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

# interpolate isochrone
def interp_isochrone(pt0, interp0, bands=None, dmags=None):
#{{{#
	Ms_new=[]
	for iM in range(len(pt0)-1):
		npts=1 # number of new data points (+1)
		for bd in bands:
			delta_mag=np.abs(pt0[iM][bd+'mag']-pt0[iM+1][bd+'mag'])
			npts_tmp=int(np.ceil(delta_mag/dmags[bd]))
			if npts_tmp>npts: npts=npts_tmp
		if npts>1:
			# too sparsely sampled
			dmass=(pt0[iM+1]['Mini']-pt0[iM]['Mini'])/npts
			Ms_new.append(pt0[iM]['Mini']+np.arange(1, npts)*dmass)
	# increase sampling?
	if len(Ms_new)>0:
		# add new data points with linear interpolation
		Ms_new=np.hstack(Ms_new)
		newpt0=Table([Ms_new], names=['Mini'])
		for bd in bands: newpt0[bd+'mag']=np.around(interp0[bd](Ms_new), 3)
		newpt0=vstack([pt0, newpt0])
		newpt0=newpt0[np.argsort(newpt0['Mini'])]
	else:
		newpt0=pt0
#}}}#
	return newpt0

# cut and interpolate the stellar isochrone
def cut_interp_isochrone(pt0, interp0, bands=None, obs=None, dmag_min=0.02):
#{{{#
	# divide the isochrone into the upper and lower branches
	Mlow=0.0
	for bd in bands:
		if obs[bd+'det']==1:
			idxGood=(pt0[bd+'mag']-2.5*np.log10(2)<=obs[bd+'mag']+5*obs[bd+'err'])
			if np.any(idxGood):
				Mlow_tmp=np.amin(pt0[idxGood]['Mini'])
				if Mlow_tmp>Mlow: Mlow=Mlow_tmp
			else:
				# this star is too bright for this isochrone
				return pt0[idxGood], pt0 # return a zero-length table
	pt_upper=pt0[pt0['Mini']>=Mlow] # upper branch
	pt_lower=pt0[pt0['Mini']<Mlow] # lower branch: single stars too faint to match the observations

	# minimum sampling
	dmags={bd: np.maximum(dmag_min, obs[bd+'err']) for bd in bands}

	# interpolate the upper branch
	newpt1=interp_isochrone(pt_upper, interp0, bands=bands, dmags=dmags)

	# interpolate the lower branch?
	Mlow=1000.0
	Ms=(pt_lower[1:]['Mini']+pt_lower[:-1]['Mini'])/2
	for bd in bands:
		binmags=addmag(np.amax(pt_upper[bd+'mag']), pt_lower[bd+'mag'])
		idxBad=(np.abs(binmags[1:]-binmags[:-1])>dmags[bd])
		if np.any(idxBad):
			Mlow_tmp=np.amin(Ms[idxBad])
			if Mlow_tmp<Mlow: Mlow=Mlow_tmp
	idx=(pt_lower['Mini']>=Mlow)
	if np.any(idx):
		newpt2=vstack([pt_lower[~idx],
			interp_isochrone(pt_lower[idx], interp0, bands=bands, dmags=dmags), newpt1])
	else:
		newpt2=vstack([pt_lower, newpt1])
#}}}#
	return newpt1, newpt2

# get the important parameter vectors and the magnitude vector/array
def get_ST_synmags(newpt1, newpt2, bands, maxM0, AA0, BB0, CC0,
	Pbin=0.5, alpha=-2.35, minStellarMass=0.08):
#{{{#
	# get vectors for the initial masses and the S and T parameters
	MMvec = np.array(newpt1['Mini'])
	SSvec = (1-Pbin)*(MMvec**alpha)/AA0 + Pbin*BB0/CC0/(maxM0-minStellarMass)
	TTvec = Pbin*(MMvec**alpha)/CC0/(MMvec-minStellarMass)

	# get synthetic magnitudes for single and binary stars
	synmags_sin={bd: np.array(newpt1[bd+'mag']) for bd in bands}
	synmags_bin={}
	for bd in bands:
		mass1, mass2 = np.meshgrid(np.array(newpt1['Mini']), np.array(newpt2['Mini']))
		synmags_bin[bd]=addmag(*np.meshgrid(np.array(newpt1[bd+'mag']),
			np.array(newpt2[bd+'mag']), sparse=True))
		synmags_bin[bd][mass2>mass1]=99.999
#}}}#
	return SSvec, TTvec, synmags_sin, synmags_bin

# calculate the marginalized likelihood for a given age and a range of Avs
def calc_LH_vector(pt0, interp0, Avs, int_coeff, bands, obs,
	dmag_min, maxM0, AA0, BB0, CC0, Pbin, alpha, minStellarMass):
#{{{#
	# cut and interpolate the stellar isochrone
	newpt1, newpt2 = cut_interp_isochrone(pt0, interp0,
		bands=bands, obs=obs, dmag_min=dmag_min)
	if len(newpt1)==0:
		# this star is too bright for this isochrone
		return np.zeros(len(Avs), dtype=np.float32)

	# get MST vectors and synthetic magnitudes
	SSvec, TTvec, synmags_sin, synmags_bin = \
		get_ST_synmags(newpt1, newpt2, bands, maxM0, AA0, BB0, CC0,
			Pbin=Pbin, alpha=alpha, minStellarMass=minStellarMass)

	# get minimum magnitudes
	minmags={bd: np.amin(pt0[bd+'mag'])-2.5*np.log10(2) for bd in bands}
	# get minimum and maximum colors
	mincols={}
	maxcols={}
	for ibd1 in range(len(bands)-1):
		bd1=bands[ibd1]
		for ibd2 in range(ibd1+1, len(bands)):
			bd2=bands[ibd2]
			mincols[bd1+'-'+bd2]=np.amin(pt0[bd1+'mag']-pt0[bd2+'mag'])
			maxcols[bd1+'-'+bd2]=np.amax(pt0[bd1+'mag']-pt0[bd2+'mag'])

	LH1s=np.zeros(len(Avs), dtype=np.float32)
	for iAv, Av0 in enumerate(Avs):

		# de-redden the observed magnitudes
		mag0s={bd: obs[bd+'mag']-int_coeff[bd]*Av0 for bd in bands}

		# magnitude check
		for bd in bands:
			if obs[bd+'det']==1:
				if mag0s[bd]+5.0*obs[bd+'err']<minmags[bd]:
					# this source is too bright than any possible magnitudes
					LH1s[iAv]=0.0
					continue

		# color check
		for ibd1 in range(len(bands)-1):
			bd1=bands[ibd1]
			for ibd2 in range(ibd1+1, len(bands)):
				bd2=bands[ibd2]
				c12=mag0s[bd1]-mag0s[bd2]
				ec12=np.sqrt(obs[bd1+'err']**2+obs[bd2+'err']**2)
				if obs[bd1+'det']==1:
					if c12+5.0*ec12<mincols[bd1+'-'+bd2]:
						# this source is too blue than any possible colors
						LH1s[iAv]=0.0
						continue
				if obs[bd2+'det']==1:
					if c12-5.0*ec12>maxcols[bd1+'-'+bd2]:
						# this source is too red than any possible colors
						LH1s[iAv]=0.0
						continue

		# calculate likelihood
		LH_sin=1.0
		LH_bin=1.0
		for bd in bands:
			LH_sin = LH_sin*like1(synmags_sin[bd], mag0s[bd], obs[bd+'err'], obs[bd+'det'])
			LH_bin = LH_bin*like1(synmags_bin[bd], mag0s[bd], obs[bd+'err'], obs[bd+'det'])
		# marginalize the likelihood over M1 and M2
		intdM2=trapezoid(LH_bin, x=np.array(newpt2['Mini']), axis=0)
		LH1s[iAv]=trapezoid(SSvec*LH_sin+TTvec*intdM2, x=np.array(newpt1['Mini']))
#}}}#
	return LH1s

def calc_LH_array(obs, LAs, Avs, pts, interps, AAs, BBs, CC0, maxMs, bands=None, int_coeff=None,
	Pbin=0.5, alpha=-2.35, minStellarMass=0.08, num_core=1, dmag_min=0.02):
#{{{#
	pool=multiprocessing.Pool(num_core)
	LHs=pool.starmap(calc_LH_vector, [(pts[iLA], interps[iLA], Avs, int_coeff, bands, obs,
		dmag_min, maxMs[iLA], AAs[iLA], BBs[iLA], CC0,
		Pbin, alpha, minStellarMass) for iLA in range(len(LAs))])
	pool.close()
	pool.join()
	LHs=np.vstack(LHs)
#}}}#
	return LHs

def starfit(cat_filename=None, iso_filename=None, index=None, bands=None,
	logAge_min=6.60, logAge_max=7.60, logAge_bin=0.01,
	intAv_min=0.0, intAv_max=2.0, intAv_bin=0.02,
	DM=None, gal_Av=None, gal_coeff=None, int_coeff=None,
	Pbin=0.5, alpha=-2.35, minStellarMass=0.08, maxStellarMass=300.0,
	dmag_min=0.02, num_core=1, path='./', overwrite=False):

	# check arguments and keywords{{{
	if (cat_filename is None): print('cat_filename not provided')
	if (iso_filename is None): print('iso_filename not provided')
	if (bands is None): print('bands not provided')
	if (DM is None): print('DM not provided')
	if (gal_Av is None) : print('gal_Av not provided')
	if (gal_coeff is None): print('gal_coeff not provided')
	if (int_coeff is None) : print('int_coeff not provided')
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
		DM=DM, gal_Av=gal_Av, gal_coeff=gal_coeff)
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
		# save and plot results
		with open(resfilename, 'wb') as ff:
			pickle.dump(LHs, ff)
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
	# perform star fit}}}
	print(); print('>>> Complete!'); print()

# End of file #
