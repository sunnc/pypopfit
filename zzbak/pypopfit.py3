import numpy as np
from astropy.io import ascii
from astropy.table import Table, vstack
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid, quad
from scipy.special import erf

# initial mass function
def imf(Mini, alpha=-2.35):
	return Mini**alpha

def get_sin_isochrones(iso, bands, alpha=-2.35,
	logAge_min=6.60, logAge_max=8.00, logAge_bin=0.01, dmag=0.05):

	# get log-ages
	LAs=np.around(np.arange(logAge_min, logAge_max, logAge_bin), 3)
	print('single-star isochrones : log(t/yr) = ', end='')
	for iLA in range(len(LAs)): print(LAs[iLA], end=' ')
	print()

	# initialize
	minMs_sin=[] # minimum stellar mass: minMs_sin[iLA]
	maxMs_sin=[] # maximum stellar mass: maxMs_sin[iLA]
	AAs=[] # normalization factor for single-star IMF: AAs[iLA]
	interps_sin=[] # Mini-mag interploation in each band: interps_sin[iLA][bd]
	Ms=[] # initial mass vectors: Ms[iLA][:] as a function of mass
	Ms_alpha=[] # Mini**alpha vectors: Ms_alpha[iLA][:] as a function of mass
	synmags_sin=[] # magnitude vectors in each band: synmags_sin[iLA][bd][:] as a function of mass

	for iLA in range(len(LAs)):

		# select isochrone for this age
		pt0_sin=iso[np.abs(iso['logAge']-LAs[iLA])<1e-4]
		pt0_sin=pt0_sin[['Mini']+[bd+'mag' for bd in bands]]

		# maximum stellar mass and normalizationf actor for this age
		minM0_sin, maxM0_sin = np.amin(pt0_sin['Mini']), np.amax(pt0_sin['Mini'])
		minMs_sin.append(minM0_sin)
		maxMs_sin.append(maxM0_sin)
		nf0, err_nf0 = quad(imf, minM0_sin, maxM0_sin, args=(alpha, ))
		AAs.append(1.0/nf0)

		# Mini-mag interpolation for this age
		interp0_sin={}
		for bd in bands:
			interp0_sin[bd]=interp1d(pt0_sin['Mini'], pt0_sin[bd+'mag'],
				fill_value=99.999, bounds_error=False)
		interps_sin.append(interp0_sin)

		# increase sampling?
		Mnews=[]
		for iM in range(len(pt0_sin)-1):
			delta_mag=0.0 # maximum spacing in magnitude
			for bd in bands:
				tmp=np.abs(pt0_sin[iM][bd+'mag']-pt0_sin[iM+1][bd+'mag'])
				if tmp>delta_mag: delta_mag=tmp
			if delta_mag>dmag:
				# too sparsely sampled
				npts=int(np.ceil(delta_mag/dmag))
				dmass=(pt0_sin[iM+1]['Mini']-pt0_sin[iM]['Mini'])/npts
				Mnews.append(pt0_sin[iM]['Mini']+np.arange(1, npts)*dmass)
		if len(Mnews)>0:
			Mnews=np.hstack(Mnews)
			ptnew_sin=Table([Mnews], names=['Mini'])
			for bd in bands:
				ptnew_sin[bd+'mag']=interp0_sin[bd](Mnews)
			pt0_sin=vstack([pt0_sin, ptnew_sin])
			# sort the table in the order of increasing Mini
			pt0_sin=pt0_sin[np.argsort(pt0_sin['Mini'])]

		# initial mass and Mini**alpha vectors for this age
		Ms.append(np.array(pt0_sin['Mini'], dtype=np.float32))
		Ms_alpha.append(np.array(pt0_sin['Mini']**alpha, dtype=np.float32))

		# magnitude vectors in each band for this age
		synmag0s={}
		for bd in bands:
			synmag0s[bd]=np.array(pt0_sin[bd+'mag'], dtype=np.float32)
		synmags_sin.append(synmag0s)

	return LAs, minMs_sin, maxMs_sin, AAs, interps_sin, Ms, Ms_alpha, synmags_sin

def get_bin_isochrones(LAs, minMs_sin, maxMs_sin, interps_sin, Ms, synmags_sin,
	bands, alpha=-2.35, massRatio_bin=0.05, maxStellarMass=300.0):

	# magnitude of added flux
	def addmag(mag1, mag2):
		mag=mag1-2.5*np.log10(1+10**((mag1-mag2)/2.5))
		return mag

	# get mass ratios
	qqs=np.around(np.arange(0.0, 1.0+1e-3, massRatio_bin), 3)
	print('binary-star isochrones : q = M2/M1 = ', end='')
	for iqq in range(len(qqs)): print(qqs[iqq], end=' ')
	print()

	# initialize
	maxMs_bin=[] # maximum stellar mass vector for the primary star: maxMs_bin[iLA][:]
	BBs=[] # normalization factor vector for single-star IMF: BBs[iLA][:]
	synmags_bin=[] # magnitude arrays in each band: synmags_bin[iLA][bd][:, :]

	for iLA in range(len(LAs)):
		# maximum stellar mass and normalizationf actor for this age
		maxMs_bin_iLA=np.zeros(len(qqs), dtype=np.float32)
		BBs_iLA=np.zeros(len(qqs), dtype=np.float32)
		for iqq in range(len(qqs)):
			if qqs[iqq]<1e-5: maxM0_bin=maxStellarMass
			else: maxM0_bin=np.minimum(maxMs_sin[iLA]/qqs[iqq], maxStellarMass)
			nf0, err_nf0 = quad(imf, minMs_sin[iLA], maxM0_bin, args=(alpha, ))
			maxMs_bin_iLA[iqq]=maxM0_bin
			BBs_iLA[iqq]=1.0/nf0
		maxMs_bin.append(maxMs_bin_iLA)
		BBs.append(BBs_iLA)

		# magnitude arrays in each band for this age
		synmags_bin_iLA={}
		for bd in bands:
			synmags_bin_iLA[bd]=np.zeros((len(qqs), len(Ms[iLA])), dtype=np.float32)
			for iqq in range(len(qqs)):
				M2s=Ms[iLA]*qqs[iqq] # initial mass of the secondary star
				synmag2s=interps_sin[iLA][bd](M2s) # points out of bounds are assigned with 99.999
				synmags_bin_iLA[bd][iqq, :]=np.around(addmag(synmags_sin[iLA][bd], synmag2s), 3)

	return qqs, maxMs_bin, BBs, synmags_bin

def like1(synmag, obsmag, err, det):
	if det==1:
		LL=1.0/np.sqrt(2*np.pi)/err*np.exp(-0.5*((synmag-obsmag)/err)**2) # detection
	elif det==0:
		LL=0.5+0.5*erf((synmag-obsmag)/np.sqrt(2)/err) # non-detection
	else:
		LL=1.0 # lacks observation
	return LL

def like_sin(tabF, ii, iLA, iAv,
	AAs, Ms, Ms_alpha, synmags_sin,
	coeff_gal, galAv, coeff_host, Avs, DM):

	LLs_sin=1.0
	for bd in bds:
		LLs_sin=LLs_sin*like1(synmags[iLA][bd],
			tabF[ii]['mag'+bd]-(coeff_gal[bd]*galAv+coeff_host[bd]*Avs[iAv]+DM),
			tabF[ii]['err'+bd],
			tabF[ii]['det'+bd])
	return AAs[iLA]*trapezoid(LLs_sin*Ms_alpha[iLA], Ms[iLA])

#def like_bin(tabF, ii,
#	iLA, AAs, pts_sin, Malphas,
#	coeff_gal, Av_gal, coeff_host, Av_host, DM):
#
#	LLs_sin=1.0
#	for bd in bds:
#		LLs_sin=LLs_sin*like1(pts_sin[iLA][bd+'mag'],
#			tabF[ii]['mag'+bd]-(coeff_gal[bd]*Av_gal+coeff_host[bd]*Av_host+DM),
#			tabF[ii]['err'+bd], tabF[ii]['det'+bd])
#	return AAs[iLA]*trapezoid(LLs_sin*Malphas[iLA], pts_sin[iLA]['Mini'])

################
# test session #
################

import time

coeff_gal={'F275W': 1.989, 'F336W': 1.633, 'F438W': 1.332,
	'F555W': 0.989, 'F625W': 0.866, 'F814W': 0.578}
coeff_host=coeff_gal.copy()
galAv=0.3

iso=ascii.read('zziso/iso1.dat', format='commented_header', header_start=13)
bands=['F275W', 'F336W', 'F438W', 'F555W', 'F625W', 'F814W']
DM=30.0

tabF=ascii.read('zzcat/cat.dat')
tabF=tabF[['Mini']+[bd+'mag' for bd in bands]]
for bd in bands:
	tabF[bd+'mag']=np.around(tabF[bd+'mag']+coeff_host[bd]*galAv+coeff_host[bd]*0.5+DM, 3)
	tabF[bd+'err']=0.1
	tabF[bd+'det']=1
	for ii in range(len(tabF)):
		if tabF[ii][bd+'mag']>28.0:
			tabF[ii][bd+'mag']=28.0
			tabF[ii][bd+'err']=0.2
			tabF[ii][bd+'det']=0

time1=time.time(); print()
LAs, minMs_sin, maxMs_sin, AAs, interps_sin, Ms, Ms_alpha, synmags_sin = get_sin_isochrones(iso,
	bands=bands, alpha=-2.35,
	logAge_min=6.6, logAge_max=8.0+1e-3, logAge_bin=0.1, dmag=0.05)
print(round(time.time()-time1, 2), 'seconds used')
for iLA in range(len(LAs)): print(len(Ms[iLA]))

time1=time.time(); print()
qqs, maxMs_bin, BBs, synmags_bin = get_bin_isochrones(LAs,
	minMs_sin, maxMs_sin, interps_sin, Ms, synmags_sin,
	bands, alpha=-2.35, massRatio_bin=0.05, maxStellarMass=300.0)
print(round(time.time()-time1, 2), 'seconds used')

Avs=np.around(np.arange(0.0, 4.0+1e-3, 0.02), 3)

##Av and LA grids
#xys=[]
#for iAv in range(len(Avs)):
#	for iLA in range(len(LAs)):
#		xys.append((iAv, iLA))


#import os
#import time
#import numpy as np
#import matplotlib.pyplot as plt
#from astropy.io import ascii
#from scipy.interpolate import interp1d
#from scipy import integrate
#from multiprocessing import Pool
#import pysunnc as ps
#
##extinction and age bins
#Avbin=0.20
#LAbin=0.02
#Avs=np.around(np.arange(0.0, 4.0+1e-3, Avbin), 2)
#LAs=np.around(np.arange(6.6, 7.5+1e-3, LAbin), 2)

#	#single-star fit
#	def sinworker(jj):
#		(iAv, iLA)=xys[jj]
#		obsmag={}
#		obserr={}
#		obsdet={}
#		for bd in bds:
#			obsmag[bd]=tabF[ii]['mag'+bd]-(cef31[bd]*3.1*galTab[ll]['ebv']+cef0[bd]*Avs[iAv]+galTab[ll]['DM'])
#			obserr[bd]=tabF[ii]['err'+bd]
#			obsdet[bd]=tabF[ii]['det'+bd]
#		def priorMini(Mini):
#			return imf(Mini)/Ntot[iLA]
#		def sinfunc(Mini):
#			LL=1.0
#			for bd in bds:
#				synmag=interp[bd][iLA](Mini)
#				LL=LL*like1(synmag, obsmag[bd], obserr[bd], obsdet[bd])
#			return LL*priorMini(Mini)
#		tmp=integrate.quad(sinfunc, a=minMini[iLA], b=maxMini[iLA], \
#			points=[5, 10, 20, 50, 100, maxMini[iLA]], full_output=1)
#		psin=tmp[0]
#		return psin
#
#	flag=0
#	if flag==1:
#		for ii in range(len(tabF)):
#			if ii>0: continue
#			time1=time.time()
#			filename=dr+snTab[ll]['name']+'/sin'+str(tabF[ii]['ID'])+'.dat'
#			if os.path.exists(filename): continue
#
#			pool=Pool(processes=51)
#			psintab=pool.map(sinworker, range(len(Avs)*len(LAs)))
#			pool.close()
#			pool.join()
#
#			num=0
#			postsin=np.zeros((len(Avs), len(LAs)), dtype=float)
#			for iAv in range(len(Avs)):
#				for iLA in range(len(LAs)):
#					postsin[iAv, iLA]=psintab[num]
#					num=num+1
#			evidence=np.sum(postsin)*Avbin*LAbin
#			postsin=postsin/evidence
#			ps.pdump(postsin, filename)
#
#			print(filename,
#				'single fit:', tabF[ii]['ID'], len(tabF),
#				'time:', round(time.time()-time1, 2))
#
#		print()
#		print('complete!')
#
#	#binary-star fitting
#	def binworker(jj):
#		(iAv, iLA)=xys[jj]
#		obsmag={}
#		obserr={}
#		obsdet={}
#		for bd in bds:
#			obsmag[bd]=tabF[ii]['mag'+bd]-(cef31[bd]*3.1*galTab[ll]['ebv']+cef0[bd]*Avs[iAv]+galTab[ll]['DM'])
#			obserr[bd]=tabF[ii]['err'+bd]
#			obsdet[bd]=tabF[ii]['det'+bd]
#		def priorMini(Mini):
#			return imf(Mini)/Ntot[iLA]
#		def binfunc(Mini):
#			Mpri=Mini
#			Apri=(Mpri<=maxMini[iLA] and Mpri>=minMini[iLA])
#			if Apri:
#				primags={}
#				for bd in bds:
#					primags[bd]=interp[bd][iLA](Mpri)
#			nqq=10
#			BB=0.0
#			for kk in range(nqq):
#				qq=(kk+0.5)*(1.0/nqq)
#				Msec=Mini*qq
#				Asec=(Msec<=maxMini[iLA] and Msec>=minMini[iLA])
#				if Asec:
#					secmags={}
#					for bd in bds:
#						secmags[bd]=interp[bd][iLA](Msec)
#				if Apri==False and Asec==False:
#					LL=0.0
#				else:
#					LL=1.0
#					for bd in bds:
#						if Apri==True and Asec==False:
#							synmag=primags[bd]
#						if Apri==False and Asec==True:
#							synmag=secmags[bd]
#						if Apri and Asec:
#							synmag=ps.addmag(primags[bd], secmags[bd])
#						LL=LL*like1(synmag, obsmag[bd], obserr[bd], obsdet[bd])
#				BB=BB+LL*(1.0/nqq)
#			return BB*priorMini(Mini)
#		tmp=integrate.quad(binfunc, a=minMini[iLA], b=300.0, \
#			points=[5, 10, 20, 50, 100, maxMini[iLA]], full_output=1)
#		pbin=tmp[0]
#		return pbin
#
#	flag=0
#	if flag==1:
#		for ii in range(len(tabF)):
##			if ii>10: continue
#			time1=time.time()
#			filename=dr+snTab[ll]['name']+'/bin'+str(tabF[ii]['ID'])+'.dat'
#			if os.path.exists(filename): continue
#
#			pool=Pool(processes=51)
#			pbintab=pool.map(binworker, range(len(Avs)*len(LAs)))
#			pool.close()
#			pool.join()
#
#			num=0
#			postbin=np.zeros((len(Avs), len(LAs)), dtype=float)
#			for iAv in range(len(Avs)):
#				for iLA in range(len(LAs)):
#					postbin[iAv, iLA]=pbintab[num]
#					num=num+1
#			evidence=np.sum(postbin)*Avbin*LAbin
#			postbin=postbin/evidence
#			ps.pdump(postbin, filename)
#
#			print(filename,
#				'binary fit:', tabF[ii]['ID'], len(tabF),
#				'time:', round(time.time()-time1, 2))
#
#		print()
#		print('complete!')
#
### binary-star fitting{{{
##def binworker(jj):
##	(iAv, iLA)=xys[jj]
##	obsmag={}
##	obserr={}
##	obsdet={}
##	for bd in bds:
##		obsmag[bd]=tabF[ii]['mag'+bd]-(cef0[bd]*Avs[iAv]+bspars['DM'])
##		obserr[bd]=tabF[ii]['err'+bd]
##		obsdet[bd]=tabF[ii]['det'+bd]
##	def priorMini(Mini):
##		return imf(Mini)/Ntot[iLA]
##	def binlikelihood(Mini, qq):
##		Mpri=Mini
##		Msec=Mini*qq
##		Apri=(Mpri<=maxMini[iLA] and Mpri>=minMini[iLA])
##		Asec=(Msec<=maxMini[iLA] and Msec>=minMini[iLA])
##		if Apri==False and Asec==False:
##			return 0.0
##		else:
##			LL=1.0
##			for bd in bds:
##				if Apri==True and Asec==False:
##					synmag=interp[bd][iLA](Mpri)
##				if Apri==False and Asec==True:
##					synmag=interp[bd][iLA](Msec)
##				if Apri and Asec:
##					primag=interp[bd][iLA](Mpri)
##					secmag=interp[bd][iLA](Msec)
##					synmag=ps.addmag(primag, secmag)
##				LL=LL*like1(synmag, obsmag[bd], obserr[bd], obsdet[bd])
##			return LL
##	def binfunc(Mini):
##		nqq=50
##		BB=0.0
##		for kk in range(nqq):
##			qq=(kk+0.5)*(1.0/nqq)
##			pp=binlikelihood(Mini, qq)
##			BB=BB+pp*(1.0/nqq)
##		return BB*priorMini(Mini)
##	tmp=integrate.quad(binfunc, a=minMini[iLA], b=300.0, full_output=1)
##	pbin=tmp[0]
##	return pbin
#
## binary-star fitting
##def binworker(jj):
##	(iAv, iLA)=xys[jj]
##	obsmag={}
##	obserr={}
##	obsdet={}
##	for bd in bds:
##		obsmag[bd]=tabF[ii]['mag'+bd]-(cef0[bd]*Avs[iAv]+bspars['DM'])
##		obserr[bd]=tabF[ii]['err'+bd]
##		obsdet[bd]=tabF[ii]['det'+bd]
##	nqq=10
##	pbin=0.0
##	for kk in range(nqq):
##		qq=(kk+0.5)*(1.0/nqq)
##		def priorMini(Mini):
##			return imf(Mini)/Ntot[iLA]
##		def binlikelihood(Mini):
##			Mpri=Mini
##			Msec=Mini*qq
##			Apri=(Mpri<=maxMini[iLA] and Mpri>=minMini[iLA])
##			Asec=(Msec<=maxMini[iLA] and Msec>=minMini[iLA])
##			if Apri==False and Asec==False:
##				return 0.0
##			else:
##				LL=1.0
##				for bd in bds:
##					if Apri==True and Asec==False:
##						synmag=interp[bd][iLA](Mpri)
##					if Apri==False and Asec==True:
##						synmag=interp[bd][iLA](Msec)
##					if Apri and Asec:
##						primag=interp[bd][iLA](Mpri)
##						secmag=interp[bd][iLA](Msec)
##						synmag=ps.addmag(primag, secmag)
##					LL=LL*like1(synmag, obsmag[bd], obserr[bd], obsdet[bd])
##				return LL
##		def binfunc(Mini):
##			return binlikelihood(Mini)*priorMini(Mini)
##		tmp=integrate.quad(binfunc, a=minMini[iLA], b=maxMini[iLA], full_output=1)
##		BB=tmp[0]
##		pbin=pbin+BB*(1.0/nqq)
##	return pbin}}}
