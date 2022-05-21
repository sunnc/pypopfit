import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
import pypopfit

DM=30.0
gal_Av=0.0
gal_coeff={'F275W': 1.989, 'F336W': 1.633, 'F438W': 1.332, 'F555W': 0.989, 'F625W': 0.866, 'F814W': 0.578}
int_coeff=gal_coeff.copy()

bands=['F275W', 'F336W', 'F438W', 'F555W', 'F625W', 'F814W']
magcuts={'F275W': 31.0, 'F336W': 31.0, 'F438W': 31.0, 'F555W': 31.0, 'F625W': 31.0, 'F814W': 31.0}

cat_filename='zzcat/cat.dat'
iso_filename='zziso/iso1.dat'

pypopfit.starfit(cat_filename=cat_filename, iso_filename=iso_filename,
	index=range(3), bands=bands,
	logAge_min=6.60, logAge_max=7.60, logAge_bin=0.01,
	intAv_min=0.0, intAv_max=2.0, intAv_bin=0.01,
	DM=DM, gal_Av=gal_Av, gal_coeff=gal_coeff, int_coeff=int_coeff,
	Pbin=0.5, alpha=-2.35, minStellarMass=0.08, maxStellarMass=300.0,
	magcuts=magcuts, dmag_min=0.02, num_core=4, path='zzres/', overwrite=True)
