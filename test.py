import os
import pypopfit

pypopfit.starfit(cat_filename='../zzcat/cat3.dat',
	iso_filename='../zziso/iso1.dat',
	bands=['F275W', 'F336W', 'F438W', 'F555W', 'F625W', 'F814W'],
	logAge_min=6.60, logAge_max=7.60, logAge_bin=0.01,
	intAv_min=0.0, intAv_max=0.5, intAv_bin=0.01,
	DM=30.0, gal_Av=0.1,
	gal_coeff={'F275W': 1.989, 'F336W': 1.633, 'F438W': 1.332,
		'F555W': 0.989, 'F625W': 0.866, 'F814W': 0.578},
	int_coeff={'F275W': 1.989, 'F336W': 1.633, 'F438W': 1.332,
		'F555W': 0.989, 'F625W': 0.866, 'F814W': 0.578},
	Pbin=0.5, alpha=-2.35, minStellarMass=0.08, maxStellarMass=100.0,
	dmag_min=0.02, num_core=50, path='../zzres3/', overwrite=True)
