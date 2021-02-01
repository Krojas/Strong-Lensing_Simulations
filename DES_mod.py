def getCoadd(ra,dec,width=100,widthType='pixels',bands=['i'],pad=True):
    import numpy as np 
    import astropy.io.fits as fits
    import fitsio
#    import wcs
    from astropy.wcs import WCS
    import glob
    import matplotlib.pyplot as plt

    tData = fits.open('/ltstorage/astro/rojasola/Y3A1_table.fits')[1].data
    raMin = tData['raMin']
    raMax = tData['raMax']
    decMin = tData['decMin']
    decMax = tData['decMax']
    cross = tData['crossRA']

    cosdec = np.cos(dec*np.pi/180.)
    if ra-1./cosdec<0:
        raMin[cross] = raMin[cross]-360.
    elif ra+1./cosdec>360:
        raMax[cross] += 360.

    c = (ra>raMin)&(ra<raMax)&(dec>decMin)&(dec<decMax)

    RA = (raMin[c]+raMax[c])/2.
    DEC = (decMin[c]+decMax[c])/2.
    dRA2 = ((RA-ra)*cosdec)**2
    dDEC2 = (dec-DEC)**2  
    offsets = dRA2+dDEC2
        
    arg = offsets.argmin()
    
    
    tile = tData['tilename'][c][arg]
    print(tile)
    path = '/ltstorage/astro/rojasola/desdr-server.ncsa.illinois.edu/despublic/dr1_tiles/'+tile+'/'

    if widthType == 'arcsecs':
        width = int(width/0.263)+1

    first = True
    oimg = None
    allcutouts = []
    for band in bands:
        filename = glob.glob(path+'/*_'+band+'.fits.fz')[0]
        desImage = fitsio.FITS(filename)[1]
        hdr = desImage.read_header()
        w = WCS(hdr)

        if first:
            x0,y0 = w.all_world2pix(ra,  dec, 0)
            x0,y0 = int(x0),int(y0)
            xmin = int(x0-width/2)
            ymin = int(y0-width/2)

            if xmin<0 or ymin<0 or xmin+width>10000 or ymin+width>10000 and pad==True:
                oimg = np.zeros((width,width))
                x1,x2 = 0,width
                if xmin<0:
                    x1 = abs(xmin)
                elif xmin+width>10000:
                    x2 = 10000-xmin
                y1,y2 = 0,width
                if ymin<0:
                    y1 = abs(ymin)
                elif ymin+width>10000:
                    y2 = 10000-ymin

            xlo = max(xmin,0)
            ylo = max(ymin,0)
            xhi = min(10000,xmin+width)
            yhi = min(10000,ymin+width)

        cutout = desImage[ylo:yhi,xlo:xhi]

        if oimg is not None:
            oimg[y1:y2,x1:x2] = cutout.copy()
            cutout = oimg.copy()
        allcutouts.append(cutout)
        #plt.figure()
        #plt.imshow(cutout, interpolation='nearest', origin='lower')
        #plt.show()
        hdr['CRPIX1'] -= xlo
        hdr['CRPIX2'] -= ylo
        '''if outroot is None:
            oname = str(ra)+str(dec)+'_%s.fits'%(band)
        else:
            oname = '%s_%s.fits'%(outroot,band)
        ohdu = fitsio.FITS(oname,'rw',clobber=True)
        ohdu.write(cutout,header=hdr)
        ohdu.close()'''

    return allcutouts,hdr,tile,x0,y0

