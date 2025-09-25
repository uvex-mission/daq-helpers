#!/usr/bin/python3
# Author: Chaz Shapiro (2022)
#
# Digitally CDS subtract a CMOS MKxNK raw image
# Output will be written to same directory as input
# Assumes 1st and 2nd frames of CDS pair are stacked channel-wise in the X-direction:
#
# | Ch0,t0 | Ch0,t1 | Ch1,t0 | Ch1,t1 | ... | ChN,t0 | ChN,t1 |
# | Ch0,t0 | Ch0,t1 | Ch1,t0 | Ch1,t1 | ... | ChN,t0 | ChN,t1 |
# | Ch0,t0 | Ch0,t1 | Ch1,t0 | Ch1,t1 | ... | ChN,t0 | ChN,t1 |
#
# FITS file can be single extension with a 3D image "cube" or multi-extension with 1 image per ext.
# Multi-extension assumes all images are the same dimensions

CHANNEL_SIZE = 256 # Numnber of columns in one MKxNK readout channel

import astropy.io.fits as pf                                                                                                        
import argparse
import os.path as path
import sys
from numpy import array, median
import numpy as np  # for greycode

## Gray Code Descrambler

def binary_to_gray(n):
    n = int(n)
    n ^= (n>>1)
    return n

def DescramblerGrayCodeImage(image):
    ysize, xsize = image.shape
    nchan = int(xsize/CHANNEL_SIZE)
    binarycode = range(CHANNEL_SIZE)
    graycode = np.array([binary_to_gray(i) for i in binarycode])
    t = np.argsort(graycode, axis=-1, kind=None, order=None) ##
    image = np.reshape(image,(ysize,CHANNEL_SIZE,nchan), order='F')
    image = image[:,t,:]
    return image.reshape((ysize,CHANNEL_SIZE*nchan), order='F')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Digitally CDS subtract a raw image from CMOS MKxNK')
    parser.add_argument('filename', type=str, nargs='*')
    parser.add_argument('-tag', type=str, default='cds' ,help='Output filename tag; default=cds')
    parser.add_argument('-nchan' ,type=int ,default=None ,help='Override number of channels')
    parser.add_argument('-outtype', type=str, default='int16' ,help='Output data type; default=int16')
    parser.add_argument('-reference', type=str, default=None, help='path and filename of reference frame to subtract')
    parser.add_argument('-verbose', action='store_true', default=False ,help='Print output filenames as they are written')
    parser.add_argument('-stack', action='store_true', default=False ,help='Stack frames via medianing')
    parser.add_argument('-skip', type=int, default=0 ,help='Discard 1st N frames')
    parser.add_argument('-grey', action='store_true', default=False ,help='Descramble rows using grey coding')
    parser.add_argument('-rawdual', action='store_true', default=False ,help='Deinterlace and stitch together raw dual gain frames')
    parser.add_argument('-raw', action='store_true', default=False ,help='Deinterlace and stitch together raw frames without subtracting')
    # parser.add_argument('-mex', action='store_true', default=False ,help='Save as multi-extention FITS')

    args = parser.parse_args()

    outtag = args.tag+'_'

    for f in args.filename:
        hdulist = pf.open(f ,memmap=False)
        hdr = hdulist[0].header  # telemetry is in 0th FITS header
        MULTIEXT = (len(hdulist)>1)   # Is the file multi-extension?

        # Trim data according to format and skip parameter
        if MULTIEXT:
            hdu0 = hdulist[0].copy()
            hdulist = hdulist[args.skip+1:]  # Remove Primary HDU (0th) and skipped images
        else:
            if hdulist[0].data.ndim > 2:  # Check if image is 3D
                hdulist[0].data = hdulist[0].data[args.skip:]

        imgcube = []

        # Check header for High Dynamic Range - unfortunately same acronym as "hdr"
        gainmode = hdr['GAIN']
        DUALGAIN = (gainmode.strip().upper() == 'HDR')

        for h, hdu in enumerate(hdulist):

            img = hdu.data.astype(float)
            if img.ndim == 2: ysize,xsize = img.shape
            elif img.ndim == 3: zsize,ysize,xsize = img.shape
            
            nchan = args.nchan if args.nchan else int(xsize/CHANNEL_SIZE/2) # 1st/2nd frames are stacked column-wise
            print('nchan',nchan)

            if xsize%(2*nchan) > 0:
                print('ERROR: Image width is not a multiple of 2x number of channels')
                sys.exit(-1)

            colsperchan = int(xsize/nchan/2)  

            # Subtract a reference file or use only data from the image?
            if args.reference:
                refimg = pf.getdata(args.reference).astype(float)
                diff = img - refimg

            elif args.rawdual: # Channel group order is 0: Reset Low, 1: Reset High, 2: Signal High, 3: Signal Low, repeat...
                img.resize(ysize,nchan*2,colsperchan)  # Organize image by channel
                reset_lo = img[:,0::4].reshape(ysize,-1)
                reset_hi = img[:,1::4].reshape(ysize,-1)
                signal_hi = img[:,2::4].reshape(ysize,-1)
                signal_lo = img[:,3::4].reshape(ysize,-1)
                diff = np.hstack((reset_lo,signal_lo, reset_hi, signal_hi))

            elif DUALGAIN: # Channel group order is 0: Reset Low, 1: Reset High, 2: Signal High, 3: Signal Low, repeat...
                img.resize(ysize,nchan*2,colsperchan)  # Organize image by channel
                reset_lo = img[:,0::4]
                reset_hi = img[:,1::4]
                signal_hi = img[:,2::4]
                signal_lo = img[:,3::4]
                diff_lo = -(signal_lo - reset_lo).reshape(ysize,-1)
                diff_hi = -(signal_hi - reset_hi).reshape(ysize,-1)
                diff = np.hstack((diff_lo,diff_hi))

            elif args.raw:
                img.resize(ysize,nchan*2,colsperchan)  # Organize image by channel
                raw0 = img[:,0::2].reshape(ysize, nchan*colsperchan)
                raw1 = img[:,1::2].reshape(ysize, nchan*colsperchan)
                diff = np.hstack((raw0,raw1))       # Difference every other channel (1st/2nd frames are adjacent)

            else:
                img.resize(ysize,nchan*2,colsperchan)  # Organize image by channel
                diff = (img[:,0::2]-img[:,1::2])       # Difference every other channel (1st/2nd frames are adjacent)
                diff.resize(ysize,int(xsize/2))        # Stitch channels back together; new width is 1/2 raw image width

            if (args.grey and not args.reference): # Reference image assumed already deinterlaced
                diff = DescramblerGrayCodeImage(diff)

            if MULTIEXT:
                hdu.data = diff.copy().astype(args.outtype)  # update the HDUlist
            else:
                imgcube.append(diff.squeeze())  # update the image cube ### This only happens once if not MULTIEXT

            del img

        if MULTIEXT and args.stack:
            imgcube = [ hdu.data for hdu in hdulist ]  # Put reduced extensions in 1 array

        imgcube = array(imgcube).squeeze()

        # Only stack images if the cube has a 3rd axis
        if args.stack and imgcube.ndim>2:
            print('Stacking images: %i' % len(imgcube))
            imgcube = median(imgcube ,axis=0)

        # Prepend tag to filename and save in same path
        inpath,basename=path.split(f)
        if inpath=='': inpath='.'
        outpath=inpath+'/'+outtag+basename

        # Save as Multi-extension?
        if imgcube.size == 0: # We didn't use imgcube
            hdulist.insert(0, hdu0)  # Add the primary HDU back
            hdulist.writeto(outpath, overwrite=True)
        else:
            pf.writeto(outpath,imgcube.astype(args.outtype),overwrite=True ,header=hdr)

        if args.verbose: print(outpath)

        hdulist.close()

