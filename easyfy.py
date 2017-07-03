'''====
Easyfy
====

Provides several easy to use implementations of arbitrary algorithms.
-------------------------------------------------------------------------------
Copyright (C) 2014 - Bjorn J. Scholz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details: http://www.gnu.org/licenses.
-------------------------------------------------------------------------------
'''
import numpy
import matplotlib.pylab
import os
import colormaps as cmaps
import tables

goldenRatio = 1.61803398875
class colors:
    def __init__(self):
        self.maroon = '#800000'
        self.salmon = '#FFA4AA'
        self.gray = '#A0A0A0'
        self.darkGray = '#333333'
        self.black = '#000000'
        self.red = '#E40505'
        self.blue = '#0041C2'
        self.green = '#00BD1A'
        self.teal = '#00996D'
        self.yellow = '#F1C40F'
        self.velvet = '#7D0552'
        self.brown = '#6E2C00'
        self.pink = '#CD10BB'
#        self.orange = '#E67E22'
        self.orange = '#FF7F00'
        self.desaturatedBlue = '#5088B2'
        self.desaturatedRed = '#EC7063'
        self.darkGreen = '#2a6e3c'
        self.desaturatedGreen = '#99bf62'
        self.desaturatedOrange = '#e06e44'
        self.desaturatedYellow = '#FADA5E'
        self.desaturatedTeal = '#9EE0D0'#'#A7EDDC'

    def get_colors(self):
        return [self.gray,self.black,self.red,self.blue,self.green,self.teal,self.yellow,self.velvet,self.brown,self.pink,self.orange]

def convertPolimiToHDF5(inputFileName,outputFileName):
    hdf5_file = tables.openFile(outputFileName, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    eid_storage = hdf5_file.createEArray(hdf5_file.root, 'eid', tables.Atom.from_dtype(numpy.dtype('i8')), shape=(0,), filters=filters, expectedrows=0)
    pid_storage = hdf5_file.createEArray(hdf5_file.root, 'pid', tables.Atom.from_dtype(numpy.dtype('i4')), shape=(0,), filters=filters, expectedrows=0)
    pt_storage  = hdf5_file.createEArray(hdf5_file.root, 'pt',  tables.Atom.from_dtype(numpy.dtype('i4')), shape=(0,), filters=filters, expectedrows=0)
    ct_storage  = hdf5_file.createEArray(hdf5_file.root, 'ct',  tables.Atom.from_dtype(numpy.dtype('i4')), shape=(0,), filters=filters, expectedrows=0)
    tar_storage = hdf5_file.createEArray(hdf5_file.root, 'tar', tables.Atom.from_dtype(numpy.dtype('i4')), shape=(0,), filters=filters, expectedrows=0)
    cel_storage = hdf5_file.createEArray(hdf5_file.root, 'cel', tables.Atom.from_dtype(numpy.dtype('i4')), shape=(0,), filters=filters, expectedrows=0)
    en_storage  = hdf5_file.createEArray(hdf5_file.root, 'en',  tables.Atom.from_dtype(numpy.dtype('f4')), shape=(0,), filters=filters, expectedrows=0)
    tim_storage = hdf5_file.createEArray(hdf5_file.root, 'tim', tables.Atom.from_dtype(numpy.dtype('f8')), shape=(0,), filters=filters, expectedrows=0)
    x_storage   = hdf5_file.createEArray(hdf5_file.root, 'x',   tables.Atom.from_dtype(numpy.dtype('f4')), shape=(0,), filters=filters, expectedrows=0)
    y_storage   = hdf5_file.createEArray(hdf5_file.root, 'y',   tables.Atom.from_dtype(numpy.dtype('f4')), shape=(0,), filters=filters, expectedrows=0)
    z_storage   = hdf5_file.createEArray(hdf5_file.root, 'z',   tables.Atom.from_dtype(numpy.dtype('f4')), shape=(0,), filters=filters, expectedrows=0)
    wgt_storage = hdf5_file.createEArray(hdf5_file.root, 'wgt', tables.Atom.from_dtype(numpy.dtype('f4')), shape=(0,), filters=filters, expectedrows=0)
    gen_storage = hdf5_file.createEArray(hdf5_file.root, 'gen', tables.Atom.from_dtype(numpy.dtype('i4')), shape=(0,), filters=filters, expectedrows=0)
    nsc_storage = hdf5_file.createEArray(hdf5_file.root, 'nsc', tables.Atom.from_dtype(numpy.dtype('i4')), shape=(0,), filters=filters, expectedrows=0)
    ico_storage = hdf5_file.createEArray(hdf5_file.root, 'ico', tables.Atom.from_dtype(numpy.dtype('i4')), shape=(0,), filters=filters, expectedrows=0)
    eni_storage = hdf5_file.createEArray(hdf5_file.root, 'eni', tables.Atom.from_dtype(numpy.dtype('f4')), shape=(0,), filters=filters, expectedrows=0)

    with open(inputFileName,'r') as f:
        for i,line in enumerate(f):
            print 'Converting line ',i
            data =numpy.array(line.split(),dtype=numpy.float)
            eid_storage.append(numpy.array([data[0]], dtype='i8'))
            pid_storage.append(numpy.array([data[1]], dtype='i4'))
            pt_storage.append(numpy.array([data[2]], dtype='i4'))
            ct_storage.append(numpy.array([data[3]], dtype='i4'))
            tar_storage.append(numpy.array([data[4]], dtype='i4'))
            cel_storage.append(numpy.array([data[5]], dtype='i4'))
            en_storage.append(numpy.array([data[6]], dtype='f4'))
            tim_storage.append(numpy.array([data[7]], dtype='f8'))
            x_storage.append(numpy.array([data[8]], dtype='f4'))
            y_storage.append(numpy.array([data[9]], dtype='f4'))
            z_storage.append(numpy.array([data[10]], dtype='f4'))
            wgt_storage.append(numpy.array([data[11]], dtype='f4'))
            gen_storage.append(numpy.array([data[12]], dtype='i4'))
            nsc_storage.append(numpy.array([data[13]], dtype='i4'))
            ico_storage.append(numpy.array([data[14]], dtype='i4'))
            eni_storage.append(numpy.array([data[15]], dtype='f4'))
    hdf5_file.close()

# =============================================================================
#               2 dimensional histogrammer
# =============================================================================
def hist2d(x,y,nBinsX,nBinsY,rngX=None,rngY=None):
    ''' Generates a 2D histogram
    Parameters
    ----------
    x : array_like
        Data that will be represented on the x-axis
    y : array_like
        Data that will be represented on the y-axis
    nBinsX : tuple
        Number of bins along x-axis
    nBinsY : tuple
        Number of bins along x-axis
    rngX : array_like
        Minima and maxima of x-axis
    rngY : array_like
        Minima and maxima of y-axis
    Returns
    -------
    h2d : array_like
        Histogram data
    extent : array_like
        Array of the outer edges of the histogram. Each bin is eaqually spaced
        within this extent. Mainly used for graphical representation.

    Example
    -------
    The following code snippet explains the use of the 2d histogrammer

    >>> import easytemplates as et
    >>> import numpy as np
    >>> import matplotlib.pylab as plt
    >>>
    >>> n = 1000000
    >>> x = numpy.random.rand(n)
    >>> y = numpy.random.rand(n)
    >>>
    >>> plt.figure(figsize=(12,7),edgecolor='k',facecolor='w')
    >>> ax = plt.subplot(111)
    >>> H, ex = et.hist2d(x,y,120,70,[0,1],[0,1])
    >>> plt.imshow(H,extent=ex,aspect='auto')
    >>> plt.tight_layout()
    >>> plt.show()
    '''
    if rngX == None and rngY == None:
        h2d, xp, yp = numpy.histogram2d(y,x,bins=(nBinsY,nBinsX))
    else:
        h2d, xp, yp = numpy.histogram2d(y,x,bins=(nBinsY,nBinsX),range=[rngY,rngX])
    extent = [yp[0],yp[-1],xp[0],xp[-1]]
    return h2d, extent

# =============================================================================
#               2 dimensional histogrammer and plotter
# =============================================================================
def plotHist2d(x,y,nBinsX,nBinsY,rngX=None,rngY=None,contours=True):
    ''' Generates and plots a 2D histogram
    Parameters
    ----------
    x : array_like
        Data that will be represented on the x-axis
    y : array_like
        Data that will be represented on the y-axis
    nBinsX : tuple
        Number of bins along x-axis
    nBinsY : tuple
        Number of bins along x-axis
    rngX : array_like
        Minima and maxima of x-axis
    rngY : array_like
        Minima and maxima of y-axis
    contours: boolean
        Determins whether contour lines are being plotted
    Returns
    -------
    Nothing

    Example
    -------
    The following code snippet explains the use of the 2d histogram plotter

    >>> import easytemplates as et
    >>> import numpy as np
    >>> import matplotlib.pylab as plt
    >>>
    >>> n = 1000000
    >>> x = numpy.random.rand(n)
    >>> y = numpy.random.rand(n)
    >>>
    >>> plt.figure(figsize=(12,7),edgecolor='k',facecolor='w')
    >>> ax = plt.subplot(111)
    >>> et.plotHist2d(x,y,120,70,[0,1],[0,1])
    >>> plt.tight_layout()
    >>> plt.show()
    '''
    cm = cmaps.plasma

    if rngX == None and rngY == None:
        h2d, xp, yp = numpy.histogram2d(y,x,bins=(nBinsY,nBinsX))
    else:
        h2d, xp, yp = numpy.histogram2d(y,x,bins=(nBinsY,nBinsX),range=[rngY,rngX])
    extent = [yp[0],yp[-1],xp[0],xp[-1]]
    palette = cm
    palette.set_under('w',0.0)
    img = matplotlib.pylab.imshow(h2d,origin='lower',extent=extent,aspect='auto',vmin=1e-50,interpolation='nearest',cmap=palette)
    if contours: matplotlib.pylab.contour(h2d,origin='lower',extent=extent,cmap=cm)
    return img

# =============================================================================
#      Runge Kutta Fehlberg iterator with variable stepsize of order 4 and 5
# =============================================================================
def RKF45(f,t0=0,y0=[0],h0=0.001,a=1e-3,steps=100):
    '''
    Runge Kutta Fehlber iterator with variable stepsize of order 4 and 5

    Parameters
    ----------
    f : function
        Right hand side of an ODE system
    t0 : float
        Starting time. Only of importance if the system explicitly depends
        on t
    y : array_like
        Initial conditions. Shape and mapping has to match the output of f
    h0 : float
        Initial size of a timestep
    a : float
        Desired accuracy that is used to calculate h inbetween steps

    Returns
    -------
    output : array_like
        Array of variables at each timestep. o[0] is always t!

    Example
    -------
    The following code snippet explains the use of the Runge Kutta Fehlberg
    iterator of order 4 and 5.

    >>> import matplotlib.pylab as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> import numpy as np
    >>> import easytemplates as et
    >>>
    >>> # Right hand side of the Lorenz Attractor System
    >>> def Lorenz(t,y):
    >>>     a = 10.
    >>>     r = 70.
    >>>     r = 28
    >>>     b = 8./3.
    >>>
    >>>     L = numpy.zeros(3)
    >>>     L[0] = -a*y[0]+a*y[1]
    >>>     L[1] = r*y[0]-y[1]-y[0]*y[2]
    >>>     L[2] = -b*y[2]+y[0]*y[1]
    >>>     return L
    >>>
    >>> out = et.RKF45(Lorenz,0,[-10,-10,15],0.01,1e-3,1000)
    >>> fig = plt.figure(figsize=(7,7),edgecolor='k',facecolor='w')
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> plt.plot(out[1],out[2],zs=out[3])
    >>> plt.tight_layout()
    >>> plt.show()
    '''
    output = numpy.append(t0,y0)
    y = numpy.array(y0)
    h = h0
    t = t0
    for i in range(steps):
        k1 = h*f(t,y)
        k2 = h*f(t+0.25*h, y+0.25*k1)
        k3 = h*f(t+3.0/8.0*h, y+(3.0*k1+9.0*k2)/32.0)
        k4 = h*f(t+12.0/13.0*h, y+(1932.0*k1-7200.0*k2+7296.0*k3)/2197.0)
        k5 = h*f(t+h, y+(439.0/216.0*k1-8.0*k2+3680.0/513.0*k3-845.0/4104.0*k4))
        k6 = h*f(t+0.5*h, y -8.0/27.0*k1 + 2.0*k2-3544.0/2565.0/k3+1859.0/4104.0*k4-11.0/40.0*k5)
        z = y + 16.0*135.0*k1 + 6656.0/12825.0*k3+28561.0/56430.0*k4-9.0/50.0*k5+2.0/55.0*k6
        y = y + 25.0/216.0*k1+1408.0/2565.0*k3+2197.0/4104.0*k4-0.2*k5
        t = t + h
        _tmpOut = numpy.append(t,y)
        output = numpy.column_stack((output,_tmpOut))
        h = (0.5*a*h/(numpy.sqrt(numpy.sum((z-y)*(z-y)))))**0.25
    return output

def heaviside(x):
    return 0.5 * (numpy.sign(x) + 1)
