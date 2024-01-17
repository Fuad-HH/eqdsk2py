'''
This file has all the necessary functions to read a GEDSK file and
do some basic manipulations on it.
'''

import numpy as np
import scipy as sp
from freeqdsk import geqdsk
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def read_gedsk(filename):
    '''
    Reads a G-EQDSK file and returns a dictionary with all the
    information in it.
    '''
    with open(filename, "r") as f:
        data = geqdsk.read(f)
    return data

def get_psi(data):
    '''
    Returns the psi values of the grid.
    '''
    return data['psi'].T

def get_meshgrid_for_psi(data, mult = 1):
    '''
    Returns the meshgrid for psi.

    
    Parameters
    ----------
    data : dict
        Dictionary with the data from the G-EQDSK file.
    mult : int
        Multiplier for the number of points in the grid. Default is 1.
    ----------
    '''
    x = np.linspace(data['rleft'], data['rleft'] + data['rdim'], mult*data['nx'])
    y = np.linspace(data['zmid'] - data['zdim'] / 2, data['zmid'] + data['zdim'] / 2, mult*data['ny'])
    xx, yy = np.meshgrid(x, y)
    return xx, yy

def get_xy(data):
    '''
    Returns the x and y values of the grid.
    '''
    x = np.linspace(data['rleft'], data['rleft'] + data['rdim'], data['nx'])
    y = np.linspace(data['zmid'] - data['zdim'] / 2, data['zmid'] + data['zdim'] / 2, data['ny'])
    return x, y

def get_bounds(data):
    '''
    Returns the bounds for the x and y values of the grid.
    '''
    x = (data['rleft'], data['rleft'] + data['rdim'])
    y = (data['zmid'] - data['zdim'] / 2, data['zmid'] + data['zdim'] / 2)
    return [x, y]

def plot_psi(data, mult = 1):
    '''
    Plots the psi values of the grid.
    '''
    fig = plt.figure()
    fig.set_size_inches(8, 8)
    ax = fig.add_subplot(111, projection='3d')
    xx, yy = get_meshgrid_for_psi(data, mult)
    ax.plot_surface(xx, yy, get_psi(data), cmap='jet')
    # add x, y, z labels
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_zlabel('Psi [Wb/rad]')
    # add a title
    ax.set_title('Psi')
    plt.tight_layout()
    plt.show()

# TODO: add points to the plot
def plotly_psi_spline(spline, x, y, points = []):
    '''
    Plots the psi values of the grid.
    '''
    if (len(points)==0):
        fig = go.Figure(data=[go.Surface(z=spline(x, y).T /10, x=x, y=y)])
    else:
        fig = go.Figure(data=[go.Surface(z=spline(x, y).T /10, x=x, y=y)])
        for point in points:
            fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=spline(point[0], point[1])/10, mode='markers', marker=dict(color='black')))
    fig.update_layout(title='Psi',
                    autosize=False,
                    width=500, height=500,
                    margin=dict(l=50, r=50, b=50, t=50))
    # add a contour plot at psi = 0
    fig.show()

def plotPoints_on_psi_contour(psi, xx, yy, points = []):
    '''
    Plots the psi values of the grid.
    '''
    fig = plt.figure()
    fig.set_size_inches(5, 5)
    ax = fig.add_subplot(111)
    plt.clabel(ax.contour(xx, yy, psi, levels=30), inline=1, fontsize=6)

    if (len(points)!=0):
        for point in points:
            ax.plot(point[0], point[1], 'ro')

    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title('Psi')
    plt.tight_layout()
    plt.show()

def get_spline_for_psi(psi, x, y):
    '''
    Returns a spline for psi.
    '''
    return sp.interpolate.RectBivariateSpline(x, y, psi.T)

def get_local_minimum(spline, bounds, x0, y0, method='Nelder-Mead', order=0, use_abs=True):
    '''
    Returns the local minimum of a spline.
    '''
    if use_abs:
        res = sp.optimize.minimize(lambda x: np.abs(spline(x[0], x[1], order, order)), x0=[x0, y0], bounds=bounds, method=method)
    else:
        res = sp.optimize.minimize(lambda x: spline(x[0], x[1], order, order), x0=[x0, y0], bounds=bounds, method=method)
    return res

def find_all_critical_points(spline, bounds, nx = 10, ny = 10):
    '''
    Returns all the critical points of a spline.
    '''
    x0 = np.linspace(bounds[0][0], bounds[0][1], nx)
    y0 = np.linspace(bounds[1][0], bounds[1][1], ny)

    res = set()
    for x in x0:
        for y in y0:
            local_res = get_local_minimum(spline, bounds, x, y, order=1, use_abs = True)
            #if local_res['success'] and local_res['fun'] < 1e-10:
            if local_res['success']:
                res.add((local_res['x'][0], local_res['x'][1]))
    return res
