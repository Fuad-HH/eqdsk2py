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


def type_aware_plot_points_on_psi_contour(psi, spline, xx, yy, crit_points = []):
    '''
    Plots the psi values of the grid.
    '''
    fig = plt.figure()
    fig.set_size_inches(5, 5)
    ax = fig.add_subplot(111)
    plt.clabel(ax.contour(xx, yy, psi, levels=30), inline=1, fontsize=6)

    if (len(crit_points)!=0):
        for point in crit_points:
            if point[2] == 'minimum':
                ax.plot(point[0], point[1], 'bo')
            elif point[2] == 'saddle':
                ax.plot(point[0], point[1], 'ro')
            elif point[2] == 'degenerate':
                ax.plot(point[0], point[1], 'go')
            elif point[2] == 'out_of_bound':
                ax.plot(point[0], point[1], 'ko')

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
            if local_res['success'] and local_res['fun'] < 1e-10:
            #if local_res['success']:
                res.add((local_res['x'][0], local_res['x'][1]))
                #print(local_res['fun'])
    return res

def calc_hessian(spline, x, y):
    '''
    Calculates the Hessian of a spline at a point.

    Parameters
    ----------
    spline : function
        Spline function.
    x : float
        x value.
    y : float
        y value.
    ----------

    Returns
    ----------
    hessian : array
        Hessian matrix.
    ----------
    '''
    hessian = np.zeros((2, 2))
    hessian[0, 0] = spline(x, y, 2, 0)
    hessian[0, 1] = spline(x, y, 1, 1)
    hessian[1, 0] = spline(x, y, 1, 1)
    hessian[1, 1] = spline(x, y, 0, 2)
    return hessian

def get_first_derivatives(spline, x, y):
    '''
    Calculates the first derivative of a spline at a point.

    Parameters
    ----------
    spline : function
        Spline function.
    x : float
        x value.
    y : float
        y value.
    ----------

    Returns
    ----------
    first_derivative : array
        First derivative vector.
    ----------
    '''
    first_derivative = np.zeros(2)
    first_derivative[0] = spline(x, y, 1, 0)
    first_derivative[1] = spline(x, y, 0, 1)
    return first_derivative

def find_critical_point_using_root(spline, x0, y0, method='krylov'):
    '''
    Returns the critical point of a spline using root.

    Parameters
    ----------
    spline : function
        Spline function.
    x0 : float
        Initial guess for x.
    y0 : float
        Initial guess for y.
    ----------
    '''
    def spline_func(array):
        x, y = array
        # utilizing the idea of vector to achieve zero in both derivatives
        first_derivative = get_first_derivatives(spline, x, y)
        return (first_derivative[0], first_derivative[1])

    root = sp.optimize.root(spline_func, x0=[x0, y0], method=method)
    return root

def find_all_critical_points_using_root(spline, bounds, nx = 10, ny = 10, method = 'krylov'):
    '''
    Returns all the critical points of a spline using root.

    Parameters
    ----------
    spline : function
        Spline function.
    bounds : list
        Bounds for the x and y values of the grid.
    nx : int
        Number of points in the x direction.
    ny : int
        Number of points in the y direction.
    ----------
    '''
    x0 = np.linspace(bounds[0][0], bounds[0][1], nx)
    y0 = np.linspace(bounds[1][0], bounds[1][1], ny)

    res = set()
    for x in x0:
        for y in y0:
            local_res = find_critical_point_using_root(spline, x, y, method=method)
            # check if the root was found and not already in the set
            # check if the root is close to any of the exisiting roots in the set
            already_in_set = all([np.linalg.norm(np.array([local_res.x[0], local_res.x[1]]) - np.array([x1, y1])) > 1e-6 for x1, y1 in res])
            if local_res['success'] and already_in_set:
                res.add((local_res['x'][0], local_res['x'][1]))
    return res

def check_out_of_bound(x, y, bounds):
    '''
    Checks if a point is out of bounds.

    Parameters
    ----------
    x : float
        x value.
    y : float
        y value.
    bounds : list
        Bounds for the x and y values of the grid.
    ----------
    '''
    return (x < bounds[0][0] or x > bounds[0][1] or y < bounds[1][0] or y > bounds[1][1])


def find_type_of_a_critical_point(spline, x, y, bounds):
    '''
    Returns the type of a critical point.

    Parameters
    ----------
    spline : function
        Spline function.
    x : float
        x value.
    y : float
        y value.
    bounds : list
        Bounds for the x and y values of the grid.
    ----------
    '''
    if check_out_of_bound(x, y, bounds):
        return 'out_of_bound'
    hessian = calc_hessian(spline, x, y)
    if abs(np.linalg.det(hessian)) < 1e-6:
        return 'degenerate'
    if np.linalg.det(hessian) < 0:
        return 'saddle'
    else:
        if hessian[0, 0] > 0:
            return 'minimum'
        elif hessian[0, 0] < 0:
            return 'maximum'

def sort_critical_points(all_critical_points, spline, bounds):
    '''
    Returns the sorted critical points as a dictionary
    with the types as keys and a list of points as values.

    Parameters
    ----------
    all_critical_points : set
        Set of all critical points.
    spline : function
        Spline function.
    bounds : list
        Bounds for the x and y values of the grid.
    ---------- 
    '''
    sorted_critical_points = {}
    for point in all_critical_points:
        point_type = find_type_of_a_critical_point(spline, point[0], point[1], bounds)
        if point_type in sorted_critical_points:
            sorted_critical_points[point_type].append(point)
        else:
            sorted_critical_points[point_type] = [point]
    return sorted_critical_points

def get_o_points(all_critical_points, spline, bounds):
    '''
    Returns the O points.

    Parameters
    ----------
    all_critical_points : set
        Set of all critical points.
    spline : function
        Spline function.
    bounds : list
        Bounds for the x and y values of the grid.
    ----------
    '''
    return set([(x, y) for x, y in all_critical_points if find_type_of_a_critical_point(spline, x, y, bounds) == 'minimum'])

def get_x_points(all_critical_points, spline, bounds):
    '''
    Returns the X points.

    Parameters
    ----------
    all_critical_points : set
        Set of all critical points.
    spline : function
        Spline function.
    bounds : list
        Bounds for the x and y values of the grid.
    ----------
    '''
    return set([(x, y) for x, y in all_critical_points if find_type_of_a_critical_point(spline, x, y, bounds) == 'saddle'])

def plot_sorted_points_with_contour(psi, xx, yy, sorted_critical_points):
    '''
    Plots the sorted critical points with the contour plot.
    Each type of point has a different color with labels.

    Parameters
    ----------
    psi : array
        Array of psi values.
    xx : array
        Array of x values.
    yy : array
        Array of y values.
    sorted_critical_points : dict
        Dictionary of sorted critical points.
    ----------
    '''
    fig = plt.figure()
    fig.set_size_inches(5, 5)
    ax = fig.add_subplot(111)
    plt.clabel(ax.contour(xx, yy, psi, levels=20), inline=1, fontsize=6)

    for point_type in sorted_critical_points:
        for point in sorted_critical_points[point_type]:
            if point_type == 'minimum' or point_type == 'maximum':
                ax.plot(point[0], point[1], 'bo', label="O point")
            elif point_type == 'saddle':
                ax.plot(point[0], point[1], 'rx', label="X point")
            elif point_type == 'degenerate':
                ax.plot(point[0], point[1], 'go', label="Degenerate/Unknown point")
            elif point_type == 'out_of_bound':
                ax.plot(point[0], point[1], 'ko', label="Out of bound point")

    # show the legend
    ax.legend()
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title('Psi')
    plt.tight_layout()
    plt.show()