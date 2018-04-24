#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

A Python implementation of the method described in [#a]_ and [#b]_ for
calculating Fourier coefficients for characterizing
closed contours.

References
----------

.. [#a] F. P. Kuhl and C. R. Giardina, “Elliptic Fourier Features of a
   Closed Contour," Computer Vision, Graphics and Image Processing,
   Vol. 18, pp. 236-258, 1982.

.. [#b] Oivind Due Trier, Anil K. Jain and Torfinn Taxt, “Feature Extraction
   Methods for Character Recognition - A Survey”, Pattern Recognition
   Vol. 29, No.4, pp. 641-662, 1996

Created by hbldh <henrik.blidh@nedomkull.com> on 2016-01-30.

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
import numpy
from sympy import *


class Model(object):

    def __init__(self, order, numPts):
        # initialize the model
        self.xlim = 0
        self.ylim = 0
        self.order = order
        self.numPts = numPts
        self.px = None
        self.px, self.py, self.zx, self.zy, self.nx, self.ny = self.init_efd_model(order)
        self.contour = None
        self.coeffs = None
        self.locus = None
        self.P = None
        self.N = None
        self.Cbar = None

    @staticmethod
    def init_efd_model(order):
        a = Symbol('a')
        b = Symbol('b')
        c = Symbol('c')
        d = Symbol('d')
        m = Symbol('m')
        n = Symbol('n')

        a1 = Symbol('a1')
        a2 = Symbol('a2')
        a3 = Symbol('a3')
        a4 = Symbol('a4')

        b1 = Symbol('b1')
        b2 = Symbol('b2')
        b3 = Symbol('b3')
        b4 = Symbol('b4')

        c1 = Symbol('c1')
        c2 = Symbol('c2')
        c3 = Symbol('c3')
        c4 = Symbol('c4')

        d1 = Symbol('d1')
        d2 = Symbol('d2')
        d3 = Symbol('d3')
        d4 = Symbol('d4')

        a_ = [a1, a2, a3, a4]
        b_ = [b1, b2, b3, b4]
        c_ = [c1, c2, c3, c4]
        d_ = [d1, d2, d3, d4]

        x = a * cos(2 * n * pi * m) + b * sin(2 * n * pi * m)
        y = c * cos(2 * n * pi * m) + d * sin(2 * n * pi * m)

        dx = x.diff(m)
        dy = y.diff(m)

        Zx_sym = 0
        Zy_sym = 0

        Px = lambdify((a, b, n, m), x)
        Py = lambdify((c, d, n, m), y)
        Zx = lambdify((a, b, n, m), dx)
        Zy = lambdify((c, d, n, m), dy)

        # precomputed symbolic stuff, will be good for real time
        for n_ in range(order):
            dx1 = dx.subs([(a, a_[n_]), (b, b_[n_]), (n, n_ + 1)])
            dy1 = dy.subs([(c, c_[n_]), (d, d_[n_]), (n, n_ + 1)])

            # symbolic value of dx,dy
            Zx_sym += dx1
            Zy_sym += dy1

        Z = sqrt(Zx_sym ** 2 + Zy_sym ** 2)
        dx_norm = Zx_sym / Z
        dy_norm = Zy_sym / Z
        ddx_norm = dx_norm.diff(m)
        ddy_norm = dy_norm.diff(m)

        tt = [m]

        ax = a_ + b_ + c_ + d_ + tt

        Nx = lambdify(ax, ddx_norm)
        Ny = lambdify(ax, ddy_norm)
        return Px, Py, Zx, Zy, Nx, Ny

    def generate_model(self, contour, xlim, ylim):
        self.contour = contour
        self.xlim = xlim
        self.ylim = ylim
        self.locus = self.calculate_dc_coefficients(self.contour)
        self.coeffs = self.elliptic_fourier_descriptors(self.contour, self.order)
        self.P, self.N, self.Cbar = self.generate_efd_model()
        # import matplotlib.pyplot as plt
        # plt.plot(contour[:, 0], contour[:, 1], 'c--', linewidth=2)
        # plt.plot(self.P[:, 0], self.P[:, 1], 'y', linewidth=2)
        # plt.show()



        return self.P, self.N, self.Cbar

    def calculate_dc_coefficients(self, contour):
        """Calculate the :math:`A_0` and :math:`C_0` coefficients of the elliptic Fourier series.

        :param numpy.ndarray contour: A contour array of size ``[M x 2]``.
        :return: The :math:`A_0` and :math:`C_0` coefficients.
        :rtype: tuple

        """
        dxy = np.diff(contour, axis=0)
        dt = np.sqrt((dxy ** 2).sum(axis=1))
        t = np.concatenate([([0., ]), np.cumsum(dt)])
        T = t[-1]

        xi = np.cumsum(dxy[:, 0]) - (dxy[:, 0] / dt) * t[1:]
        A0 = (1 / T) * np.sum(((dxy[:, 0] / (2 * dt)) * np.diff(t ** 2)) + xi * dt)
        delta = np.cumsum(dxy[:, 1]) - (dxy[:, 1] / dt) * t[1:]
        C0 = (1 / T) * np.sum(((dxy[:, 1] / (2 * dt)) * np.diff(t ** 2)) + delta * dt)

        # A0 and CO relate to the first point of the contour array as origin.
        # Adding those values to the coefficients to make them relate to true origin.
        return contour[0, 0] + A0, contour[0, 1] + C0

    def elliptic_fourier_descriptors(self, contour, order=10, normalize=False):
        """Calculate elliptical Fourier descriptors for a contour.

        :param numpy.ndarray contour: A contour array of size ``[M x 2]``.
        :param int order: The order of Fourier coefficients to calculate.
        :param bool normalize: If the coefficients should be normalized;
            see references for details.
        :return: A ``[order x 4]`` array of Fourier coefficients.
        :rtype: :py:class:`numpy.ndarray`

        """
        dxy = np.diff(contour, axis=0)
        dt = np.sqrt((dxy ** 2).sum(axis=1))
        t = np.concatenate([([0., ]), np.cumsum(dt)])
        T = t[-1]

        phi = (2 * np.pi * t) / T

        coeffs = np.zeros((order, 4))

        for n in range(1, order + 1):
            const = T / (2 * n * n * np.pi * np.pi)
            phi_n = phi * n
            d_cos_phi_n = np.cos(phi_n[1:]) - np.cos(phi_n[:-1])
            d_sin_phi_n = np.sin(phi_n[1:]) - np.sin(phi_n[:-1])
            a_n = const * np.sum((dxy[:, 0] / dt) * d_cos_phi_n)
            b_n = const * np.sum((dxy[:, 0] / dt) * d_sin_phi_n)
            c_n = const * np.sum((dxy[:, 1] / dt) * d_cos_phi_n)
            d_n = const * np.sum((dxy[:, 1] / dt) * d_sin_phi_n)
            coeffs[n - 1, :] = a_n, b_n, c_n, d_n

        if normalize:
            coeffs = self.normalize_efd(coeffs)

        return coeffs

    def normalize_efd(self, coeffs, size_invariant=True):
        """Normalizes an array of Fourier coefficients.

        See [#a]_ and [#b]_ for details.

        :param numpy.ndarray coeffs: A ``[n x 4]`` Fourier coefficient array.
        :param bool size_invariant: If size invariance normalizing should be done as well.
            Default is ``True``.
        :return: The normalized ``[n x 4]`` Fourier coefficient array.
        :rtype: :py:class:`numpy.ndarray`

        """
        # Make the coefficients have a zero phase shift from
        # the first major axis. Theta_1 is that shift angle.
        theta_1 = 0.5 * np.arctan2(
            2 * ((coeffs[0, 0] * coeffs[0, 1]) + (coeffs[0, 2] * coeffs[0, 3])),
            ((coeffs[0, 0] ** 2) - (coeffs[0, 1] ** 2) + (coeffs[0, 2] ** 2) - (coeffs[0, 3] ** 2)))
        # Rotate all coefficients by theta_1.
        for n in range(1, coeffs.shape[0] + 1):
            coeffs[n - 1, :] = np.dot(
                np.array([[coeffs[n - 1, 0], coeffs[n - 1, 1]],
                          [coeffs[n - 1, 2], coeffs[n - 1, 3]]]),
                np.array([[np.cos(n * theta_1), -np.sin(n * theta_1)],
                          [np.sin(n * theta_1), np.cos(n * theta_1)]])).flatten()

        # Make the coefficients rotation invariant by rotating so that
        # the semi-major axis is parallel to the x-axis.
        psi_1 = np.arctan2(coeffs[0, 2], coeffs[0, 0])
        psi_rotation_matrix = np.array([[np.cos(psi_1), np.sin(psi_1)],
                                        [-np.sin(psi_1), np.cos(psi_1)]])
        # Rotate all coefficients by -psi_1.
        for n in range(1, coeffs.shape[0] + 1):
            coeffs[n - 1, :] = psi_rotation_matrix.dot(
                np.array([[coeffs[n - 1, 0], coeffs[n - 1, 1]],
                          [coeffs[n - 1, 2], coeffs[n - 1, 3]]])).flatten()

        if size_invariant:
            # Obtain size-invariance by normalizing.
            coeffs /= np.abs(coeffs[0, 0])

        return coeffs

    def generate_efd_model(self):
        m_ = np.linspace(0, 1.0, self.numPts)

        Px = np.ones(self.numPts) * self.locus[0]
        Py = np.ones(self.numPts) * self.locus[1]

        Zx = 0
        Zy = 0

        a = []
        b = []
        c = []
        d = []

        # precompute symbollic stuff, will be good for real time
        for n_ in range(self.coeffs.shape[0]):
            a.append(self.coeffs[n_, 0])
            b.append(self.coeffs[n_, 1])
            c.append(self.coeffs[n_, 2])
            d.append(self.coeffs[n_, 3])

            Px += self.px(a[n_], b[n_], (n_ + 1), m_)
            Py += self.py(c[n_], d[n_], (n_ + 1), m_)
            Zx += self.zx(a[n_], b[n_], (n_ + 1), m_)
            Zy += self.zy(c[n_], d[n_], (n_ + 1), m_)

        # put together all the variables:
        N = np.zeros((self.numPts, 3))
        for i in range(0, self.numPts):
            ax = a + b + c + d
            ax.append(m_[i])
            N[i, 0] = self.nx(*ax)
            N[i, 1] = self.ny(*ax)
            N[i, 2] = 0

        # calculate norm of normal vector
        # N = np.zeros((numPts, 3))
        # N[:, 0] = Nx
        # N[:, 1] = Ny
        # N[:, 2] = 0

        Px[Px < 0] = 0
        Py[Py < 0] = 0
        Px[Px > self.xlim-1] = self.xlim-1
        Py[Py > self.ylim-1] = self.ylim-1


        P = np.zeros((self.numPts, 3))
        P[:, 0] = Px
        P[:, 1] = Py
        P[:, 2] = 0

        C = np.linalg.norm(N, axis=1)

        # cross product tells whether we have concave or convex curvature.
        crossProd = np.zeros(len(Zx))
        for ii in range(0, len(Zx)):
            aa = np.array([Zx[ii], Zy[ii], 0])
            bb = np.array(N[ii, :])
            crossProd[ii] = np.cross(aa, bb)[2]

        Cbar = np.sign(crossProd) * abs(C)
        return P, N, Cbar





