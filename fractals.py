from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy.matlib
import time


# Some presets to play around with,
# JULIA SET:
#     colors=5, color_offset=0.25, c=complex(0.25, -0.54)
# NEWTON SET:
#     xmin=-3.0, xmax=3.0, ymin=-3.0, ymax=3.0, f=TRIG1


# Configure numpy printing options.
np.set_printoptions(threshold=np.nan)
np.warnings.filterwarnings("ignore")


class Fractal(object):

    def render(self, width=300, height=300, zoom=None, itermax=50,
            colors=5, color_offset=0.5, **kwargs):
        """
        Return RGB image representing fractal.

        :param width: Pixel resolution in x-axis
        :param height: Pixel resolution in y-axis
        :param itermax: Maximum number of iterations
        :param zoom: Tuple containing (x-min, x-max, y-min, y-max)
        :param colors: Number of colors (hues) included in the rendered image
        """
        if zoom is None:
            zoom = self._defaultZoom()

        complex_plane = self._complexPlane(width, height, *zoom)
        fractal = self._computeFractal(complex_plane, itermax, **kwargs)
        rgb_image = self._toRgbImage(fractal, colors, color_offset)

        # Display fractal on screen.
        self._show(rgb_image)

    def _show(self, a):
        """
        Display given RGB array.

        :type a: np.ndarray
        """
        fig = plt.figure()
        fig.set_size_inches((2, 2))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.set_cmap('hot')
        ax.imshow(a, aspect='equal')
        plt.show()

    def _toRgbImage(self, fractal, colors, color_offset):
        """Converts the generated fractal into an RGB image array."""
        hsv_img = np.array(
            [
                # Cycle through color wheel.
                (fractal * colors + color_offset) % 1,

                # Saturation = 1 where fractal values > 0,
                # Saturation = 0 otherwise.
                fractal.astype(dtype=bool).astype(dtype=float),

                # Invert colours
                1 - fractal
            ]
        ).astype(dtype=float).T

        rgb_img = (mpl.colors.hsv_to_rgb(hsv_img) * 255).astype(dtype=np.uint8)
        return rgb_img

    def _complexPlane(self, n, m, xmin, xmax, ymin, ymax):
        """Return matrix representing the complex plane."""
        # Create two matrices of size n x m
        ix, iy = np.mgrid[0:n, 0:m]

        # Create range of values in the x- and y-axis
        real_part = np.linspace(xmin, xmax, n)[ix]
        imag_part = np.linspace(ymin, ymax, m)[iy] * complex(0, 1)

        complex_plane = real_part + imag_part
        return complex_plane

    def _defaultZoom(self):
        """Return default zoom setting."""
        return (-1.0, 1.0, -1.0, 1.0)


# {{{ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# }}} Mandelbrot Fractal

class Mandelbrot(Fractal):

    def _computeFractal(self, complex_plane, itermax, p=2):
        c = complex_plane
        z = np.copy(c)

        # Create matrix to represent this fractal and escaped values.
        fractal = np.matlib.zeros(c.shape, dtype=float)
        escaped = np.matlib.zeros(c.shape, dtype=bool)

        for i in range(itermax):
            # Mandelbrot uses function: f(z) = z^p + c; z, p are const.
            z = np.power(z, p) + c

            # Smooth borders in fractal.
            temp_escaped = (abs(z) > 2.0)
            np.copyto(
                fractal,
                (i + 1 - np.log(np.log(np.absolute(z))) / np.log(2)),
                casting='no',
                where=np.invert(escaped) & temp_escaped
            )
            escaped = temp_escaped

        # Represent fractal as floats ranging between 0 and 1.
        fractal /= itermax
        fractal[fractal > 1] = 1
        fractal[fractal < 0] = 0

        return fractal

    def _defaultZoom(self):
        return (-2.0, 1.0, -1.5, 1.5)


# {{{ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# }}} Julia Fractal

class Julia(Fractal):

    def _computeFractal(
            self, complex_plane, itermax, p=2, c1=0.25, c2=-0.54):
        c = complex(c1, c2)
        z = complex_plane

        # Create matrix to represent this fractal.
        fractal = np.matlib.zeros(z.shape, dtype=float)

        for i in range(itermax):
            # Julia set uses function: f(z) = z^p + c; c, p are const.
            z = np.power(z, p) + c

            # Fractal shows number of iterations before values 'escape'.
            np.copyto(
                fractal,
                fractal + np.exp(-np.absolute(z)),
                casting='no',
                where=np.invert(np.isnan(z))
            )

        # Represent fractal as floats ranging between 0 and 1.
        fractal /= itermax
        fractal[fractal > 1] = 1
        fractal[fractal < 0] = 0

        # Returns the image matrix
        return fractal

    def _toRgbImage(self, fractal, colors, color_offset):
        """
        Converts the generated fractal into an RGB image array
        
        :return: ndarry of shape (n, m, 3)
        """
        hsv_img = np.array(
            [
                # Cycle through color wheel.
                (fractal * colors + color_offset) % 1,

                # Saturation = fractal value.
                fractal,

                # Maximum value.
                np.ones(fractal.shape)
            ]
        ).astype(dtype=float).T

        rgb_img = (mpl.colors.hsv_to_rgb(hsv_img) * 255).astype(dtype=np.uint8)
        return rgb_img

    def _defaultZoom(self):
        return (-1.25, 1.25, -1.25, 1.25)


# {{{ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# }}} Newton Fractal

class Function(object):
    """Represents a mathematical function."""

    def __init__(self, name, func, deriv):
        self.name = name
        self._f = func
        self._df = deriv

    def __call__(self, *args):
        return self._f(*args)

    def newtonsMethod(self, x, a):
        """Approximates root of this function using single iteration of 
           Newton's method.
        """
        return x - a * (self._f(x) / self._df(x))


SIN = Function(
    "Sin",
    func=np.sin,
    deriv=np.cos
)
COS = Function(
    "Cosine",
    func=np.cos,
    deriv=lambda x: -1 * np.sin(x)
)
TRIG1 = Function(
    "Composite Trig",
    func=lambda x: np.cos(np.sin(x)) - np.pi,
    deriv=lambda x: -1 * np.cos(x) * np.sin(np.sin(x))
)
POLY1 = Function(
    "Polynomial",
    func=lambda x: np.power(x, 3) + 1,
    deriv=lambda x: 3 * np.power(x, 2)
)


class Newton(Fractal):

    def _computeFractal(
            self, complex_plane, itermax, f=TRIG1, a=1.0, e=1 * 10 ** -8):
        z = complex_plane

        # Matrix of number of iterations required to reach solution.
        root_iters = np.matlib.zeros(z.shape)

        for i in range(itermax):
            z = f.newtonsMethod(z, a)

            # Increment points where solutions have been found.
            roots = np.where(abs(f(z)) < e)
            root_iters[roots] += 1

        return [z.real, z.imag, root_iters]

    def _toRgbImage(self, fractal, colors, color_offset):
        """
        Converts the generated fractal into an RGB image array
        
        :return: ndarry of shape (n, m, 3)
        """
        soln_real = adjustRange(fractal[0], 0, 127)
        soln_imag = adjustRange(fractal[1], 0, 127)
        iters = adjustRange(fractal[2], 0, 128)

        rgb_image = np.array([
                soln_real + iters,
                soln_imag + iters,
                iters
            ]
        ).astype(dtype=np.uint8)

        return rgb_image.T

    def _defaultZoom(self):
        return (-3.0, 3.0, -3.0, 3.0)


# {{{ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# }}} Rendering

def adjustRange(a, vmin=0, vmax=255):
    """
    Return array with values compressed into given range.

    :type a: np.ndarray
    :param vmin: Minimum value
    :param vmax: Maximum value
    :return: Array with values ranging from vmin to vmax.
    :rtype: np.ndarray
    """
    new_a = (
        (
            # Represent array as floats ranging between 0 and 1.
            a.astype(dtype=float) / np.nanmax(a)

            # Fill given range.
            * (vmax - vmin) + vmin
        )
        # Convert back to regular array.
        .astype(dtype=np.uint8)
    )

    return new_a