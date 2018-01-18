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

    def render(self, n=300, m=300, itermax=50,
            xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0,
            colors=5, color_offset=0.5, **kwargs):
        """
        Return RGB image representing fractal.

        :param n: Pixel resolution in x-axis
        :param m: Pixel resolution in y-axis
        :param itermax: Maximum number of iterations
        :param xmin: Minimum x-coord (real number line)
        :param xmax: Maximum x-coord (real number line)
        :param ymin: Minimum y-coord (imag number line)
        :param ymax: Maximum y-coord (imag number line)
        :param colors: Number of colors (hues) included in the rendered image
        """
        complex_plane = self._complexPlane(n, m, xmin, xmax, ymin, ymax)
        fractal = self._computeFractal(complex_plane, itermax, **kwargs)
        rgb_image = self._toRgbImage(fractal, colors, color_offset)
        return rgb_image

    def _toRgbImage(self, fractal, colors, color_offset):
        """
        Converts the generated fractal into an RGB image array
        
        :return: ndarry of shape (n, m, 3)
        """
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


# {{{ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# }}} Mandelbrot Fractal

class Mandelbrot(Fractal):

    def _computeFractal(self, complex_plane, itermax, p=2, smooth=True):
        c = complex_plane
        z = np.copy(c)

        # Create matrix to represent this fractal and escaped values.
        fractal = np.matlib.zeros(c.shape, dtype=float)
        escaped = np.matlib.zeros(c.shape, dtype=bool)

        for i in xrange(itermax):
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


# {{{ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# }}} Julia Fractal

class Julia(Fractal):

    def _computeFractal(
            self, complex_plane, itermax, p=2, c=complex(0.25, -0.54)):
        z = complex_plane

        # Create matrix to represent this fractal.
        fractal = np.matlib.zeros(z.shape, dtype=float)

        for i in xrange(itermax):
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

        for i in xrange(itermax):
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


# {{{ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# }}} Rendering

def showImage(a, size=(2, 2)):
    """
    Display given RGB array.

    :type a: np.ndarray
    :param size: Scaling factor of image.
    """
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.set_cmap('hot')
    ax.imshow(a, aspect='equal')
    plt.show()


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