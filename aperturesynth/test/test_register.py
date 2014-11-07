from register import find_max_correlation, extract_patches, Registrator
from numpy import testing
import numpy as np
from skimage.color import rgb2gray


class TestRegister():
    def setup(self):
        self.image = np.zeros((8, 8))
        self.image[3:5, 3:5] = 1
        self.template = np.zeros((3, 3))
        self.template[:2, :2] = 0.5
        self.colour = np.tile(self.image[..., np.newaxis], [1, 1, 3])
        patch = np.zeros((2, 2, 3))
        patches = [patch.copy(), patch.copy(), patch.copy()]
        patches[0][1, 1, :] = 1
        patches[1][0, 0, :] = 1
        patches[2][0, 1, :] = 1
        self.patches = list(map(rgb2gray, patches))
        self.windows = np.array([[2, 2], [4, 4],
                                 [4, 4], [6, 6],
                                 [4, 2], [6, 4]])

    def test_correlation(self):
        """Test the correlation template returns the correct location"""
        point = find_max_correlation(self.image, self.template)
        testing.assert_equal(point, (3, 3))

    def test_window_no_pad(self):
        """ Test the window extraction grabs correct location """
        extracted = extract_patches(self.colour, self.windows)
        print(extracted)
        testing.assert_equal(extracted, self.patches)

    def test_window_with_pad(self):
        """ Test the padding creates the correct size patch """
        extracted, windows = extract_patches(self.colour, self.windows, pad=2)
        testing.assert_equal(extracted[0].shape, [6, 6])

    def test_window_edges(self):
        """ Test that the extracted patches clip at the boundary. """
        extracted, windows = extract_patches(self.colour, self.windows, pad=10)
        testing.assert_equal(extracted[0].shape, [8, 8])

    def test_Registrator(self):
        registerer = Registrator(self.windows, self.colour, pad=1)
        matched, tform = registerer(self.colour)
        testing.assert_allclose(matched, self.colour, rtol=1e-6, atol=1e-6)

if __name__ == "__main__":
    testing.run_module_suite()
