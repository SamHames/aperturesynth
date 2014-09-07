from setuptools import setup

setup(name='Aperture Synthesis',
      version='0.0.1',
      packages=['aperturesynth'],
      license='MIT',
      description='A tool for registering and combining series of handheld photographs',
      long_description=open('README.rst').read(),
      author='Sam Hames',
      author_email='samuel.hames@uqconnect.edu.au',
      install_requires=['scikit-image',
                        'matplotlib',
                        'numpy',
                        'docopt'],
      entry_points={'console_scripts':['aperture-synthesis=aperturesynth.synthesise:main']
                      }
                      )