from setuptools import setup
from setuptools import find_packages


if __name__== '__main__':
    setup(include_package_data=True,
          name='keras_genomics',
          version='0.1.0.0',
          description='Keras Deep Learning for Genomics layers',
          author='Kundaje Lab',
          author_email='avanti.shrikumar@gmail.com',
          url='https://github.com/kundajelab/keras-genomics',
          license='MIT',
          install_requires=['keras'],
          classifiers=[
              'Development Status :: 3 - Alpha',
              'Intended Audience :: Developers',
              'Intended Audience :: Education',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: MIT License',
              'Programming Language :: Python :: 2',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.6',
              'Topic :: Software Development :: Libraries',
              'Topic :: Software Development :: Libraries :: Python Modules'
          ],
          packages=find_packages())
