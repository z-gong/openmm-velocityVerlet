from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform

openmm_dir = '@OPENMM_DIR@'
velocityverletplugin_header_dir = '@VELOCITYVERLETPLUGIN_HEADER_DIR@'
velocityverletplugin_library_dir = '@VELOCITYVERLETPLUGIN_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = []
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-std=c++11', '-mmacosx-version-min=10.7']
    extra_link_args += ['-std=c++11', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_velocityverletplugin',
                      sources=['VelocityVerletPluginWrapper.cpp'],
                      libraries=['OpenMM', 'OpenMMVelocityVerlet'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), velocityverletplugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), velocityverletplugin_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='velocityverletplugin',
      version='1.0',
      py_modules=['velocityverletplugin'],
      ext_modules=[extension],
     )
