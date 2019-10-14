#!/usr/bin/env python
"""
Package setup file for python module vision.logo
"""
import os
import setuptools
import sys


def scripts():
    """
    Get the scripts in the "scripts" directory
    Returns
    list
        List of filenames
    """
    script_list = []
    if os.path.isdir('scripts'):
        for item in os.listdir('scripts'):
            filename = os.path.join('scripts', item)
            if os.path.isfile(filename):
                script_list.append(filename)
    return script_list


def setuptools_version_supported():
    major, minor, patch = setuptools.__version__.split('.')
    if int(major) > 31:
        return True
    return False
        

if __name__ == '__main__':
    # Check for a working version of setuptools here because earlier versions did not
    # support python_requires.
    if not setuptools_version_supported():
        print('Setuptools version 32.0.0 or higher is needed to install this package')
        sys.exit(1)

    # We're being run from the command line so call setup with our arguments
    setuptools.setup(scripts=scripts())
