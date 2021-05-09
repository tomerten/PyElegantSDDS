# -*- coding: utf-8 -*-

"""
Module pyelegantsdds.elegantrun 
=================================================================

A module containing the class ElegantRun to run Elegant simulations in 
a singularity container.

"""

import os
import shlex
import subprocess as subp
from io import StringIO

import numpy as np
import pandas as pd
from scipy import constants as const
