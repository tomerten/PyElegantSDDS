# -*- coding: utf-8 -*-

"""
Module pyelegantsdds.sdds 
=================================================================

A module containing the SDDSCommand class and  SDDS class.

"""

import os
import shlex
import subprocess as subp
from io import StringIO
from itertools import count
from typing import Dict, List

import numpy as np
import pandas as pd
from dask import dataframe as dd

from .tools.sddsutils import sddsconvert2ascii, sddsconvert2binary


class SDDSCommand:
    _COMMANDLIST = [
        "plaindata2sdds",
        "sdds2stream",
        "sddsconvert",
        "sddsquery",
        "sddsprocess",
        "sddsplot",
        "sddsprintout",
        "sddsmakedataset",
        "sddsnaff",
    ]

	def __init__(self, sif):
        self.sif = sif
        self.command = {}


	 def createCommand(self, command, note="", **params):
        """
        Method to add a command to the command file.
        Generates a dict to reconstruct command string,
        that can be added to the command file.

        Parameters:
        ----------
        command     : str
            valid Elegant command
        note        : str
			info
        
		Key Parameters:
		---------------
		params : dict
			additional parameters for the command

		Returns:
		--------
		None
			updates the command property
        """
        # check if it is a valid Elegant command
        if self.checkCommand(command) == False:
            print("The command {} is not recognized.".format(command))
            return

        # init command dict
        thiscom = {}
        thiscom["NAME"] = command

        # add command parameters to the command dict
        for k, v in params.items():
            thiscom[k] = v

        # add the command dict to the command list
        self.command = thiscom

	def checkCommand(self, typename):
        """
        Check if a command is a valid
        Elegant command.

        Parameters:
        -----------
        typename    : str
            command type

		Returns:
		--------
		ok : boolean

        """
        for tn in self._COMMANDLIST:
            if tn == typename.lower():
                return True
        return False
	
	def clearCommand(self):
        """
        Clear the command list.
        """
        self.command = {}

	def getCommand(self, command, **params):
		"""Method to get the command.count()

		Parameters
		----------
		command : dict
			command and arguments in dictionary form

		Returns
		-------
		str
			command string to be executed 
		"""
		# create command
        self.createCommand(command, **params)

		# init command string
        cmdstr = "{} {}".format(self.sif, self.command["NAME"])

		# build the command string
        for k, v in params.items():
            if "file" in k.lower():
                cmdstr += " {}".format(v)
            elif "separator" in k.lower():
                cmdstr += ' "-separator= {}"'.format(v)
            else:
                if v is not None:
                    cmdstr += " -{}={}".format(k.split("_")[0], v)
                else:
                    cmdstr += " -{}".format(k.split("_")[0])

        return cmdstr

	def runCommand(self, commandstring):
		"""Run the command.

		Parameters
		----------
		commandstring : str
			command string to be exectured
		"""
        print("Running command {}".format(commandstring))
        subp.run(commandstring, check=True, shell=True)

	def get_particles_plain_2_SDDS_command(self, **params):
        """
        Returns sdds command to turn plain data table, separated
        by a *space* into SDDS particle initial coordinates SDDS file.
        Easy to generate plain data with pandas.

        Example:
            pd.DataFrame([
            {
                'x':0.000,
                'px':1.
            },
             {
                'x':5.000,
                'px':1.
            }
            ]).to_csv('testplain.dat',sep=' ',header=None, index=False)

        Parameters:
        ----------
        params      :
            - outputMode: set ascii for serial elegant and binary for Pelegant

        """
        return self.getCommand(
            "plaindata2sdds",
            file_1=params.get("file_1", "temp_plain_particles.dat"),
            file_2=params.get("file_2", "temp_particles_input.txt"),
            inputMode=params.get("inputMode", "ascii"),
            outputMode=params.get("outputMode", "ascii"),
            separator=" ",
            column_1="x,double,units=m",
            column_2="xp,double",
            column_3="y,double,units=m",
            column_4="yp,double",
            column_5="t,double,units=s",
            column_6='p,double,units="m$be$nc"',
            columns_7="particleID,long",
            noRowCount=None,
        )
