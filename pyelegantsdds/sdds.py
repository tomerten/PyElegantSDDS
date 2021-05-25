# -*- coding: utf-8 -*-

"""
Module pyelegantsdds.sdds 
=================================================================

A module containing the SDDSCommand class and  SDDS class.

The SDDSCommand class is responsible for creating, editing, etc ... of SDDS
commands.

The SDDS class uses the SDDSCommand class to build template and composed commands.

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


# ================================================================================
# SDDSCommand class:
#
#
# ================================================================================
class SDDSCommand:
    """SDDSCommand manupilation class"""

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
        "sddsresdiag",
    ]

    def __init__(self, sif):
        self.sif = sif
        self.command = {}

    def createCommand(self, command: str, note: str = "", **params) -> None:
        """
        Method to add a command to the command file.
        Generates a dict to reconstruct command string,
        that can be added to the command file.

        Parameters:
        ----------
        command     : str
        note        : str

        Key Parameters:
        ---------------
        params : dict

        Returns:
        --------
        None
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

    def checkCommand(self, typename: str) -> bool:
        """
        Check if a command is a valid
        Elegant command.

        Parameters:
        -----------
        typename    : str

        Returns:
        --------
        boolean

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
        """Method to get the command.

        Parameters
        ----------
        command : dict

        Returns
        -------
        str
        command string
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

    def runCommand(self, commandstring=None):
        """Run the command.

        Parameters
        ----------
        commandstring : str
        """
        if commandstring is None:
            name = self.command.pop("NAME")
            commandstring = self.getCommand(name, **self.command)
        print("Running command {}".format(commandstring))
        subp.run(commandstring, check=True, shell=True)

    def get_particles_plain_2_SDDS_command(self, **params) -> str:
        """
        Returns sdds command to turn plain data table, separated
        by a *space* into SDDS particle initial coordinates SDDS file.
        Easy to generate plain data with pandas.

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


# ================================================================================
# SDDS CLASS
# ================================================================================


class SDDS:
    """
    Class for interacting with SDDS files.
    """

    def __init__(self, sif: str, filename: str, filetype: int):
        assert filetype in [0, 1]
        self.sif = sif
        self._filetype = filetype
        self._filename = filename
        self.columnlist = None
        self.parameterlist = None
        self.commandlist = []
        self.command_history = {}
        self._count_iter = count(start=0, step=1)

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, value):
        self._filename = value

    @property
    def filetype(self):
        return self._filetype

    @filetype.setter
    def filetype(self, value):
        assert value in [0, 1]
        self._filetype = value

    def addCommand(self, command: str, **params) -> None:
        """Add a command to the command list.

        Parameters
        ----------
        command : str
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        valid sdds command
        """
        sddscommand = SDDSCommand(self.sif)
        cmdstr = sddscommand.getCommand(command, **params)

        # add command to current commandlist
        self.commandlist.append(cmdstr)

    def clearCommandList(self, save: bool = True) -> None:
        """
        Clear the command list.

        Parameters:
        -----------
        bool: save
        """
        if save and len(self.commandlist) > 0:
            self._addHistory(self.commandlist)

        self.commandlist = []

    def clearHistory(self):
        """
        Clear command history.
        """
        self.command_history = {}
        self._count_iter = count(start=0, step=1)

    def _addHistory(self, cmdlist):
        """
        Add commands to history.

        Parameters:
        -----------
        list | str : cmdlist
        """
        _key = next(self._count_iter)
        if not isinstance(cmdlist, list):
            cmdlist = [cmdlist]
        _value = cmdlist
        self.command_history[_key] = _value

    def runCommand(self):
        """
        Run all the sddscommands in the command list.
        """
        # print info
        if len(self.commandlist) > 0:
            # add commands to history
            self._addHistory(self.commandlist)

            for cmd in self.commandlist:
                # run command
                print("Executing : \n{}".format(cmd))

                p = subp.Popen(cmd, stdout=subp.PIPE, shell=True)
                (output, err) = p.communicate()
                p_status = p.wait()

        else:
            print("No commands entered - nothing to do!")

        # clear the commandlist
        self.clearCommandList()

    def printHistory(self):
        """Print command history."""
        if len(self.command_history) > 0:
            for i, l in self.command_history.items():
                print("History key: {}\n".format(i))
                print("---------------\n")
                for c in l:
                    print("{}\n".format(c))

                # print extra newline
                print("\n")
        else:
            print("History is empty.")

    def reload_from_history(self, history_idx: int) -> None:
        """Reloads the commandlist from history with index
        history_idx.

        Parameters
        ----------
        history_idx : int
                                        index of history command to reload
                                        use `printHistory()` to get an overview
        """
        self.clearCommandList(save=True)
        self.commandlist = self.command_history.get(history_idx)

    def rerun_from_history(self, history_idx):
        """Rerun a history entry.count()

        Parameters
        ----------
        history_idx : int
        """
        assert history_idx in list(self.command_history.keys())

        self.clearCommandList(save=True)
        self.commandlist = self.command_history.get(history_idx)
        self.runCommand()

    def load_raw_data(self):
        """Read the data from file."""
        if self.filetype == 1:
            # ASCII FORMAT
            with open(self.filename, "r") as f:
                self.raw_content = f.read()

        else:
            # BINARY FORMAT
            with open(self.filename, "rb") as f:
                self.raw_content = f.read()

    def convert(self, outfile: str = None):
        """Convert between filetypes.

        Parameters
        ----------
        outfile : str
        """
        if self.filetype == 0:
            converted_filename = self.filename + ".txt"

            if outfile is not None:
                converted_filename = outfile

            cmdstr = "{} sddsconvert -ascii {} {}".format(
                self.sif, self.filename, converted_filename
            )
        else:
            converted_filename = self.filename + ".bin"

            if outfile is not None:
                converted_filename = outfile

            cmdstr = "{} sddsconvert -binary {} {}".format(
                self.sif, self.filename, converted_filename
            )
        # add to command history
        self._addHistory(cmdstr)

        # reset filename
        print("Warning - auto filename set")
        print("Changed from {} to {}".format(self.filename, converted_filename))
        self.filename = converted_filename

        # reset filetype
        print("Warning - auto filetype set")
        print("Changed from {} to {}".format(self.filetype, abs(1 - self.filetype)))
        self.filetype = abs(1 - self.filetype)

        with open(os.devnull, "w") as f:
            subp.call(shlex.split(cmdstr), stdout=f)

    def getColumnList(self):
        """Get a list of columns available in sdds file.

        Returns
        -------
        list
        """
        # command string
        cmdstr = "{} sddsquery -columnList {}".format(self.sif, self.filename)

        # add to command history
        self._addHistory(cmdstr)

        # run the command
        p = subp.Popen(cmdstr, stdout=subp.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()

        # set the columnlist
        self.columnlist = [l.decode("utf-8") for l in output.splitlines()]

        return self.columnlist

    def getColumnValues(self, memory_threshold=100e6):
        """Get column values from file and return either
        as dataframe or return filename of converted data.

        Parameters
        ----------
        memory_threshold : float, optional

        Returns
        -------
        str | pandas.DataFrame
        """
        # make sure we have the column names
        if self.columnlist is None:
            self.getColumnList()

        # data can be large (bigger than memory)
        # use threshold to preve
        if os.path.getsize(self.filename) > memory_threshold:
            print(
                "File is large, output redirected to file {}".format(
                    self.filename + "_columnvalues.dat"
                )
            )

            # build command string
            cmdstr = "{} sdds2stream -col={} {} > {}".format(
                self.sif,
                ",".join(self.columnlist),
                self.filename,
                self.filename + "_columnvalues.dat",
            )

            # run command
            p = subp.Popen(cmdstr, stdout=subp.PIPE, shell=True)
            (output, err) = p.communicate()
            p_status = p.wait()

            return self.filename + "_columnvalues.dat"

        else:
            # build command string
            cmdstr = "{} sdds2stream -col={} {}".format(
                self.sif, ",".join(self.columnlist), self.filename
            )

            # run command
            p = subp.Popen(cmdstr, stdout=subp.PIPE, shell=True)
            (output, err) = p.communicate()
            p_status = p.wait()

            # load file output into dataframe
            output = output.decode("utf-8")
            output = pd.read_csv(StringIO(output), names=self.columnlist, delim_whitespace=True)

            return output

    def getParameterList(self):
        """Get list of parameter names available in sdds file.

        Returns
        -------
        list
        """
        # command string
        cmdstr = "{} sddsquery -parameterList {}".format(self.sif, self.filename)

        # add to command history
        self._addHistory(cmdstr)

        # run command
        p = subp.Popen(cmdstr, stdout=subp.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()

        # set the parameter list
        self.parameterlist = [l.decode("utf-8") for l in output.splitlines()]

        return self.parameterlist

    def getParameterValues(self):
        """Get parameter values from sdds file.

        Returns
        -------
        dataframe
        """
        cmdstr = "{} sddsprintout -parameters=* -spreadsheet {}".format(self.sif, self.filename)
        p = subp.Popen(cmdstr, stdout=subp.PIPE, shell=True)
        (output, err) = p.communicate()
        p_status = p.wait()
        df = pd.read_csv(
            StringIO(output.decode("utf-8")),
            error_bad_lines=False,
            delim_whitespace=True,
            skip_blank_lines=True,
            skiprows=1,
            names=["ParameterName", "ParameterValue"],
            index_col=False,
        )

        df = df.set_index("ParameterName", drop=True)
        df = pd.to_numeric(
            df.drop(["SVNVersion", "Stage", "PreviousElementName"], errors="ignore")[
                "ParameterValue"
            ]
        )
        self.ParameterName = df
        return df

    def readParticleData(self, vary=False):
        """Read the particle tracking data.

        Parameters
        ----------
        vary : bool, optional
                                        adds a step column for distinguising between the
                                        different vary_element steps.
        Returns
        -------
        DataFrame (pandas or dask)
        """
        # if vary command was used in ele command file
        # convert data and add step column
        if vary:
            self.process_scan()

        self.getColumnList()

        # if the file type is binary convert it first
        if self.filetype == 0:
            self.convert(outfile=None)

        # try to load the column data
        df = self.getColumnValues()

        # check if data or filename is returned
        if isinstance(df, str):
            # if filename was returned lazy load the data using dask dataframe
            data = dd.read_csv(df, delimiter=" ", names=self.columnlist, header=None)
        else:
            data = df

        # if vary commmand was used group the data and add Turn column
        if vary:
            grouped = data.groupby(by="step")

            def f(group):
                return group.join(
                    pd.DataFrame({"Turn": group.groupby("particleID").cumcount() + 1})
                )

            data = grouped.apply(f)

        else:
            data["Turn"] = data.groupby("particleID").cumcount() + 1

        # return data eager or lazy depending on the datasize
        return data

    def process_scan(self):
        """Process sdds file containig particle tracking data where
        *vary_element* was used in the command file/
        """
        # add command
        self.addCommand(
            "sddsprocess",
            define="column,step,Step {}".format(self.filename),
            outfile="{}_processed.{}".format(
                self.filename.split(".")[0], self.filename.split(".")[1]
            ),
        )

        # run command
        self.runCommand()

        # cmdstr = "{} sddsprocess -define=column,step,Step {} {}_processed.{}".format(
        #    self.sif, self.filename, *self.filename.split(".")
        # )
        # print(cmdstr)
        # p = subp.Popen(cmdstr, stdout=subp.PIPE, shell=True)
        newfilename = "{}_processed.{}".format(*self.filename.split("."))
        print("Warning - auto filename set")
        print("Changed from {} to {}".format(self.filename, newfilename))

        # update the filename
        self.filename = newfilename

    def sddsplot_tunediagram(self, **kwargs):

        self.addCommand("sddsresdiag", file1="resdiag.sdds")
        self.runCommand()

        newkwargs = {
            "columnNames": "nux,nuy",
            "file": self.filename,
            "scale": "0,1,0,1",
            "graph": "sym,fill,vary=subtype",
            "order": "spect",
            "split": "col=x",
            "col": "nux,nuy",
            "file2": "resdiag.sdds",
            "sever": None,
        }
        newkwargs = {**newkwargs, **kwargs}
        self.addCommand("sddsplot", **newkwargs)

        self.runCommand()

    def sddsplot_base(self, **kwargs):
        """
        Basic sddsplot command.
        """
        sddscommand = SDDSCommand(self.sif)
        cmd = sddscommand.getCommand("sddsplot", file=self.filename, **kwargs)

        # add to command history
        self._addHistory(cmd)

        sddscommand.runCommand(cmd)

    def sddsplot(
        self,
        file=None,
        columnNames="x,xp",
        scale="0,0,0,0",
        graph="dot,vary",
        order="spectral",
        split="columnBin=particleID",
        **kwargs,
    ):
        """
        Quick standard multi particle tracking plotting
        of phase space - add columnNames to plot other
        variables.

        Colors are assigned per particle ID.

        Parameters:
        -----------
        """
        if file is not None:
            newkwargs = {
                "file": file,
                "columnNames": columnNames,
                "scale": scale,
                "graph": graph,
                "order": order,
                "split": split,
            }

            newkwargs = {**newkwargs, **kwargs}
            self.sddsplot_base(**newkwargs)
        else:
            print("File missing.")

    def sddsplot_fma(
        self,
        file=None,
        col="x,y",
        split="column=diffusionRate",
        graph="sym,vary=subtype,fill,scale=2,fill",
        order="spectral",
        **kwargs,
    ):
        newkwargs = {
            "file": file,
            "col": col,
            "graph": graph,
            "order": order,
            "split": split,
        }
        if file is None:
            newkwargs.pop("file")
        newkwargs = {**newkwargs, **kwargs}
        self.sddsplot_base(**newkwargs)

    def generate_scan_dataset(self, datasetdict):
        """
		Generates a file called "temp_scan.sdds" containing columns of values
		to be used by elegant to scan over using vary_element method.

		Arguments:
		----------
		datadict: dict
			dictionary where the keys are the column headers and values are list of values to scan over
			Note: all dict values need to have the same length
		
		Example:
		--------
		>>> datasetdc = {
			"Q1" : [1.205055,1.555550505],
			"Q2" : [-1.45000,-1.800000000],\
			"Q3D": [-2.02000,-2.020000000],\
			"Q4D": [1.408000,1.4080000000],\
			"Q3T": [0.000000,0.0000000000],\
			"Q4T": [0.000000,0.0000000000],\
			"Q5T": [0.000000,0.0000000000],\
			"S1" : [0.000000,0.0000000000],\
			"S2" : [0.000000,0.0000000000],\
			"S3D": [0.000000,0.0000000000],\
			"S4D": [0.000000,0.0000000000],\
			"S3T": [0.000000,0.0000000000],\
			"S4T": [0.000000,0.0000000000]
		}
		>>> sdds = SDDS(sif, "temp.sdds",0)
		>>> sdds.generate_scan_dataset(datasetdc)

		"""
        cmd = f"{self.sif}  sddsmakedataset {self.filename} "

        for k, v in datasetdict.items():
            cmd += f"-column={k},type=double -data=" + ",".join([str(vv) for vv in v]) + " "

        subp.run(cmd, check=True, shell=True)
        self.sdds_scan = "temp.sdds"
