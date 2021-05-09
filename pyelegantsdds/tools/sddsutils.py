# -*- coding: utf-8 -*-

"""
Module pyelegantsdds.tools.sddsutils
=================================================================

A module containing a list of sdds comamnds.

"""
import os
import subprocess
import time
from itertools import product

import numpy as np
import pandas as pd
from dask import dataframe as dd
from termcolor import colored


def sddsconvert2ascii(sif, filename):
    """
    Convert sdds binary file to ascii.

    Parameters:
    -----------
    sif: str
        Singularity executable container where sdds is installed.

    filename: str
        filename of the file to convert

    Returns:
    --------
    None
        A new file is created "filename.txt" in ascii format.
    """
    subprocess.run(f"{sif} sddsconvert -ascii {filename} {filename}.txt", shell=True)


def sddsconvert2binary(sif, filename):
    """
    Convert sdds ascii file to binary.

    Parameters:
    -----------
    sif: str
        Singularity executable container where sdds is installed.

    filename: str
        filename of the file to convert

    Returns:
    --------
    None
        A new file is created "filename.txt" in binary format.

    """
    subprocess.run(f"{sif} sddsconvert -binary {filename} {filename}.bin", shell=True)


def getParameterList(parameterlistfile):
    """
    Method to read back the parameter list,
    extracted with sddsextractparameternames.

    Parameters:
    -----------
    parameterlistfile: str
        Filename of the file where the parameter list is
        stored.

    Returns:
    --------
    parameterlist: List[str]
        List of the parameter names.
    """
    with open(parameterlistfile, "r") as f:
        lines = f.read().splitlines()

    return [line for line in lines]


def sddsextractparameternames(sif, elefma, ext="-001.w1"):
    """
    Extract the parameter names from an SDDS file.

    Parameters:
    -----------
    sif: str
        Singularity executable container where sdds is installed.
    elefma: str
        Elegant .ele base name (filename of the .ele file without the extension)
    ext: str
        Extension of the file, format is "elefma{ext}", usually watchpoint extension as
        in default.

    Returns:
    --------
    filename: str
        Filename where the parameter list is stored.
    parameterlist: List[str]
        List of extracted parameter names.

    """
    subprocess.run(
        f"{sif} sddsquery  -parameterList " f"{elefma}{ext} > {elefma}_parameterlist.txt",
        shell=True,
    )

    filename = f"{elefma}_parameterlist.txt"

    return filename, getParameterList(filename)


def getColumnList(columnlistfile):
    """
    Method to read back the column list,
    extracted with sddsextractcolumnnames.

    Parameters:
    -----------
    columnlistfild: str
        Filename of the file where the column list is
        stored.

    Returns:
    --------
    columnlist: List[str]
        List of the column names.
    """
    with open(columnlistfile, "r") as f:
        lines = f.read().splitlines()

    return [line for line in lines]


def sddsextractcolumnnames(sif, elefma, ext="-001.w1"):
    """
    Extract the columns names from an SDDS file.

    Parameters:
    -----------
    sif: str
        Singularity executable container where sdds is installed.
    elefma: str
        Elegant .ele base name (filename of the .ele file without the extension)
    ext: str
        Extension of the file, format is "elefma{ext}", usually watchpoint extension as
        in default.

    Returns:
    --------
    filename: str
        Filename where the column list is stored.
    columnlist: List[str]
        List of extracted column names.

    """
    subprocess.run(
        f"{sif} sddsquery -columnList " f"{elefma}{ext} > {elefma}_columnlist.txt", shell=True
    )

    filename = f"{elefma}_columnlist.txt"

    return filename, getColumnList(filename)


def sddsextractparametervalues():
    """
    Extract the parameter values from an sdds file.
    :return:
    """
    pass


def sddsextractcolumnvalues(sif, elefma, cols="x,xp,y,yp,t,p,dt,particleID", ext="-001.w1"):
    """
    Extract the column values from an sdds file.

    Parameters:
    -----------
    sif: str
        Singularity executable container where sdds is installed.
    elefma: str
        Elegant .ele base name (filename of the .ele file without the extension)
    cols: str
        String representation of the list of columns to extract the values for.
    ext: str
        Extension of the file, format is "elefma{ext}", usually watchpoint extension as
        in default.

    Returns:
    --------
    filename: str
        Filename where the values are stored.

    """
    subprocess.run(
        f"{sif} sdds2stream -col={cols} " f"{elefma}{ext} > {elefma}_particle_data.txt", shell=True
    )
    return f"{elefma}_particle_data.txt"


def processvaryelementoutput(sif, basename_binary):
    """
    Processes coordinate output file from tracking in combination
    with vary element. It adds a column "step" such that the resulting
    coordinate table can be grouped per "step" and "particleID".
    This allows to study individual tracked particles for individual vary
    steps.

    Parameters:
    -----------
    sif : str
        Singularity executable container where sdds is installed.add()
    basename_binary : str
        name of the file to be processed

    Returns:
    --------
    filename : str
        Filename where the processed data is stored.

    """
    subprocess.run(
        f"{sif} sddsprocess -define=column,step,Step {basename_binary} processed_{basename_binary}",
        check=True,
        shell=True,
    )
    return f"processed_{basename_binary}"


def generate_scan_dataset(sif, datasetdict, filepath):
    """
    Generates a file called "scan.sdds" containing columns of values
    to be used by elegant to scan over using vary_element method.

    Parameters:
    -----------
    datadict: dict
        dictionary where the keys are the column headers and values are list of values to scan over
        Note: all dict values need to have the same length

    filepath: str
        path where the simulation will be run (i.e where ele and lte files are)

    Returns:
    --------
    None
        Creates file scan.sdds in the filepath to be used in the simulations.add()

    """
    # get current working dir to be able to get back
    currdir = os.getcwd()

    # change to simulation dir
    os.chdir(filepath)
    print("File path used: {}".format(filepath))

    # create scan.sdds
    cmd = f"{sif}  sddsmakedataset scan.sdds "

    for k, v in datasetdict.items():
        cmd += f"-column={k},type=double -data=" + ",".join([str(vv) for vv in v]) + " "

    subprocess.run(cmd, check=True, shell=True)

    # change back to original working dir
    os.chdir(currdir)


def sddsplot(
    sif,
    filepath,
    columnNames=["x", "xp"],
    markerstyle="sym",
    vary="subtype",
    scalemarker=1,
    fill=True,
    order="spectral",
    split="page",
    scale="0,0,0,0",
):
    """
    Method to generate sdds plot.

    Parameters:
    -----------
    sif:
    filepath
    columnNames
    markerstyle
    vary
    scalemarker
    """
    if fill:
        strfill = ",fill"
    else:
        strfill = ""
    cmd = f"{sif} sddsplot -columnNames={','.join(columnNames)} {filepath} "
    cmd += f"-graph={markerstyle},vary={vary}{strfill},scale={str(scalemarker)} -order={order} -split={split} -scale={scale}"
    subprocess.run(cmd, check=True, shell=True)


def readParticleData(coordfile, collist, simtype="spt", vary=False):
    """ """
    data = dd.read_csv(coordfile, delimiter=" ", names=collist, header=None)
    if vary:
        grouped = data.groupby(by="step")

        def f(group):
            return group.join(pd.DataFrame({"Turn": group.groupby("particleID").cumcount() + 1}))

        data = grouped.apply(f)

    elif simtype == "spt":
        data["Turn"] = data.groupby("particleID").cumcount() + 1

    return data
