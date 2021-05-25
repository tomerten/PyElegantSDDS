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

from .elegant_command import ElegantCommandFile
from .sdds import SDDS, SDDSCommand

# ==============================================================================
#
# HELPER FUNCTIONS
#
# ==============================================================================


def write_parallel_elegant_script():
    """
    Method to generate a script that runs
    pelegant from bash.
    """

    # list of strings to write
    bashstrlist = [
        "#!/usr/bin/env bash",
        "if [ $# == 0 ] ; then",
        '   echo "usage: run_Pelegant <inputfile>"',
        "   exit 1",
        "fi",
        "n_cores=`grep processor /proc/cpuinfo | wc -l`",
        "echo The system has $n_cores cores.",
        "n_proc=$((n_cores-1))",
        "echo $n_proc processes will be started.",
        "if [ ! -e ~/.mpd.conf ]; then",
        '  echo "MPD_SECRETWORD=secretword" > ~/.mpd.conf',
        "  chmod 600 ~/.mpd.conf",
        "fi",
        "mpiexec -host $HOSTNAME -n $n_proc Pelegant  $1 $2 $3 $4 $5 $6 $7 $8 $9",
    ]

    bashstr = "\n".join(bashstrlist)

    # write to file
    with open("temp_run_pelegant.sh", "w") as f:
        f.write(bashstr)


def write_parallel_run_script(sif):
    """
    Method to generate parallel elegant run
    script.

    Parameters:
    -----------
    sif: str
                                                                                                                                                                                                                                                                    path to singularity container

    """
    bashstrlist = [
        "#!/bin/bash",
        "pele={}".format(sif),
        'cmd="bash temp_run_pelegant.sh"',
        "",
        "$pele $cmd $1",
    ]
    bashstr = "\n".join(bashstrlist)

    # write to file
    with open("run_pelegant.sh", "w") as f:
        f.write(bashstr)


def GenerateNDimCoordinateGrid(N, NPOINTS, pmin=1e-6, pmax=1e-4, man_ranges=None):
    """
    Method to generate an N dimensional coordinate grid for tracking,
    with fixed number of point in each dimension.
    The final shape is printed at creation.

    IMPORTANT:
                                                                                                                                    Number of grid points scales with N * NPOINTS**N, i.e.
                                                                                                                                    very large arrays are generated already with
                                                                                                                                    quite some small numbers for NPOINTS and N.

                                                                                                                                    Example: NPOINTS = 2, N = 6 -> 6*2*6 = 384 elements

    Parameters:
    -----------
    N: int
                                                                                                                                    dimension of the coordinate grid
    NPOINTS: int
                                                                                                                                    number of points in each dimension
    pmin: float
                                                                                                                                    min coordinate value in each dim
    pmax: float
                                                                                                                                    max coordinate value in each dim

    Returns:
    ========
    coordinate_grid : numpy array
                                                                                                                                    coordinate grid with particle ID in last column
    """
    rangelist = [np.linspace(pmin, pmax, NPOINTS)] * N
    if man_ranges is not None:
        # print(man_ranges)
        for k, v in man_ranges.items():
            rangelist[int(k)] = v
            # print(rangelist)
    grid = np.meshgrid(*rangelist)
    coordinate_grid = np.array([*grid])
    npart = coordinate_grid.size // N
    coordinate_grid = coordinate_grid.reshape(N, npart).T
    print("Shape: {} - Number of paritcles: {} ".format(coordinate_grid.shape, npart))
    # add particle id
    coordinate_grid = np.hstack((coordinate_grid, np.array(range(1, npart + 1)).reshape(npart, 1)))
    # print(coordinate_grid)

    return coordinate_grid


def generate_sphere_grid(dim=2, rmin=1e-6, rmax=1, rsteps=3, phisteps=3, **kwargs):
    """Method to generate grid point within n-dim ball, like polar but n-dim.
    Dimension 6 is a special case - as we need it for Elegant tracking. In this case
    the final two dimensions are not polar but fixed for dim 5 and in dim 6 and array
    passed via the kwargs 'deltaGamma'.

    Parameters
    ----------
    dim : int, optional
                                                                                                                                    dimension of the ball, by default 2
    rmin : float, optional
                                                                                                                                    minimal radius to use, by default 1e-6
    rmax : float, optional
                                                                                                                                    maximal radius to use, by default 1
    rsteps : int, optional
                                                                                                                                    number of steps in radius grid, by default 3
    phisteps : int, optional
                                                                                                                                    number of steps in the angle grid, by default 3
    """
    R = np.linspace(rmin, rmax, rsteps)
    mangle = np.pi

    # only track one kwadrant
    if kwargs.get("half", False):
        mangle = mangle / 2.0

    PHI1 = np.linspace(0, mangle, phisteps)
    PHI2 = np.linspace(
        0, mangle, phisteps
    )  # full sphere is 2 pi reduced for tracking to upper half

    # the special case
    if dim != 6:
        matrices = (R,) + tuple((PHI1 for _ in range(dim - 2))) + (PHI2,)
    else:
        # elegant t shift is fixed to zero
        # TODO: fix the fixed t shift
        matrices = (
            (R,)
            + tuple((PHI1 for _ in range(dim - 4)))
            + (PHI2,)
            + (np.array([0.0]), kwargs.get("deltaGamma", np.array([0.0])))
        )

    # create meshgrid to make all combinations
    meshmatrices = np.array(np.meshgrid(*matrices))

    # count the number of particles
    npart = meshmatrices.size // dim

    # reshape
    coord_T = meshmatrices.reshape(dim, npart).T

    #     X = (coord_T[:,0] * np.cos(coord_T[:,1]),)
    X = tuple()

    if dim == 6:
        ndim = 4
    else:
        ndim = dim

    for i in range(1, ndim):
        X += (coord_T[:, 0] * np.prod(np.sin(coord_T[:, 1:i]), axis=1) * np.cos(coord_T[:, i]),)

    X += (coord_T[:, 0] * np.prod(np.sin(coord_T[:, 1:-1]), axis=1) * np.sin(coord_T[:, -1]),)

    if dim != 6:
        sphere_grid = np.vstack(X)
    else:
        sphere_grid = np.vstack(X + (coord_T[:, 4], coord_T[:, 5]))
    print("Shape: {} - Number of paritcles: {} ".format(sphere_grid.T.shape, npart))

    # add particle id
    coordinate_grid = np.hstack((sphere_grid.T, np.array(range(1, npart + 1)).reshape(npart, 1)))
    # print(coordinate_grid)
    return coordinate_grid


# ==============================================================================
#
# MAIN CLASS
#
# ==============================================================================


class ElegantRun:
    """
    Class to interact with Elegant and Parallel Elegant from Python.
    """

    _REQUIRED_KWARGS = ["use_beamline", "energy"]

    def __init__(self, sif, lattice: str, parallel=False, **kwargs):
        self.sif = sif
        self.lattice = lattice
        self.parallel = parallel
        self.kwargs = kwargs
        self.check()
        self.commandfile = ElegantCommandFile("temp.ele")

        # setting up executable
        if parallel:
            self._write_parallel_script()
            self.exec = "bash {}".format(self.pelegant)
        else:
            self.exec = "{} elegant ".format(self.sif)

    def check(self):
        """
        Check if all necessary info
        is given for Elegant to be able to run.
        """
        if not all(req in self.kwargs.keys() for req in self._REQUIRED_KWARGS):
            print("Missing required kwargs...")
            print("Minimum required are:")
            for r in self._REQUIRED_KWARGS:
                print(r)

    def _write_parallel_script(self):
        """
        Generate the script to run parallel elegant.
        Method sets self.pelegant to the script file.
        """
        write_parallel_elegant_script()
        write_parallel_run_script(self.sif)
        self.pelegant = "run_pelegant.sh"

    def clearCommands(self):
        """Clear the command list."""
        self.commandfile.clear()

    def clearCommandHistory(self):
        """Clear the command history."""
        self.commandfile.clearHistory()

    def clearAll(self):
        """Clears both command list and command history."""
        self.clearCommands()
        self.clearCommandHistory()

    def run(self):
        """
        Run the commandfile.
        """
        # check if commandfile is not empty
        if len(self.commandfile.commandlist) == 0:
            print("Commandfile empty - nothing to do.")
            return

        # write Elegant command file to disk
        self.commandfile.write()

        # generate command string
        # Debug: print(self.exec)
        cmdstr = "{} temp.ele".format(self.exec)
        # Debug: print(cmdstr)

        # run
        with open(os.devnull, "w") as f:
            subp.call(shlex.split(cmdstr), stdout=f)

    def add_basic_setup(self, **kwargs):
        """
        Add basic setup command.
        """
        self.commandfile.clear()
        self.commandfile.addCommand(
            "run_setup",
            lattice=self.lattice,
            use_beamline=self.kwargs.get("use_beamline", None),
            p_central_mev=self.kwargs.get("energy", 1700.00),
            # centroid="%s.cen",
            default_order=kwargs.get("default_order", 1),
            concat_order=kwargs.get("concat_order", 3),
            rootname="temp",
            parameters="%s.params",
            semaphore_file="%s.done",
            magnets="%s.mag",  # for plotting profile
            final="%s.fin",
        )

    def add_basic_twiss(self):
        """
        Add basic twiss.
        """
        self.commandfile.addCommand(
            "twiss_output", filename="%s.twi", matched=1, radiation_integrals=1
        )

    def add_vary_element(self, **kwargs):
        """
        Add single vary element line.
        """
        self.commandfile.addCommand(
            "vary_element",
            name=kwargs.get("name", "*"),
            item=kwargs.get("item", "L"),
            intial=kwargs.get("initial", 0.0000),
            final=kwargs.get("final", 0.0000),
            index_number=kwargs.get("index_number", 0),
            index_limit=kwargs.get("index_limit", 1),
        )

    def add_vary_element_from_file(self, **kwargs):
        """
        Add single vary element line, loading value from
        dataset file.
        """
        if "enumeration_file" not in kwargs.keys():
            print("External filename missing.")
        else:
            self.commandfile.addCommand(
                "vary_element",
                name=kwargs.get("name", "*"),
                item=kwargs.get("item", "L"),
                index_number=kwargs.get("index_number", 0),
                index_limit=kwargs.get("index_limit", 1),
                enumeration_file=kwargs.get("enumeration_file"),
                enumeration_column=kwargs.get("enumeration_column"),
            )

    def add_basic_controls(self):
        """Adding basic controls for tracking"""
        # add controls
        self.commandfile.addCommand("run_control")
        self.commandfile.addCommand("bunched_beam")
        self.commandfile.addCommand("track")

    def add_watch(self, **kwargs):
        """Add watch point."""
        self.commandfile.addCommand(
            "insert_elements",
            name=kwargs.get("name", ""),
            type=kwargs.get("type", ""),
            exclude="",
            s_start=kwargs.get("s_start", -1),
            s_end=kwargs.get("s_end", -1),
            skip=kwargs.get("skip", 1),
            insert_before=kwargs.get("insert_before", 0),
            add_at_end=kwargs.get("add_at_end", 0),
            add_at_start=kwargs.get("add_at_start", 0),
            element_def=kwargs.get(
                "element_def", r'"WQ: WATCH, FILENAME=\"%s-%03ld.wq\", mode=\"coordinates\""'
            ),
        )

    def add_watch_at_start(self):
        """Add watch point at start of lattice."""
        self.add_watch(
            name="W",
            add_at_start=1,
            element_def=r'"W: WATCH, FILENAME=\"%s-%03ld.wq\", mode=\"coordinates\""',
        )

    def add_fma_command(self, **kwargs):
        """
        Add elegant standard fma command.
        """

        self.commandfile.addCommand(
            "frequency_map",
            output="%s.fma",
            xmin=kwargs.get("xmin", -0.1),
            xmax=kwargs.get("xmax", 0.1),
            ymin=kwargs.get("ymin", 1e-6),
            ymax=kwargs.get("ymax", 0.1),
            delta_min=kwargs.get("delta_min", 0),
            delta_max=kwargs.get("delta_max", 0),
            nx=kwargs.get("nx", 21),
            ny=kwargs.get("ny", 21),
            ndelta=kwargs.get("ndelta", 1),
            verbosity=0,
            include_changes=kwargs.get("include_changes", 1),
            quadratic_spacing=kwargs.get("quadratic_spacing", 0),
            full_grid_output=kwargs.get("full_grid_output", 1),
        )

    def add_DA_command(self, **kwargs):
        """
        Add DA find aperture command.
        """
        self.commandfile.addCommand(
            "find_aperture",
            output="%s.aper",
            mode=kwargs.get("mode", "n-line"),
            verbosity=0,
            xmin=kwargs.get("xmin", -0.1),
            xmax=kwargs.get("xmax", 0.1),
            xpmin=kwargs.get("xpmin", 0.0),
            xpmax=kwargs.get("xpmax", 0.0),
            ymin=kwargs.get("ymin", 0.0),
            ymax=kwargs.get("ymax", 0.1),
            ypmin=kwargs.get("ypmin", 0.0),
            ypmax=kwargs.get("ypmax", 0.0),
            nx=kwargs.get("nx", 21),
            ny=kwargs.get("ny", 11),
            n_lines=kwargs.get("n_lines", 11),
            split_fraction=kwargs.get("split_fraction", 0.5),
            n_splits=kwargs.get("n_splits", 0),
            desired_resolution=kwargs.get("desired_resolution", 0.01),
            offset_by_orbit=kwargs.get("offset_by_orbit", 0),
            full_plane=kwargs.get("full_plane", 1),
        )

    def findtwiss(self, **kwargs):
        """
        Run Twiss and return Twiss parameters
        together with Twiss data.

        Parameters:
        ----------
        kwargs  : dict
                                                                                                                                        twiss command options
        """
        # TODO: add matched = 0 case
        matched = kwargs.get("matched", 1)
        initial_optics = kwargs.get("initial_optics", [])
        alternate_element = kwargs.get("alternate_elements", {})
        closed_orbit = kwargs.get("closed_orbit", 1)

        # make sure not residual is there
        self.commandfile.clear()

        # add setup command
        self.add_basic_setup()

        # add twiss calc
        self.commandfile.addCommand(
            "twiss_output",
            matched=matched,
            output_at_each_step=0,
            filename="%s.twi",
            radiation_integrals=1,
        )

        # add controls
        self.add_basic_controls()

        # write command file
        self.commandfile.write()

        # set cmdstr and run
        cmdstr = "{} elegant temp.ele".format(self.sif)
        with open(os.devnull, "w") as f:
            subp.call(shlex.split(cmdstr), stdout=f)

        # load twiss output
        twifile = SDDS(self.sif, "temp.twi", 0)
        twiparams = twifile.getParameterValues()
        twidata = twifile.getColumnValues()

        twiparams["length"] = np.round(twidata.iloc[-1]["s"], 3)

        return twidata, twiparams

    def find_matrices(self, **kwargs):
        """
        Find element by element matrix and map elements (depending on given order).
        Constant vector and R matrix are returned as numpy arrays, the maps are
        returned as dicts.

        Parameters:
        -----------
        kwargs  :
                                                                                                                                        - SDDS_output_order : order of maps (max is 3)

        Returns:
        --------
        C       : np.array
                                                                                                                                        constant vector
        R       : np.array
                                                                                                                                        R matrix
        T_dict  : dict
                                                                                                                                        T map Tijk as key
        Q_dict  : dict
                                                                                                                                        U map Qijkl as key
        """
        assert kwargs.get("SDDS_output_order", 1) < 4

        self.commandfile.clear()
        self.add_basic_setup()
        self.commandfile.addCommand(
            "matrix_output",
            SDDS_output="%s.sdds",
            SDDS_output_order=kwargs.get("SDDS_output_order", 1),
            printout="%s.mat",
            printout_order=kwargs.get("SDDS_output_order", 1),
            full_matrix_only=kwargs.get("full_matrix_only", 0),
            individual_matrices=kwargs.get("individual_matrices", 1),
            output_at_each_step=kwargs.get("output_at_each_step", 1),
        )

        # add controls
        self.add_basic_controls()

        # write command file
        self.commandfile.write()

        # set cmdstr
        cmdstr = "{} elegant temp.ele".format(self.sif)
        with open(os.devnull, "w") as f:
            subp.call(shlex.split(cmdstr), stdout=f)

        with open("temp.mat", "r") as f:
            mdata = f.read()

        # get full turn matrix and
        dfmat = pd.read_csv(
            StringIO("\n".join(mdata.split("full", 1)[1].splitlines()[1:])),
            delim_whitespace=True,
            names=[1, 2, 3, 4, 5, 6],
        )
        C = dfmat.loc[dfmat.index == "C:"].values.T
        R = dfmat.loc[dfmat.index.str.contains("R")].values
        T = dfmat.loc[dfmat.index.str.contains("T")]
        Q = dfmat.loc[dfmat.index.str.contains("Q")]

        T_dict = {}
        for _, row in T.iterrows():
            _basekey = row.name[:-1]
            for c in T.columns:
                _key = _basekey + str(c)
                _value = row[c]
                if not pd.isna(_value):
                    T_dict[_key] = _value

        Q_dict = {}
        for _, row in Q.iterrows():
            _basekey = row.name[:-1]
            for c in Q.columns:
                _key = _basekey + str(c)
                _value = row[c]
                if not pd.isna(_value):
                    Q_dict[_key] = _value

        sddsmat = SDDS(self.sif, "temp.sdds", 0)
        ElementMatrices = sddsmat.getColumnValues()

        return C, R, ElementMatrices, T_dict, Q_dict

    def generate_sdds_particle_inputfile(self, grid_type="rectangular", **kwargs):
        """
        Generates an SDDS file containing initial
        particle coordinates on a grid. The grid
        can be defined through the kwargs.

        Parameters:
        ----------
        kwargs      :
                                        - pmin: min value of grid on each dim
                                        - pmax: max value of grid on each dim
                                        - pcentralmev: particle energy (code converts it to beta * gamma )
                                        - man_ranges: dict containing as key dim num - in order x xp y yp s p
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        and as values an array of values to be used
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        For p this is autoset to beta gamma based on pcentralmev
                                        - NPOINTS: number of linear spaced points in each dim for the grid

        Returns:
        --------
        None, writes the data to pre-defined named file.

        """
        assert grid_type in ["rectangular", "spherical"]
        pcentral = kwargs.get("pcentralmev", self.kwargs.get("energy"))
        # convert to beta * gamma
        pcentral = np.sqrt(
            (pcentral / const.physical_constants["electron mass energy equivalent in MeV"][0]) ** 2
            - 1
        )

        if grid_type == "rectangular":
            npoints_per_dim = kwargs.get("NPOINTS", 2)
            pmin = kwargs.get("pmin", 0)
            pmax = kwargs.get("pmax", 1e-4)
            man_ranges = kwargs.get("man_ranges", {"5": np.array([pcentral])})
            if "5" not in man_ranges.keys() and 5 not in man_ranges.keys():
                man_ranges["5"] = np.array([pcentral])
            # example : man_ranges={'0':np.array([1e-6,1e-5]),'1':[0]})

            # generate coordinate grid, with particle id as last column
            # and save it as plain data table seperated by a whitespace
            particle_df = pd.DataFrame(
                GenerateNDimCoordinateGrid(
                    6, npoints_per_dim, pmin=pmin, pmax=pmax, man_ranges=man_ranges
                )
            )
            particle_df.to_csv("temp_plain_particles.dat", sep=" ", header=None, index=False)

            # cleanup kwargs
            kwargs.pop("NPOINTS", None)
            kwargs.pop("pmin", None)
            kwargs.pop("pmax", None)
            kwargs.pop("man_ranges", None)
        else:
            rmin = kwargs.get("rmin", 1e-6)
            rmax = kwargs.get("rmax", 1e-1)
            rsteps = kwargs.get("rsteps", 3)
            half = kwargs.get("half", True)
            phisteps = kwargs.get("phisteps", 5)
            deltaGamma = kwargs.get("deltaGamma", np.array([pcentral]))

            particle_df = pd.DataFrame(
                generate_sphere_grid(
                    dim=6,
                    rmin=rmin,
                    rmax=rmax,
                    rsteps=rsteps,
                    phisteps=phisteps,
                    deltaGamma=deltaGamma,
                    half=half,
                )
            )

            particle_df.to_csv("temp_plain_particles.dat", sep=" ", header=None, index=False)
            # clean up kwargs
            kwargs.pop("rmin", None)
            kwargs.pop("rmax", None)
            kwargs.pop("rsteps", None)
            kwargs.pop("half", None)
            kwargs.pop("phisteps", None)
            kwargs.pop("deltaGamma", None)

        kwargs.pop("pcentralmev", None)

        # Create sddscommand object
        sddscommand = SDDSCommand(self.sif)

        # update the command parameters
        if self.parallel:
            outputmode = "binary"
        else:
            outputmode = "ascii"
        kwargs["outputMode"] = outputmode
        kwargs["file_2"] = (
            "temp_particles_input.txt" if not self.parallel else "temp_particles_input.bin"
        )

        # load the pre-defined  convert plain data to sdds command
        cmd = sddscommand.get_particles_plain_2_SDDS_command(**kwargs)

        # run the sdds command
        sddscommand.runCommand(cmd)

        self.sdds_beam_file = kwargs["file_2"]

    def simple_single_particle_track(self, coord=np.zeros((5, 1)), **kwargs):
        """
        Track a single particle with given initial coordinates.

        Important:
        ----------
        Be careful with giving the 6th coordinate, this is beta * gamma. If not
        given it will be calculated automatically either using standard 1700 MeV
        or kwargs["pcentralmev"].

        """
        # generate particle input file
        self.generate_sdds_particle_inputfile(
            man_ranges={k: v for k, v in zip(range(coord.shape[0] + 1), coord)}, **kwargs
        )

        # construct command file
        self.commandfile.clear()
        self.add_basic_setup()
        self.commandfile.addCommand("run_control", n_passes=kwargs.get("n_passes", 2 ** 8))
        self.commandfile.addCommand("bunched_beam")
        self.commandfile.addCommand(
            "sdds_beam",
            input=self.sdds_beam_file,
            input_type='"elegant"',
        )
        self.commandfile.addCommand("track")

        # run will write command file and execute it
        self.run()

    def track_simple(self, **kwargs):
        """
        Track a set of particles.
        """
        # construct command file
        self.commandfile.clear()
        self.add_basic_setup()
        self.commandfile.addCommand("run_control", n_passes=kwargs.get("n_passes", 2 ** 8))
        self.commandfile.addCommand("bunched_beam")
        self.commandfile.addCommand(
            "sdds_beam",
            input=self.sdds_beam_file,
            input_type='"elegant"',
        )
        self.commandfile.addCommand("track")

        # run will write command file and execute it
        self.run()

    def track_vary(
        self, varydict: dict, varyitemlist=None, mode="row", add_watch_start=False, **kwargs
    ):
        """
        Track a set of particles in combination with a
        vary command.
        """
        assert varyitemlist is not None
        assert len(varyitemlist) == len(varydict)
        assert mode.lower() in ["row", "table"]

        # generate the sdds input file
        sdds = SDDS(self.sif, "temp.sdds", 0)
        sdds.generate_scan_dataset(varydict)

        self.commandfile.clear()
        self.add_basic_setup()
        if add_watch_start:
            self.add_watch_at_start()
        n_idx = 1 if mode == "row" else len(varydict)
        self.commandfile.addCommand(
            "run_control", n_indices=n_idx, n_passes=kwargs.get("n_passes", 2 ** 8)
        )
        if mode == "table":
            for i, it in enumerate(varydict.items()):
                k, v = it
                self.add_vary_element_from_file(
                    name=k,
                    item=varyitemlist[i],
                    index_number=i,
                    index_limit=len(v),
                    enumeration_file="temp.sdds",
                    enumeration_column=k,
                )
        else:
            for i, it in enumerate(varydict.items()):
                k, v = it
                self.add_vary_element_from_file(
                    name=k,
                    item=varyitemlist[i],
                    index_number=0,
                    index_limit=len(v),
                    enumeration_file="temp.sdds",
                    enumeration_column=k,
                )
        self.commandfile.addCommand("bunched_beam")
        self.commandfile.addCommand(
            "sdds_beam", input=self.sdds_beam_file, input_type='"elegant"', reuse_bunch=1
        )
        self.commandfile.addCommand("track")
        self.run()

    def fma(self, **kwargs):
        """
        Run Elegant fma.
        """
        self.commandfile.clear()
        self.add_basic_setup()

        self.commandfile.addCommand("run_control", n_passes=kwargs.pop("n_passes", 2 ** 8))
        self.add_basic_twiss()
        self.add_fma_command(**kwargs)

        self.run()

    def dynap(self, **kwargs):
        """
        Run Elegant's Dynamic Aperture.
        """
        self.commandfile.clear()
        self.commandfile.addCommand(
            "run_setup",
            lattice=self.lattice,
            use_beamline=self.kwargs.get("use_beamline", None),
            p_central_mev=self.kwargs.get("energy", 1700.00),
            centroid="%s.cen",
            default_order=kwargs.get("default_order", 2),
            concat_order=kwargs.get("concat_order", 3),
            rootname="temp",
            parameters="%s.params",
            semaphore_file="%s.done",
            magnets="%s.mag",  # for plotting profile
            losses="%s.los",
        )

        self.commandfile.addCommand("twiss_output", filename="%s.twi", output_at_each_step=1)
        self.commandfile.addCommand("run_control", n_passes=kwargs.pop("n_passes", 2 ** 9))
        self.add_DA_command(**kwargs)

        self.run()

    def dynapmom(self):
        """
        Run Elegant's Dynamic Momentum Aperture.
        """
        # TODO
        pass

    def table_scan(self, scan_list_of_dicts, mode="row", add_watch_start=True, **kwargs):
        """ """
        assert mode in ["row", "table"]
        n_idx = 1 if mode == "row" else len(scan_list_of_dicts)
        print(n_idx)
        self.commandfile.clear()
        self.add_basic_setup()
        if add_watch_start:
            self.add_watch_at_start()
        self.commandfile.addCommand(
            "run_control", n_indices=n_idx, n_passes=kwargs.get("n_passes", 2 ** 8)
        )
        for i, l in enumerate(scan_list_of_dicts):
            if mode == "table":
                inx = i
            else:
                inx = 0
            self.commandfile.addCommand(
                "vary_element",
                name=l.get("name"),
                item=l.get("item"),
                initial=l.get("initial"),
                final=l.get("final"),
                index_number=inx,
                index_limit=l.get("index_limit"),
            )
        self.commandfile.addCommand(
            "sdds_beam", input=self.sdds_beam_file, input_type='"elegant"', reuse_bunch=1
        )
        self.commandfile.addCommand("track")

        # run will write command file and execute it
        self.run()
