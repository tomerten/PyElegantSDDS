#!/usr/bin/env bash
if [ $# == 0 ] ; then
   echo "usage: run_Pelegant <inputfile>"
   exit 1
fi
n_cores=`grep processor /proc/cpuinfo | wc -l`
echo The system has $n_cores cores.
n_proc=$((n_cores-1))
echo $n_proc processes will be started.
if [ ! -e ~/.mpd.conf ]; then
  echo "MPD_SECRETWORD=secretword" > ~/.mpd.conf
  chmod 600 ~/.mpd.conf
fi
mpiexec -host $HOSTNAME -n $n_proc Pelegant  $1 $2 $3 $4 $5 $6 $7 $8 $9