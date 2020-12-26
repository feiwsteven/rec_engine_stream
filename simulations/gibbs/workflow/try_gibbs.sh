#!/bin/bash

   for scale in 1

    do

   for n in 0, 1
  
    do


   for l in 2

    do



sbatch --export arg1=$scale,arg2=$n,arg3=$l  gibbs.cmd

done
done
done
