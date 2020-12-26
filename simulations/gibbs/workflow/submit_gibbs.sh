#!/bin/bash

   for scale in 1, 2, 5

    do

   for n in 0, 1, 2, 3
  
    do


   for l in 2, 5, 10

    do



sbatch --export arg1=$scale,arg2=$n,arg3=$l  gibbs.cmd

done
done
done
