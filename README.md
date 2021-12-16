# MCG_LIGGGHTS_Source
Please replace fix_nve_sphere.cpp and fix_nve_sphere.h files in your src folder of LIGGGHTS-Public 3.8.0. 
Compile the src using "make auto" command.
The crossover planes for hopper geometry explored in our article "A particle location based multi-level coarse-graining technique for Discrete Element Method (DEM) simulation, Powder Technology" are specified in initial_integrate function of fix_nve_sphere.cpp. For any other geometry, changes are required accordingly. 
