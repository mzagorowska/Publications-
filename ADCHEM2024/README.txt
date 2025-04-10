README for ADCHEM 2024 submission, paper "Tuning of Online Feedback Optimization for setpoint tracking in centrifugal compressors" by M. Zagorowska, L. Ortmann, A. Rupenyan, M. Mercangoez, L. Imsland

Contact person: marta.zagorowska@ntnu.no

Introduction:
- The files are designed to be used separately, the necessary functions and packages are repeated in all of them
- It is necessary to comment/uncomment desired lines to obtain plots for responses or control inputs
- If plots are not visible after running a script, run plot!() in REPL

Output of versioninfo():
Julia Version 1.9.0
Platform Info:
  OS: Windows (x86_64-w64-mingw32)
  CPU: 8 × 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, tigerlake)
  Threads: 1 on 8 virtual cores
Environment:
  JULIA_EDITOR = code
  JULIA_NUM_THREADS =

Julia setup:
1. envADCHEM/Project.toml <- make sure that all the packages in Julia are available in the environment
2. envADCHEM/Manifest.toml <- (optional) ideally don't touch

To reproduce the results from the paper, use:

For optimization: 
1. READY_OFOTuning_NOMAD_2param.jl <- to run the optimization in NOMAD for selected values of parameter \beta from the paper 

For manual tuning: 
1. READY_OFOTuning_TimeToSteadyState.jl <- to discover the time constant
2. READY_OFOTuning_Manual.jl <- to simulate the real system for selected values, emulating "manual tuning"

For plotting impact of parameters:
1. READY_OFOTuning_PlottingContour.jl <- to plot the contour plots with the error and oscillations as functions of \Delta T and \nu
2. READY_OFOTuning_PlottingImpact_2param.jl <- to plot the responses and the control inputs as functions of parameters of OFO

For plotting results with the tuning trajectories:
1. READY_OFOTuning_PlottingResults_2param.jl <- adjust which setpoint trajectory you need by (un)commenting certain lines

For validation for step and sine trajectory:
1. READY_OFOTuning_PlottingValidation_2param.jl <- call p1 for step, p2 for for sine setpoint AFTER running the script
