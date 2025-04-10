README for ECC 2025 submission, paper "Sensitivity of Online Feedback Optimization to time-varying parameters" by M. Zagorowska and L. Imsland

Contact person: m.a.zagorowska@tudelft.nl

Introduction:
- The files are designed to be used separately, the necessary functions and packages are repeated in all of them
- It is necessary to comment/uncomment desired lines to obtain plots for responses or control inputs
- If plots are not visible after running a script, run plot!() in REPL

Output of versioninfo():
Julia Version 1.10.5
Commit 6f3fdf7b36 (2024-08-27 14:19 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Windows (x86_64-w64-mingw32)
  CPU: 32 × 13th Gen Intel(R) Core(TM) i9-13950HX
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, goldmont)
Threads: 1 default, 0 interactive, 1 GC (on 32 virtual cores)
Environment:
  JULIA_EDITOR = code
  JULIA_NUM_THREADS =

Julia setup:
1. envECC/Project.toml <- make sure that all the packages in Julia are available in the environment
2. envECC/Manifest.toml <- (optional) ideally don't touch

To reproduce the results from the paper, use:

For obtaining derivatives wrt parameters in the synthetic case study
Folder: parametersderivatives_validation
1. DerivativesAlpha_Validation.jl <- to compute derivatives (two ways) wrt alpha
2. DerivativesG_Validation.jl <- to compute derivatives (two ways) wrt G
3. Derivativesu0_Validation.jl <- to compute derivatives (two ways) wrt u0

For obtaining derivatives wrt mismatch for the gas lift case study
Folder: gaslift_mismatch
1. CharacteristicsPlots.jl <- to plot the characteristics of the five wells
2. OFO_GasLift_Derivatives.jl <- to run the case studies for unconstrained and constrained cases
3. OFO_GasLift_ValidationMismatch.jl <- to run the case with no derivatives for no mismatch and mismatch due to constant gradients
