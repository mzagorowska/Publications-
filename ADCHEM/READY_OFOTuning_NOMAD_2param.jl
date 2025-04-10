using OrdinaryDiffEq, Plots, Zygote, LinearAlgebra, OSQP
using SparseArrays
# using Interpolations
using Integrals
using NOMAD
using Memoization
# using Random

function realSystem(initValue, tspan, contr)
    x_0 = initValue
    testF = t -> contr
    p = testF

    #Define the problem
    function compressor!(du, u, p, t)

        ctrl1 = p

        #####Parameters
        ## Physics
        SpeedSound = 340      # speed of sound
        In_pres = 1.05e5   # input pressure
        Out_pres = 1.55e5  # output pressure

        Valve_in_gain = (1 / (sqrt(1.05e5 - 1e5))) * 4e1 * 4    # opening gain - inlet
        Valve_out_gain = (1 / (sqrt(1.2891e5 - 1.2e5))) * 4e1  # opening gain - outlet

        VolumeT1 = 50e-2  # volume of the first tank
        VolumeT2 = 45e-3  # volume of the second tank

        AdivL = 0.2e-3   # Duct cross section divided by length

        ## Compressor
        Dt = 0.18   # impeller diameter
        sigma_slip = 0.9    # compressor slip
        J = 10   # shaft inertia
        pr_coeff = [2.690940401290303 -0.013878128060951 -0.040925719808930 0.000986961896765 -0.000418575028867 0.000024527875520]

        ## Saturation conditions
        satTorqueVector = [73.3235171311808 0.0724497813744527 0.00517720515117661 6.41760174259480e-07]

        #Controllers
        SpeedControlNoSat(u) = ctrl1(t)

        # #Saturation
        SpeedControl(u) = satTorqueVector[1] / (satTorqueVector[2] + exp(-satTorqueVector[3] * SpeedControlNoSat(u))) + satTorqueVector[4]
        #Algebraic variables
        m_in(u) = Valve_in_gain * 0.4541 * sqrt(abs(In_pres - u[1]))
        m_out(u) = Valve_out_gain * 0.8 * sqrt(abs(u[2] - Out_pres))
        omega_rpm(u) = u[4] / (2 * pi / 60)
        omega_percent(u) = omega_rpm(u) / 8370 * 100
        p_ratio(u) = pr_coeff[1] + pr_coeff[2] * u[3] + pr_coeff[3] * omega_percent(u) + pr_coeff[4] * u[3] * omega_percent(u) + pr_coeff[5] * u[3] * u[3] + pr_coeff[6] * omega_percent(u)^2
        torque_comp(u) = sigma_slip * (Dt / 2)^2 * u[4] * u[3]

        du[1] = SpeedSound / VolumeT1 * (m_in(u)[1] - u[3])
        du[2] = SpeedSound / VolumeT2 * (u[3] - m_out(u)[1])
        du[3] = AdivL * (p_ratio(u) * u[1] - u[2])
        du[4] = 1 / J * (SpeedControl(u) - torque_comp(u))
    end
    #Pass to solvers
    prob = ODEProblem(compressor!, x_0, tspan, p)
    sol = OrdinaryDiffEq.solve(prob, Rosenbrock23())

    return sol #####We return a measurement after a given tspan
end

function hMap(u, d)
    m = d[1]
    pr_coeff = [2.690940401290303 -0.013878128060951 -0.040925719808930 0.000986961896765 -0.000418575028867 0.000024527875520]
    Dt = 0.18   # impeller diameter
    sigma_slip = 0.9    # compressor slip

    omega_rad = 1.0 / (sigma_slip * (Dt / 2.0)^2 * m) * u
    omega_rpm = omega_rad / (2 * pi / 60)
    omega_percent = omega_rpm / 8370 * 100
    p_ratio = pr_coeff[1] + pr_coeff[2] * m + pr_coeff[3] * omega_percent + pr_coeff[4] * m * omega_percent + pr_coeff[5] * m^2 + pr_coeff[6] * omega_percent^2

    y = (1.05e5 - ((1 / (sqrt(1.2891e5 - 1.2e5))) * 4e1 * 0.8 / ((1 / (sqrt(1.05e5 - 1e5))) * 4e1 * 4 * 0.4541))^2 * 1.55e5) / (1 - ((1 / (sqrt(1.2891e5 - 1.2e5))) * 4e1 * 0.8 / ((1 / (sqrt(1.05e5 - 1e5))) * 4e1 * 4 * 0.4541))^2 * p_ratio)
    return y
end

function Φ(y, u, setp)
    z = (1e-2 * (y[1] - setp))^2
    return z
end

function internalProjectionFO(α, ucurrent, ycurrent, dcurrent, setp, Gval)
    ycurrent = ycurrent[1]
    ∇h = u -> jacobian(x -> hMap(x, dcurrent), u)
    HmatT = u -> [I ∇h(u)[1]]
    ΦmatT = (a, b) -> [jacobian((y, u) -> Φ(y, u, setp), a, b)[1] jacobian((y, u) -> Φ(y, u, setp), a, b)[2]]'

    Gmat = Gval

    P = 2.0 .* sparse([0.0 0.0; 0.0 Gmat])
    q = [0.0; 0.0]

    A = sparse([-1.0 1.0])
    l = (yv, uv) -> Gmat^(-1) * HmatT(uv) * ΦmatT(yv, uv)
    r = (yv, uv) -> Gmat^(-1) * HmatT(uv) * ΦmatT(yv, uv)

    prob = OSQP.Model()
    OSQP.setup!(prob; P=P, q=q, A=A, l=vec(l(ycurrent, ucurrent)), u=vec(r(ycurrent, ucurrent)), verbose=false)

    results = OSQP.solve!(prob)
    return results

end

function OFOoptim(α, ΔT, tuningHorizon=200, ucurrent=323.6, ycurrent=x_0, dcurrent=[75.0, 194085.875534926], setp=1e5, Gval=I)

    kMax = tuningHorizon ./ ΔT - 1

    yAll = [ycurrent]
    uAll = [ucurrent]
    uAllUcstr = [ucurrent]
    yAllReal = []
    tAllReal = []
    for k = 0:kMax
        projResult = internalProjectionFO(α, ucurrent, ycurrent, dcurrent, setp(k * ΔT), Gval)
        if projResult.info.status == :Solved
            uucstr = ucurrent - α * projResult.x[1]
        else
            uucstr = ucurrent
        end

        unew = max(300, min(1000, uucstr))
        tspan = (k * ΔT, (k + 1) * ΔT)
        ySol = realSystem(ycurrent, tspan, unew)
        ycurrent = ySol.u[end]
        ucurrent = unew
        dcurrent = [ycurrent[3], ycurrent[2]]
        yAllReal = push!(yAllReal, ySol.u)
        yAll = push!(yAll, ycurrent)
        uAll = push!(uAll, ucurrent)
        uAllUcstr = push!(uAllUcstr, uucstr)
        tAllReal = push!(tAllReal, ySol.t)

    end

    return yAll, uAll, yAllReal, tAllReal

end

####################MEMOIZATION

@memoize function perfIndicators(x)
    xx = (ubnd .- lbnd) .* x .+ lbnd
    y, yy = OFOoptim(xx[1], xx[2], tuningHorizon, 323.6, x0, [x0[3], x0[2]], setpoint)[1:2:3]

    y1 = i -> y[i][1]
    y1int(s, p) = (y1(Int(floor(s / p) + 1)) - setpoint(s))^2

    probInt = IntegralProblem(y1int, 0, tuningHorizon, xx[2])
    solInt = Integrals.solve(probInt, QuadGKJL())

    yy1 = i -> yy[i][1]
    yy1intTest(s, ps) = (mapreduce(permutedims, vcat, yy1.(Int.(floor.(s ./ ps) .+ 1)))[:, 1] .- setpoint.(s))

    sT = LinRange(0, tuningHorizon - xx[2], 5000)

    yy1int(s, ps) = mapreduce(permutedims, vcat, yy1.(Int.(floor.(s ./ ps) .+ 1)))[:, 1] .- setpoint.(s)
    yy1intEval = abs.(yy1int(sT[sT.>=50], xx[2]))

    xtest = yy1int(sT, xx[2]) .<= 0.0
    intError = solInt.u
    maxOvershoot = maximum(yy1intEval)
    numOsc = count(x -> (x < 0 || x > 0), diff(xtest))
    return intError, maxOvershoot, numOsc

end

β = [150 50; 75 50; 37.5 50; 37.5 25; 18.75 25; 9 25; 9 12; 6 20; 4.5 100]

optSol = []
plotAll = []

###########################Starting values
###########Steady state for u=323.6:
x0 = [101538.78020791685, 186796.53698194958, 60.450634195325335, 647.2042336910482]

controller = 323.6

###Initial conditions for OFO
setpoint = t -> max(94000, 95000 + 5000 * sin(0.04 * t)) #####Sine setpoint
# setpoint = t -> (t>=75 && t<= 125) ? 98000 : (t>=150 ? 93000 : 95000) #####Step setpoint
# setpoint = t -> 92500 ####Constant setpoint

tuningHorizon = 200
ucurrent = controller
ycurrent = x0

######Initial guess for optimization
initP = [1e-1, 5e1] #####WORKS for step & const

α = initP[1]
ΔT = initP[2]

yAll, uAll1, yAll2, tAll2 = OFOoptim(α, ΔT, tuningHorizon, 323.6, x0, [x0[3], x0[2]], setpoint, 1)
y1 = i -> yAll[i][1]
y2 = i -> yAll[i][2]
y3 = i -> yAll[i][3]
y4 = i -> yAll[i][4]

y1intp = x -> y1(Int(floor(x / ΔT) + 1))
y2intp = x -> y2(Int(floor(x / ΔT) + 1))
y3intp = x -> y3(Int(floor(x / ΔT) + 1))
y4intp = x -> y4(Int(floor(x / ΔT) + 1))
u1intp = x -> uAll1[Int(floor(x / ΔT) + 1)]

plot([0:0.01:tuningHorizon], y1intp.(0:0.01:tuningHorizon), ls=:dot, linetype=:steppre, lc=:red, lw=2)
plot!([0:0.01:tuningHorizon], setpoint.(0:0.01:tuningHorizon), lw=2, lc=:black, ls=:dash, leg=false)

plot!(ylabel="Suction pressure [bar]", xlabel="Time [s]")

#######################
#####Lower and upper bounds
lbnd = [0, 5e-3]
ubnd = [1e3, tuningHorizon / 2 - 1]

for kb in eachrow(β)
    objTime(x) = -x[2]
    cstrError(x) = 1e-8 * perfIndicators(x)[1] - kb[1]
    cstrOsc(x) = perfIndicators(x)[3] - kb[2]

    ######NOMAD setup
    function eval_fctTimeAll(x)
        bb_outputs = [objTime(x), cstrError(x), cstrOsc(x)]
        success = true
        count_eval = true
        return (success, count_eval, bb_outputs)
    end

    ######Initial guess scaled
    initPNomad = 1.0 ./ (ubnd .- lbnd) .* initP .- lbnd ./ (ubnd .- lbnd)

    #     ################################ NOMAD SETUP

    pb = NomadProblem(2, # number of inputs of the blackbox
        3, # number of outputs of the blackbox
        ["OBJ", "PB", "PB"], # type of outputs of the blackbox
        eval_fctTimeAll;
        lower_bound=[0.0, 0],
        upper_bound=[1.0, 1.0],
        min_mesh_size=[1e-3, 1e-3],
        initial_mesh_size=[1e-2, 1e-2]
    )

    pb.options.display_degree == 10
    pb.options.max_bb_eval = 1000 # total number of evaluations
    pb.options.quad_model_search = false
    pb.options.speculative_search = false
    pb.options.nm_search = false

    result = NOMAD.solve(pb, initPNomad)

    push!(optSol, result)
    # #######################Optimized values
    if isnothing(result.x_best_inf)
        α, ΔT = (ubnd .- lbnd) .* result.x_best_feas .+ lbnd

        yAll, uAll1, yAll2, tAll2 = OFOoptim(α, ΔT, tuningHorizon, 323.6, x0, [x0[3], x0[2]], setpoint, I)
        y1 = i -> yAll[i][1]
        y2 = i -> yAll[i][2]
        y3 = i -> yAll[i][3]
        y4 = i -> yAll[i][4]

        y1intp = x -> y1(Int(floor(x / ΔT) + 1))
        y2intp = x -> y2(Int(floor(x / ΔT) + 1))
        y3intp = x -> y3(Int(floor(x / ΔT) + 1))
        y4intp = x -> y4(Int(floor(x / ΔT) + 1))
        u1intp = x -> uAll1[Int(floor(x / ΔT) + 1)]

        plot!([0:0.01:tuningHorizon], y1intp.(0:0.01:tuningHorizon), linetype=:steppre, lw=2, label=raw"β=" * string(kb))
    else
        break
    end
end
plot!(legend=true)