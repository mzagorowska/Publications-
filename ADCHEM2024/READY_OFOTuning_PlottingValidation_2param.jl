using OrdinaryDiffEq, Plots, Zygote, LinearAlgebra, OSQP
using SparseArrays
using Interpolations
using Integrals
using Memoization
using Random

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
        m_in(u) = Valve_in_gain * 0.4541 * sqrt(abs(In_pres - u[1]))  #######Disturbance in inlet valve opening
        m_out(u) = Valve_out_gain * 0.8 * sqrt(abs(u[2] - Out_pres)) #######Disturbance in outlet valve opening
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

    omega_rad = 1 / (sigma_slip * (Dt / 2)^2 * m) * u
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
    Gmat = Gval ####Was I

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

    kMax = tuningHorizon ./ ΔT

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
            uucstr = ucurrent# + α*projResult.x[1]
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

@memoize function perfIndicators(x, x0, horizon)
    xx = (ubnd .- lbnd) .* x .+ lbnd
    y, yy = OFOoptim(xx[1], xx[2], horizon, 323.6, x0, [75.0, 194085.875534926], setpoint, I)[1:2:3]
    y1 = i -> y[i][1]
    y1int(s, p) = (y1(Int(floor(s / p) + 1)) - setpoint(s))^2

    probInt = IntegralProblem(y1int, 0, horizon, xx[2])
    solInt = Integrals.solve(probInt, QuadGKJL())

    yy1 = i -> yy[i][1]
    yy1intTest(s, ps) = (mapreduce(permutedims, vcat, yy1.(Int.(floor.(s ./ ps) .+ 1)))[:, 1] .- setpoint.(s))

    sT = LinRange(0, horizon - xx[2], 5000)

    yy1int(s, ps) = mapreduce(permutedims, vcat, yy1.(Int.(floor.(s ./ ps) .+ 1)))[:, 1] .- setpoint.(s)
    yy1intEval = abs.(yy1int(sT[sT.>=0], xx[2]))
    xtest = yy1int(sT, xx[2]) .<= 0.0

    intError = solInt.u
    maxOvershoot = maximum(yy1intEval)
    numOsc = count(x -> (x < 0 || x > 0), diff(xtest))

    return intError, maxOvershoot, numOsc

end

##############
#Initial Conditions for the compressor
x0 = [100745.298145799, 194085.875534926, 75.0225656320447, 662.307900948556]

controller = 323.6;

###Initial conditions for OFO
tuningHorizon = 200
ucurrent = controller
ycurrent = x0

######Initial guess for optimization
initP = [1e-1, 5e1] #####WORKS
#####Lower and upper bounds
lbnd = [0, 5e-3]
ubnd = [1e3, tuningHorizon / 2 - 1]
######Initial guess scaled

#######Validation
allVal = [
    0.4681 0.05833550633869 #####For constant tuning
    0.2071 0.13002550633869 #####Best error for step tuning
    0.1991 0.8080255063387#####Best error for sine tuning
    0.15 0.4797717056416991 #####Manual tuning
]

#         #####################Different setup for validation
x0 = [91745, 200085.0, 80.0, 700.5]

p1All = []
p2All = []
tuningHorizon = 1000
NUMB = 100.0
rng = MersenneTwister(1234)
xs = 0:NUMB:tuningHorizon

xss = vec(sort!(tuningHorizon * rand!(rng, zeros(length(xs)))))
A = vec((100000 - 92000) * rand!(rng, zeros(length(xs))) .+ 92000)

itsp = scale(interpolate(A, BSpline(Constant())), xs)

nodes = (xss,)
itsps = extrapolate(interpolate(nodes, A, Gridded(Constant())), Flat())
setpoint = t -> itsps(t)
p1 = plot() ###For responses
p2 = plot()
# p1 =plot!(p1,[0,1000],[1000,1000],lc=:black,ls=:dash,lw=2,label=:none)  #####For control
# p2 =plot!(p2,[0,1000],[1000,1000],lc=:black,ls=:dash,lw=2,label=:none)

plot!(p1, [0:0.01:tuningHorizon], setpoint.(0:0.01:tuningHorizon) * 1e-5, lw=2, lc=:black, ls=:dash, label="Setpoint") #####For responses
plot!(p1, ylabel="p_s [bar]", xlabel="Time [s]")
setpoint = t -> max(88000, min(97500, max(91000, 95000 + 5000 / 4 * (sin(0.009 * t) + sin(pi / 3 + 0.02 * t) + cos(0.02 * t) + cos(0.03 * t + pi / 2)))) - 5587.531754730546)
p2 = plot!(p2, [0:0.01:tuningHorizon], setpoint.(0:0.01:tuningHorizon) * 1e-5, lw=2, lc=:black, ls=:dash, label="Setpoint")
p2 = plot!(p2, ylabel="p_s [bar]", xlabel="Time [s]")
is = 0

labelling = ["Set 1", "Set 2", "Set 3", "Manual"]
for i3 in eachrow(allVal)
    is = is + 1
    setpoint = t -> max(88000, min(97500, max(91000, 95000 + 5000 / 4 * (sin(0.009 * t) + sin(pi / 3 + 0.02 * t) + cos(0.02 * t) + cos(0.03 * t + pi / 2)))) - 5587.531754730546)
    rng = MersenneTwister(1234)
    xs = 0:NUMB:tuningHorizon

    xss = vec(sort!(tuningHorizon * rand!(rng, zeros(length(xs)))))
    A = vec((100000 - 92000) * rand!(rng, zeros(length(xs))) .+ 92000)

    itsp = scale(interpolate(A, BSpline(Constant())), xs)

    nodes = (xss,)
    itsps = extrapolate(interpolate(nodes, A, Gridded(Constant())), Flat())
    setpoint = t -> itsps(t)

    initP = (ubnd .- lbnd) .* i3 .+ lbnd

    α, ΔT = initP

    yAll, uAll1, yAll2, tAll2 = OFOoptim(α, ΔT, tuningHorizon, 323.6, x0, [x0[3], x0[2]], setpoint, I)

    # # # #####Picking values
    y1 = i -> yAll[i][1]
    y2 = i -> yAll[i][2]
    y3 = i -> yAll[i][3]
    y4 = i -> yAll[i][4]
    u1intp = x -> uAll1[Int(floor(x / ΔT) + 1)]

    y1intp = x -> y1(Int(floor(x / ΔT) + 1))

    # p = plot!(p1,[0:0.01:tuningHorizon],u1intp.(0:0.01:tuningHorizon),linetype=:steppre,lw=2,label=labelling[is]) #####Control

    function valueReal(yy, k, i, process)
        return yy[k][i][process]
    end
    y1All = (k, i) -> valueReal(yAll2, k, i, 1)
    ik = 0
    for k in eachindex(tAll2)
        if ik <= 0
            plot!(p1, tAll2[k], 1e-5 * y1All.(k, 1:length(tAll2[k])), lw=2, label=labelling[is], lc=palette(:default)[is+1])
        else
            plot!(p1, tAll2[k], 1e-5 * y1All.(k, 1:length(tAll2[k])), lw=2, label=:none, lc=palette(:default)[is+1])
        end
        ik += 1
    end

    initPNomad = 1.0 ./ (ubnd .- lbnd) .* initP .- lbnd ./ (ubnd .- lbnd)
    resError, resOver, resOsc = perfIndicators(initPNomad, x0, tuningHorizon)
    @show resError, resOver, resOsc

    setpoint = t -> max(88000, min(97500, max(91000, 95000 + 5000 / 4 * (sin(0.009 * t) + sin(pi / 3 + 0.02 * t) + cos(0.02 * t) + cos(0.03 * t + pi / 2)))) - 5587.531754730546)
    yAll, uAll1, yAll2, tAll2 = OFOoptim(α, ΔT, tuningHorizon, 323.6, x0, [x0[3], x0[2]], setpoint, I)

    y1intp = x -> y1(Int(floor(x / ΔT) + 1))

    # p = plot!(p2,[0:0.01:tuningHorizon],u1intp.(0:0.01:tuningHorizon),linetype=:steppre,lw=2,label=labelling[is]) #####Control

    function valueReal(yy, k, i, process)
        return yy[k][i][process]
    end
    y1All = (k, i) -> valueReal(yAll2, k, i, 1)
    ik = 0
    for k in eachindex(tAll2)
        if ik <= 0
            plot!(p2, tAll2[k], 1e-5 * y1All.(k, 1:length(tAll2[k])), lw=2, label=labelling[is], lc=palette(:default)[is+1])
        else
            plot!(p2, tAll2[k], 1e-5 * y1All.(k, 1:length(tAll2[k])), lw=2, label=:none, lc=palette(:default)[is+1])
        end
        ik += 1
    end

    # plot!(p2,ylabel="p_s [bar]", xlabel="Time [s]")

    initPNomad = 1.0 ./ (ubnd .- lbnd) .* initP .- lbnd ./ (ubnd .- lbnd)
    resError, resOver, resOsc = perfIndicators(initPNomad, x0, tuningHorizon)
    @show resError, resOver, resOsc

end
# plot!(p1,[0,1000],[300,300],lc=:black,ls=:dashdot,lw=2,label=:none)
plot!(p1, xtickfontsize=10, ytickfontsize=10, yguidefontsize=12, xguidefontsize=12, legendfontsize=12)
plot!(p1, legendcolumn=3)
plot!(p1, fontfamily="helvetica")
plot!(p1, leg=:bottom)
# ylims!(p1,0,1100)
# ylabel!(p1,"Torque [Nm]")
# xlabel!(p1,"Time [s]")
ylims!(p1, 0.85, 1.025)

# plot!(p2,[0,1000],[300,300],lc=:black,ls=:dashdot,lw=2,label=:none)
plot!(p2, xtickfontsize=10, ytickfontsize=10, yguidefontsize=12, xguidefontsize=12, legendfontsize=12)
plot!(p2, legendcolumn=3)
plot!(p2, fontfamily="helvetica")
plot!(p2, leg=:top)
# ylims!(p2,0,1100)
# ylabel!(p2,"Torque [Nm]")
# xlabel!(p2,"Time [s]")
ylims!(p2, 0.875, 1.025)