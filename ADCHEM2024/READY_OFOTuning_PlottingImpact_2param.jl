using OrdinaryDiffEq, Plots, Zygote, LinearAlgebra, OSQP
using SparseArrays
using Interpolations
using Integrals
using NOMAD
using Memoization

function realSystem(initValue, tspan, contr)
    x_0 = initValue
    testF = t -> contr
    p = testF

    #Define the problem
    function compressor!(du, u, p, t)

        # KProp,KInt, Kconst = p
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
        tauComp = 8      # time constant of the pressure ratio 8
        pr_coeff = [2.690940401290303 -0.013878128060951 -0.040925719808930 0.000986961896765 -0.000418575028867 0.000024527875520]

        ## Saturation conditions
        satValveVector = [0.071543358190296 0.071387873190811 5.278547153336578 -0.001031252436620]
        satTorqueVector = [73.3235171311808 0.0724497813744527 0.00517720515117661 6.41760174259480e-07]

        #Controllers
        SpeedControlNoSat(u) = ctrl1(t)

        # #Saturation
        SpeedControl(u) = satTorqueVector[1] / (satTorqueVector[2] + exp(-satTorqueVector[3] * SpeedControlNoSat(u))) + satTorqueVector[4]
        #Algebraic variables
        m_in(u) = Valve_in_gain * 0.4541 * (1.0 .+ 0.1 * 0 * (2 * rand(1) .- 1)) * sqrt(abs(In_pres - u[1]))  #######Disturbance in inlet valve opening
        m_out(u) = Valve_out_gain * 0.8 * (1.0 .+ 0.05 * 0 * (2 * rand(1) .- 1)) * sqrt(abs(u[2] - Out_pres)) #######Disturbance in outlet valve opening
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
    m, pd = d
    pr_coeff = [2.690940401290303 -0.013878128060951 -0.040925719808930 0.000986961896765 -0.000418575028867 0.000024527875520]
    Dt = 0.18   # impeller diameter
    sigma_slip = 0.9    # compressor slip

    omega_rad = 1 / (sigma_slip * (Dt / 2)^2 * m) * u ######Better approximation, even though the approximation ω=2τ also works
    omega_rpm = omega_rad / (2 * pi / 60)
    omega_percent = omega_rpm / 8370 * 100
    p_ratio = pr_coeff[1] + pr_coeff[2] * m + pr_coeff[3] * omega_percent + pr_coeff[4] * m * omega_percent + pr_coeff[5] * m^2 + pr_coeff[6] * omega_percent^2
    y = (1.05e5 - ((1 / (sqrt(1.2891e5 - 1.2e5))) * 4e1 * 0.8 / ((1 / (sqrt(1.05e5 - 1e5))) * 4e1 * 4 * 0.4541))^2 * 1.55e5) / (1 - ((1 / (sqrt(1.2891e5 - 1.2e5))) * 4e1 * 0.8 / ((1 / (sqrt(1.05e5 - 1e5))) * 4e1 * 4 * 0.4541))^2 * p_ratio)
    return y
end

function Φ(y, u, setp)
    z = (1e-2 * (y[1] - setp))^2# + 0.00001*u.^2
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

@memoize function perfIndicators(x, x0)
    xx = (ubnd .- lbnd) .* x .+ lbnd
    y, yy = OFOoptim(xx[1], xx[2], tuningHorizon, 323.6, x0, [75.0, 194085.875534926], setpoint, I)[1:2:3]
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

@memoize function perfIndicators(x, x0, fix)
    xx = (ubnd .- lbnd) .* x .+ lbnd
    y1int(s, p) = (x0[1] .- setpoint(s))^2
    probInt = IntegralProblem(y1int, 0, tuningHorizon, xx[2])
    solInt = Integrals.solve(probInt, QuadGKJL())

    yy1 = i -> yy[i][1]
    yy1intTest(s, ps) = (mapreduce(permutedims, vcat, yy1.(Int.(floor.(s ./ ps) .+ 1)))[:, 1] .- setpoint.(s))

    sT = LinRange(0, tuningHorizon - xx[2], 5000)

    yy1int(s, ps) = x0[1] .- setpoint.(s)

    xtest = yy1int(sT, xx[2]) .<= 0.0
    intError = solInt.u
    numOsc = count(x -> (x < 0 || x > 0), diff(xtest))
    return intError, 0.0, numOsc

end

tuningHorizon = 200
lbnd = [0, 5e-3,]
ubnd = [1e3, tuningHorizon / 2 - 1]

pAll = []
######################################################################
###############FOR TESTS
# setpoint = t -> max(94000,95000+5000*sin(0.04*t)) 
# setpoint = t -> (t>=75 && t<= 125) ? 98000 : (t>=150 ? 93000 : 95000)
setpoint = t -> 92500

initP = [1e-1, 5e1]


#########Function of time
allVal = [
    1e0 5e-3
    1e0 5e-2#####Best time
    1e0 5e-1
    1e0 5e0
    1e0 5e1
]

#########Function of step size
allVal = [1e3 5e-2
    # 1e2 5e-2 
    1e1 5e-2
    1e0 5e-2
    1e-1 5e-2
    # 1e-2 5e-2  
    1e-3 5e-2
]

########################################
tuningHorizon = 200
x0 = [101538.78020791685, 186796.53698194958, 60.450634195325335, 647.2042336910482]

#  p = plot([0:0.01:tuningHorizon],setpoint.(0:0.01:tuningHorizon).*1e-5,lw=2,lc=:black,ls=:dash,label="Setpoint") #####For responses

p = plot() #####For control
plot!([0, 200], [1000, 1000], lc=:black, ls=:dash, lw=2, label=:none) #####For control

resErrorAll = []
resOscAll = []
for i3 in eachrow(allVal)
    initP = i3
    α, ΔT = initP
    tuningHorizon = 200
    uAll1, yAll2, tAll2 = OFOoptim(α, ΔT, tuningHorizon, 323.6, x0, [x0[3], x0[2]], setpoint, I)[2:4]

    u1intp = x -> uAll1[Int(floor(x / ΔT) + 1)]

    # pu = plot!([0:0.01:tuningHorizon],u1intp.(0:0.01:tuningHorizon),linetype=:steppre,lw=2,label=raw"ΔT="*string(ΔT)) ######CONTROL for ΔT
    pu = plot!([0:0.01:tuningHorizon], u1intp.(0:0.01:tuningHorizon), linetype=:steppre, lw=2, label=raw"ν=" * string(α)) ######CONTROL for ν

    initPNomad = 1.0 ./ (ubnd .- lbnd) .* initP .- lbnd ./ (ubnd .- lbnd)
    resError, resOver, resOsc = perfIndicators(initPNomad, x0)
    push!(resErrorAll, resError)
    push!(resOscAll, resOsc)
    @show resError, resOver, resOsc

    ##########Plotting real responses, not at discrete timesteps
    # function valueReal(yy,k,i,process)
    #     return yy[k][i][process]
    # end
    # y1All=(k,i) -> valueReal(yAll2,k,i,1)

    # for k in eachindex(tAll2)
    #     plot!(p,tAll2[k],1e-5*y1All.(k,1:length(tAll2[k])),lc=:black,lw=2,label=raw"ν="*string(α)) ######Responses for ν
    #   # plot!(p,tAll2[k],1e-5*y1All.(k,1:length(tAll2[k])),lc=:black,lw=2,label=raw"ΔT="*string(ΔT)) ######Responses for ΔT
    # end

end
# p=plot!(ylabel = "p_s [bar]",xlabel = "Time [s]")
# ylims!(0.87, 1.03)

initP = [1e-1, 5e1] #####Initial guess chosen for optimization

α = initP[1]
ΔT = initP[2]

uAll1, yAll2, tAll2 = OFOoptim(α, ΔT, tuningHorizon, 323.6, x0, [x0[3], x0[2]], setpoint, I)[2:4]

u1intp = x -> uAll1[Int(floor(x / ΔT) + 1)]
function valueReal(yy, k, i, process)
    return yy[k][i][process]
end
y1All = (k, i) -> valueReal(yAll2, k, i, 1)
# for k in eachindex(tAll2)
#     plot!(p,tAll2[k],1e-5*y1All.(k,1:length(tAll2[k])),lc=:black,lw=2,label="Initial guess") ######Responses for ν
# end

p = plot!([0:0.01:tuningHorizon], u1intp.(0:0.01:tuningHorizon), linetype=:steppre, ls=:dot, lw=2, lc=:red, label="Initial guess") #####Control for initial guess
plot!([0, 200], [300, 300], lc=:black, ls=:dashdot, lw=2, label=:none) #####For control

ylims!(0.84, 1.03)
plot!(xtickfontsize=10, ytickfontsize=10, yguidefontsize=12, xguidefontsize=12, legendfontsize=12)
plot!(legendcolumn=3)
plot!(fontfamily="helvetica")
plot!(leg=:bottom)
ylims!(0, 1100)
ylabel!("Torque [Nm]")
xlabel!("Time [s]")