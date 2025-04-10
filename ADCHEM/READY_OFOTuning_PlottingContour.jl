using OrdinaryDiffEq, Plots, Zygote, LinearAlgebra, OSQP
using SparseArrays
using Interpolations
using Integrals
using Memoization
using Roots

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
        m_in(u) = Valve_in_gain * 0.4541  * sqrt(abs(In_pres - u[1]))  #######Disturbance in inlet valve opening
        m_out(u) = Valve_out_gain * 0.8  * sqrt(abs(u[2] - Out_pres)) #######Disturbance in outlet valve opening
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

    omega_rad = 1 / (sigma_slip * (Dt / 2)^2 * m) * u ######Better approximation, even though the approximation ω=2τ also works
    omega_rpm = omega_rad / (2 * pi / 60)
    omega_percent = omega_rpm / 8370 * 100
    p_ratio = pr_coeff[1] + pr_coeff[2] * m + pr_coeff[3] * omega_percent + pr_coeff[4] * m * omega_percent + pr_coeff[5] * m^2 + pr_coeff[6] * omega_percent^2

    y = (1.05e5 -((1 / (sqrt(1.2891e5 - 1.2e5))) * 4e1*0.8/((1 / (sqrt(1.05e5 - 1e5))) * 4e1 * 4*0.4541))^2*1.55e5)/(1-((1 / (sqrt(1.2891e5 - 1.2e5))) * 4e1*0.8/((1 / (sqrt(1.05e5 - 1e5))) * 4e1 * 4*0.4541))^2*p_ratio)
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

@memoize function perfIndicatorsContour(x, x0)
    xx = x
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
    numOsc = count(x -> (x < 0 || x > 0), diff(xtest))
    return intError, numOsc

end

tuningHorizon = 200
lbnd = [0, 5e-3,]
ubnd = [1e3, tuningHorizon / 2 - 1]

pAll = []
######################################################################
setpoint = t -> 92500
x0 = [101538.78020791685, 186796.53698194958, 60.450634195325335, 647.2042336910482]

##########################################
noPoints = 20   #############################NUMBER OF POINTS FOR CONTOUR PLOTS. The more points, the longer it takes so handle with care
x = range(0, 1000, length=10*noPoints)
y = range(5e-3, 50, length=noPoints)

zfun = (x,y) -> 1e-8*perfIndicatorsContour([x y], x0)[1]
zfun2 = (x,y) -> perfIndicatorsContour([x y], x0)[2]

val=zfun.(x',y)
val2=zfun2.(x',y)

########Removing unstable points
val2=Float64.(val2)
for i = 1:length(y)
    for j=1:length(x)
        if val[i,j]>=200
            val[i,j]=NaN
            val2[i,j]=NaN
        end
    end
end

p1= plot()
p1 = contourf(p1,x,y,val,lw=0,color=:vik,levels=100,label="Error")
xlabel!(p1, "ν")
ylabel!(p1,"ΔT")
p2=plot()
p2 = contourf(p2, x,y,val2,lw=0,color=:vik,levels=200,label="Oscillations")
xlabel!(p2, "ν")
ylabel!(p2, "ΔT")

itpval = interpolate((y, x), val, Gridded(Linear()))
itpval2 = interpolate((y, x), val2, Gridded(Linear()))
valAll = []
for ii in range(1e-3, 50, 100)
    ff(xx) = itpval(ii, xx) - itpval2(ii, xx)
    try
        intersectVall = find_zero(ff, (0, 500))
        push!(valAll, [ii intersectVall ff(intersectVall)])
    catch
        println("Intersection failed")
    end

end
valInt = Float64.(permutedims(vcat(valAll...))')
p1 = plot!(p1,valInt[:,2],valInt[:,1],lc=:white,lw=4,markershape=:circle, markercolor=:white,label=:none)
p2 = plot!(p2,valInt[:,2],valInt[:,1],lc=:white,lw=4,markershape=:circle, markercolor=:white,label=:none)
xlabel!(p1,"ν")
ylabel!(p1, "ΔT")
xlabel!(p2,"ν")
ylabel!(p2, "ΔT")
plot!(p1,xtickfontsize=10, ytickfontsize=10, yguidefontsize=12, xguidefontsize=12, legendfontsize=12)
plot!(p1,fontfamily="helvetica")
annotate!(p1, valInt[:,2].+5.0,valInt[:,1].-0.1, text.(string.(round.(itpval.(valInt[:,1],valInt[:,2]);sigdigits=2)), :white, :left, 8))
xlims!(p1,0,250)
ylims!(p1,0,12)
plot!(p2,xtickfontsize=10, ytickfontsize=10, yguidefontsize=12, xguidefontsize=12, legendfontsize=12)
plot!(p2,fontfamily="helvetica")
xlims!(p2,0,250)
ylims!(p2,0,12)
annotate!(p2, valInt[:,2].+5.0,valInt[:,1].-0.1, text.(string.(round.(itpval2.(valInt[:,1],valInt[:,2]);sigdigits=2)), :white, :left, 8))