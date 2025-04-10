##################This file is a part of ADCHEM 2024 submission
##################This script calculates the time constant for steady-state OFO for 

using OrdinaryDiffEq, Plots, ForwardDiff
using ControlSystems

function realSystem(initValue, tspan, contr)
    x_0 = initValue
    testF = contr
    p = testF

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
        SpeedControlNoSat = ctrl1

        # #Saturation
        SpeedControl = satTorqueVector[1] / (satTorqueVector[2] + exp(-satTorqueVector[3] * SpeedControlNoSat)) + satTorqueVector[4]
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
        du[4] = 1 / J * (SpeedControl - torque_comp(u))
    end

    function compressorCtr!(x)
        satTorqueVector = [73.3235171311808 0.0724497813744527 0.00517720515117661 6.41760174259480e-07]
        J = 10   # shaft inertia
        # #Saturation
        SpeedControl = satTorqueVector[1] / (satTorqueVector[2] + exp(-satTorqueVector[3] * x)) + satTorqueVector[4]
        du = 1 / J * (SpeedControl)
    end

    function diff_function(u, p)
        du = zero(u)
        compressor!(du, u, p, t)
        du
    end

    prob = ODEProblem(compressor!, x_0, tspan, p)

    Jx = ForwardDiff.jacobian(x -> diff_function(x, contr), x0)
    Ju = ForwardDiff.derivative(x -> compressorCtr!(x), contr)

    return prob, Jx, Ju
end

x0 = [101538.78020791685, 186796.53698194958, 60.450634195325335, 647.2042336910482]

controller = 323.6

tuningHorizon = 200
t = 0
mySystem, myJacX, myJacU = realSystem(x0, (0, 200), controller)
B = [0, 0, 0, myJacU]
C = [1 0 0 0]
D = 0
sys = ss(myJacX, B, C, D)

uu(x, t) = [1] ####Step input
t = 0:0.1:200  ####Time for simulation
res = lsim(sys, uu, t, x0=[0, 0, 0, 0]) ####Simulation with desired uu step size over time t
si = stepinfo(res; settling_th=0.05)  ##Computing the time constants
plot(si) ##Plotting

