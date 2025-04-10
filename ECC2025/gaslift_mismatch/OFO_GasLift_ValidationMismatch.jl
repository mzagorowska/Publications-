using JuMP, Ipopt, LinearAlgebra, Plots, MadNLP, InfiniteOpt, ForwardDiff, DiffOpt, OSQP, Clarabel, BlockDiagonals, Zygote
using ChunkSplitters, BlockArrays

############
function realSystem(u)
    normalisedU = (u - muAll'[:, 1]) ./ muAll'[:, 2]
    uMatrix = ones(size(normalisedU))
    for ni = 1:4
        uMatrix = hcat(normalisedU .^ ni, uMatrix)
    end
    allWells = diag(pAll * uMatrix')
    sol = [
        sum(allWells[1:2]);
        sum(allWells[3:end])
    ]
    return sol #####We return a measurement
end

function hMap(u) 
    normalisedU = (u-muAll'[:,1])./muAll'[:,2]
    uMatrix = ones(size(normalisedU))
    for ni=1:4
        uMatrix = hcat(normalisedU.^ni,uMatrix)
    end
    allWells = diag(pAll*uMatrix')
    y= [ 
        sum(allWells[1:2]);
        sum(allWells[3:end])
    ]
    return y
end

function Φ(u,y)
    c2 = -[1;1]  ######Linear fcn of outputs
    z = c2'*y
    return z
end
################


function internalProjectionFO(α, ucurrent, ycurrent,mapp,objFcn,dudβcurrent,bval,timestamp)
    ycurrent = ycurrent
    noInputs = size(ucurrent)[1]
    noOutputs = size(ycurrent)[1]

    β=zeros(noOutputs,noInputs)
    β[1,1:2] = bval[1:2]
    β[2,3:end] =  bval[3:end]
    if typeof(mapp) == String
        ∇h = u-> [ 
            0.00228076  0.0439291  0.0         0.0         0.0
            0.0         0.0        0.00603082  0.00263142  0.0106929
            ]   ####Constant mismatch as linear approximation at uC
    else
            ∇h = u -> Zygote.jacobian(x -> mapp(x), u)[1]
    end
    ΦmatT = (a, b) -> [Zygote.jacobian((u,y) -> objFcn(u,y), a, b)[1] Zygote.jacobian((u,y) -> objFcn(u,y), a, b)[2]]'
    solver = optimizer_with_attributes(Clarabel.Optimizer);

    m = Model(() -> DiffOpt.diff_optimizer(solver))
    set_silent(m)
    @variable(m, w[1:noInputs])
    @expression(m,gradMapping, ∇h(ucurrent)+ β)
    @expression(m,gradMappingH, [I(noInputs) transpose(gradMapping)])

    @expression(m,objExp,(w+Gmat^(-1)* gradMappingH * ΦmatT(ucurrent, ycurrent))'*Gmat*(w+Gmat^(-1)* gradMappingH * ΦmatT(ucurrent, ycurrent)))
    @objective(m, Min, objExp[1])

    Adoublebar = vcat(Am* α, Cm* α*gradMapping)
    bdoublebar = vcat(Bm- Am*ucurrent, Dm - Cm * realSystem(ucurrent)) 
    @constraint(m,cstIn,Adoublebar*w.<= bdoublebar)

    optimize!(m)
    results = value.(m[:w])

    return results

end


function OFOoptim(α, ΔT, tuningHorizon, ucurrent, ycurrent,maps,objFcn,bval)
    kMax = tuningHorizon ./ ΔT - 1
    yAll = [ycurrent]
    uAll = [ucurrent]
    for k = 0:kMax
        currentIt = Int(k+1)

        projResult = internalProjectionFO(α, ucurrent, ycurrent, maps, objFcn, 0,bval,currentIt)
        uucstr = ucurrent + α * projResult
        unew = uucstr
        ySol = realSystem(unew)
        ycurrent = ySol
        ucurrent = unew

        yAll = push!(yAll, ycurrent)
        uAll = push!(uAll, ucurrent)
    end
    return yAll, uAll

end

##########This example comes from Andersen et al.
######Polynomial coefficients in order p[i,1]x_i^4+...+p[i,end]
pAll = [
    0.191797938199848 -0.113868722033584 -2.760639730334368 3.111091052053845 51.705723863963335;
    1.155485286000176 1.073136842565429 -23.553246910200947 25.072845333766857 81.180958934479548;
    0.219717600847841 1.056535523587069 -8.912549987255872 15.255547812632539 29.078594520354670;
    0.021993247841200 1.084969367582595 -3.661574294801760 3.803685277969326 23.920698366641496;
    -0.476453301251982 1.645356449258222 -3.545378432203434 4.461827859437030 34.797756781171159
]
######Normalisation factors in order m[1,i]=μ_i, m[2,i]=σ_i, 
muAll = 1e3 .* [
    5.541054239285714 8.924018118833333 3.901365037555556 4.685453349499999 6.316213543111111
    3.229821089421953 1.716584841625922 1.098052591864462 1.753597775232707 1.772797059963122
]

Am = [ 1 0 0 0 0; -1 0 0 0 0; 0 1 0 0 0; 0 -1 0 0 0; 0 0 1 0 0;0 0 -1 0 0;0 0 0 1 0;0 0 0 -1 0;0 0 0 0 1;0 0 0 0 -1]
Bm = [9576;-1157;11745;-6819;5972;-2714;7377;-2399;9043;-4125]
Cm = [ 1 0; -1 0; 0 1; 0 -1]
Dm = [150.0;0.0;150;0.0]

β=[0.0; 0.0; 0.0; 0.0;0.0]

noInputs = size(Am)[2]
noOutputs=size(Cm)[2]
noMismatch = size(β)[1]

Gmat = Matrix(1.0I, noInputs,noInputs)

α = 500.0

uC = [2500.0,7000.0,4500.0,4500.0,4500.0]

yC = realSystem(uC)
display(yC)

ΔT = 1
tH = 500 
print("Ideal case")

#####No mismatch case
rsC = u -> realSystem(u)
resOFOyId, resOFOuId = OFOoptim(α, ΔT, tH, uC, yC, rsC, Φ,β)

uSolId = (i, j) -> resOFOuId[j][i]
ySolId = (i, j) -> resOFOyId[j][i]

#####Mismatch due to constant gradient
rsC = "constant gradient"
resOFOyId, resOFOuId = OFOoptim(α, ΔT, tH, uC, yC, rsC, Φ,β)

######Uncomment to get a plot of inputs
# plot(uSolId.(1, 1:ΔT:tH), lw=1, label="u1")
# plot!(uSolId.(2, 1:ΔT:tH), lw=1, label="u2")
# plot!(uSolId.(3, 1:ΔT:tH), lw=1, label="u3")
# plot!(uSolId.(4, 1:ΔT:tH), lw=1, label="u4")
# plot!(uSolId.(5, 1:ΔT:tH), lw=1, label="u5")

######Uncomment to get a plot of outputs
# plot(ySolId.(1,1:ΔT:tH), lw=1, label="y1")
# plot!(ySolId.(2,1:ΔT:tH), lw=1, label="y2")

######Uncomment to get a plot of the objective
allPhi = []
for si=1:ΔT:tH
    push!(allPhi,Φ(uSolId.(:,si),ySolId.(:,si)))
end
plot!(-allPhi, label="Objective value")
