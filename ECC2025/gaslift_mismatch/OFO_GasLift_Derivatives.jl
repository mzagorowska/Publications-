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
    timecurrent = timestamp

    ycurrent = ycurrent
    noInputs = size(ucurrent)[1]
    noOutputs = size(ycurrent)[1]
    noMismatch = prod(size(bval))

    β=zeros(noOutputs,noInputs)
    β[1,1:2] = bval[1:2]
    β[2,3:end] =  bval[3:end]
    ∇h = u -> Zygote.jacobian(x -> mapp(x), u)[1]

    ΦmatT = (a, b) -> [Zygote.jacobian((u,y) -> objFcn(u,y), a, b)[1] Zygote.jacobian((u,y) -> objFcn(u,y), a, b)[2]]'
    solver = optimizer_with_attributes(Clarabel.Optimizer);

    m = Model(() -> DiffOpt.diff_optimizer(solver))
    set_silent(m)
    @variable(m, w[1:noInputs])
    @expression(m,gradMapping, ∇h(ucurrent)+ β)  ###reshape(β,noOutputs,noInputs), vcat(β[1:noInputs]',β[noInputs+1:end]'). The current version is ONLY for TWO OUTPUTS
    @expression(m,gradMappingH, [I(noInputs) transpose(gradMapping)])

    @expression(m,objExp,(w+Gmat^(-1)* gradMappingH * ΦmatT(ucurrent, ycurrent))'*Gmat*(w+Gmat^(-1)* gradMappingH * ΦmatT(ucurrent, ycurrent)))
    @objective(m, Min, objExp[1])
    cdoublebar = 2.0 .* gradMappingH * ΦmatT(ucurrent, ycurrent)

    Adoublebar = vcat(Am* α, Cm* α*gradMapping)
    bdoublebar = vcat(Bm- Am*ucurrent, Dm - Cm * realSystem(ucurrent)) 
    @constraint(m,cstIn,Adoublebar*w.<= bdoublebar)

    optimize!(m)
    results = value.(m[:w])
    ##############For manual computations

    gradhM = ∇h(ucurrent)+β###########This line defines how the mismatched gradient is parametrised as a function of β

    gradPhi = ΦmatT(ucurrent,ycurrent)
    Hbig = [I(noInputs) gradhM']
    cdoublebar = 2.0 .* Hbig * gradPhi[1:noInputs+noOutputs]
    Adoublebar = vcat(Am * α, Cm * α * gradhM)
    bdoublebar = vcat(Bm - Am * ucurrent, Dm - Cm * ycurrent) 
    Gdoublebar = 2 * Gmat

    LHM = [
        Gdoublebar -transpose(Adoublebar)
        -Diagonal(dual.(cstIn)[1:size(Adoublebar)[1]])*Adoublebar -Diagonal(Adoublebar*value.(w)-bdoublebar) 
    ]
    LHMinv = inv(LHM)
    k1 = length(LHM[:,1])
    k2 = length(Diagonal(dual.(cstIn)[1:size(Adoublebar)[1]])[1,:])
    RHMb = [zeros(k1-k2,k2)
    -Diagonal(dual.(cstIn)[1:size(Adoublebar)[1]])] 
    RHMc = [-Matrix(1.0I, noInputs,noInputs)
    zeros(k2,noInputs)]

    dallb = LHMinv*RHMb    ####both w and λ
    dallc = LHMinv*RHMc    ####both w and λ
    dwb = dallb[1:noInputs,1:end] ####picking only w
    dwc = dallc[1:noInputs,1:end] ####picking only w
    dAm = zeros(size(Adoublebar))
    dwA = []
    for i = 1:size(Adoublebar)[1]
        for j = 1:size(Adoublebar)[2]
            dAm = zeros(size(Adoublebar))
            dAm[i, j] = 1.0 ######Do for every element of Adoublebar
            RHMA = [
                transpose(dAm)* dual.(cstIn)[1:size(Adoublebar)[1]]
                Diagonal(dual.(cstIn)[1:size(Adoublebar)[1]])* dAm* value.(w)
            ]
            dallAtemp = LHMinv * RHMA
            dwAtemp=dallAtemp[1:noInputs,1:end] ####picking only w
            push!(dwA, dwAtemp)
        end
    end
    dwA = reduce(hcat,dwA)
    dwG = []
    for i = 1:size(Gdoublebar)[1]
        for j = 1:size(Gdoublebar)[2]
            dGm = zeros(size(Gdoublebar))
            dAm = zeros(size(Adoublebar))

            dGm[i, j] = 1.0 ######Do for every element of Gdoublebar

            RHMG = -[
                dGm*value.(w)
                0.0*Diagonal(dual.(cstIn)[1:size(Adoublebar)[1]])* dAm* value.(w)
            ]
            dallGtemp = LHMinv * RHMG
            dwGtemp=dallGtemp[1:noInputs,1:end] ####picking only w
            push!(dwG, dwGtemp)
        end
    end
    dwG = reduce(hcat,dwG)
    ####Zygote computations
    gradhTf = (u,β) -> Zygote.jacobian(x -> mapp(x), u)[1]
    
    gradhMf = (u,β) -> gradhTf(u,β)+ [β[1] β[2] 0 0 0; 0 0 β[3] β[4] β[5]] ########### HARD CODED BAD. This line defines how the mismatched gradient is parametrised as a function of β
        
    gradPhif = (u,β,y) -> ΦmatT(u,y)
    Hbigf = (u,β) -> [I(noInputs) gradhMf(u,β)']
    cdoublebarf = (u,β,y) -> 2.0 .* Hbigf(u,β) * gradPhif(u,β,y)[1:noInputs+noOutputs]
    Adoublebarf = (u,β) -> vcat(Am * α, Cm * α * gradhMf(u,β))
    bdoublebarf = (u,β,y) -> vcat(Bm - Am * u, Dm .- Cm * y)
    Gdoublebarf = (u,β) -> 2 * Gmat

    dcβ = ForwardDiff.jacobian( b -> cdoublebarf(ucurrent,b,ycurrent), β)
    dcu = ForwardDiff.jacobian( a -> cdoublebarf(a,β,ycurrent), ucurrent)
    dcy = ForwardDiff.jacobian( c -> cdoublebarf(ucurrent,β,c), ycurrent)
    dyu = ForwardDiff.jacobian( x -> realSystem(x), ucurrent)

    dbβ = ForwardDiff.jacobian( b -> bdoublebarf(ucurrent,b,ycurrent), β)
    dbu = ForwardDiff.jacobian( a -> bdoublebarf(a,β,ycurrent), ucurrent)
    dby = ForwardDiff.jacobian( c -> bdoublebarf(ucurrent,β,c), ycurrent)
    dyu = ForwardDiff.jacobian( x -> realSystem(x), ucurrent)

    dAβ = ForwardDiff.jacobian( b -> Adoublebarf(ucurrent,b), β) 
    dAu = ForwardDiff.jacobian( a -> Adoublebarf(a,β), ucurrent)

    #################Rearranging dAβ to get the same order of derivatives as in dwA
    tempInd = []
    for s =1:noInputs
        push!(tempInd,10*collect(range(1,size(Adoublebar)[1], step=1)).+s)
    end
    tempMat = hcat(reduce(vcat,tempInd),dAβ)
    dAβrearranged = sortslices(tempMat,dims=1)[:,2:end]
    ###################################

    dQPdβ = (dwb*(dbβ)+dwc*(dcβ)+dwA*dAβrearranged)[1:noInputs,1:noMismatch] 
    dQPdu = (dwb*(dbu+dby*dyu)+dwc*(dcu+dcy*dyu)+dwA*(dAu))

    dQPdu = reshape(dQPdu,:,noInputs)
    BigBlock = dQPdβ
    if timecurrent >1
        BigBlock = BlockDiagonal([dQPdβ,kron(Matrix(1I, Int(timecurrent-1), Int(timecurrent-1)),dQPdu)])
    end
    dwdβ = BigBlock*dudβcurrent

    return results,dwdβ

end


function OFOoptim(α, ΔT, tuningHorizon, ucurrent, ycurrent,maps,objFcn,bval)
    kMax = tuningHorizon ./ ΔT - 1
    yAll = [ycurrent]
    uAll = [ucurrent]
    wAll = [0.0.*ucurrent]
    derivucurrent = I(noMismatch)
    derivwcurrent = 0
    derivuAll = derivucurrent
    derivwAll = []
    derivΦAll = []
    derivΦuAll = []

    derivwmatrix = zeros(Int(noInputs*(kMax+1)),Int(noMismatch*(kMax+1)))
    derivumatrix = zeros(Int(noInputs*(kMax+1)),Int(noMismatch*(kMax+1)))
    derivPhimatrix = zeros(Int(kMax)+1,noMismatch*Int(kMax)+5)
    for k = 0:kMax
        currentIt = Int(k+1)

        projResult, derivwcurrent = internalProjectionFO(α, ucurrent, ycurrent, maps, objFcn, derivuAll,bval,currentIt)
        push!(derivwAll, derivwcurrent)
        reversederivw = []

        for c=1:currentIt
            cTemp = BlockArray(derivwAll[currentIt], vec(Int.(noInputs*ones(1,currentIt))),[noMismatch,0])
            cr = cTemp[Block(c,1)]
            push!(reversederivw,cr)
        end
        derivwmatrix[noInputs*currentIt+1-noInputs:noInputs*currentIt+1-noInputs+noInputs-1,1:currentIt*noMismatch] = reduce(hcat,reshape(reverse(reversederivw),1,:))
        
        for si=1:noInputs
            derivumatrix[noInputs*currentIt+1-noInputs+si-1,:] = α*reshape(sum(derivwmatrix[si:noInputs:end,:];dims=1),1,:)
        end
        derivuAll = derivumatrix[noInputs*currentIt+1-noInputs:noInputs*currentIt+1-noInputs+noInputs-1,1:currentIt*noMismatch]
        reversederivu = []
        derivuAllTemp = copy(transpose(derivuAll))
        for c=1:currentIt
            cTemp = BlockArray(derivuAllTemp, vec(Int.(noMismatch*ones(1,currentIt))),[noInputs,0])
            cr = transpose(cTemp[Block(c,1)])
            push!(reversederivu,cr)
        end
        reversederivu = reduce(vcat,reverse(reversederivu))
        derivuAll = vcat(I(noMismatch),reversederivu)

        uucstr = ucurrent + α * projResult
        unew = uucstr
        ySol = realSystem(unew)
        ycurrent = ySol
        ucurrent = unew

        dΦu = transpose(ForwardDiff.gradient(a -> Φ(a,ycurrent), ucurrent))
        dΦy = transpose(ForwardDiff.gradient( b -> Φ(ucurrent,b), ycurrent))

        dyu = ForwardDiff.jacobian( x -> realSystem(x), ucurrent)

        dPhidu = dΦu+dΦy*dyu
        dPhidu = reshape(dPhidu,1,:)
        BigBlockPhi = kron(Matrix(1I, currentIt, currentIt),dPhidu[1:noInputs])'
        dΦβAll = BigBlockPhi*derivuAll[noMismatch+1:end,:]
        dΦβ = sum(dΦβAll;dims=1)

        derivPhimatrix[1:currentIt,5*currentIt-4:5*currentIt-4+noMismatch-1] = dΦβAll
       
        push!(derivΦAll, dΦβ)
        push!(derivΦuAll, dΦu)
        
        yAll = push!(yAll, ycurrent)
        uAll = push!(uAll, ucurrent)
        wAll = push!(wAll, projResult)

    end
    return yAll, uAll, derivuAll, derivwAll, derivΦAll, derivPhimatrix,wAll

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


#####Mismatch - here none
β=[0.0; 0.0; 0.0; 0.0;0.0]

noInputs = size(Am)[2]
noOutputs=size(Cm)[2]
noMismatch = size(β)[1]

####The scaling matrix and the step size
Gmat = Matrix(1.0I, noInputs,noInputs)
α = 500.0

####The initial condition
uC = [2500.0,7000.0,4500.0,4500.0,4500.0]

yC = realSystem(uC)
display(yC)

ΔT = 1
tH = 500 
print("Ideal case")

rsC = u -> realSystem(u)
hmC = u -> hMap(u)

########Run OFO with no active constraints
Am = [ 1 0 0 0 0; -1 0 0 0 0; 0 1 0 0 0; 0 -1 0 0 0; 0 0 1 0 0;0 0 -1 0 0;0 0 0 1 0;0 0 0 -1 0;0 0 0 0 1;0 0 0 0 -1]
Bm = [9576;-1157;11745;-6819;5972;-2714;7377;-2399;9043;-4125]
Cm = [ 1 0; -1 0; 0 1; 0 -1]
Dm = [150.0;0.0;150;0.0]

resOFOyId, resOFOuId, resOFOderivuId, resOFOderivwId, resOFOderivΦId, resMatrix,allSolutions = OFOoptim(α, ΔT, tH, uC, yC, rsC, Φ,β)

uSolId = (i, j) -> resOFOuId[j][i]
ySolId = (i, j) -> resOFOyId[j][i]

# ########Run OFO with active constraints
# Am = [ 1 0 0 0 0; -1 0 0 0 0; 0 1 0 0 0; 0 -1 0 0 0; 0 0 1 0 0;0 0 -1 0 0;0 0 0 1 0;0 0 0 -1 0;0 0 0 0 1;0 0 0 0 -1; 1 1 1 1 1]
# Bm = [9576;-1157;11745;-6819;5972;-2714;7377;-2399;9043;-4125; 26000]
# Cm = [ 1 0; -1 0; 0 1; 0 -1]
# Dm = [130.0;0.0;150;0.0]
# resOFOyId, resOFOuId, resOFOderivuId, resOFOderivwId, resOFOderivΦId, resMatrix,allSolutions = OFOoptim(α, ΔT, tH, uC, yC, rsC, Φ,β)
# uSolId = (i, j) -> resOFOuId[j][i]
# ySolId = (i, j) -> resOFOyId[j][i]

######Uncomment to get a plot of inputs
# plot(uSolId.(1, 1:ΔT:tH), lw=1, label="u1")
# plot!(uSolId.(2, 1:ΔT:tH), lw=1, label="u2")
# plot!(uSolId.(3, 1:ΔT:tH), lw=1, label="u3")
# plot!(uSolId.(4, 1:ΔT:tH), lw=1, label="u4")
# plot!(uSolId.(5, 1:ΔT:tH), lw=1, label="u5")

######Uncomment to get a plot of outputs
# plot(ySolId.(1,1:ΔT:tH), lw=1, label="y1")
# plot!(ySolId.(2,1:ΔT:tH), lw=1, label="y2")

# ######Uncomment to get a plot of the objective
# allPhi = []
# for si=1:ΔT:tH
#     push!(allPhi,Φ(uSolId.(:,si),ySolId.(:,si)))
# end
# plot!(-allPhi, label="Objective value")

######Uncomment to get a plot of the derivatives of the objective. Change here [:,1:2] which wells to plot
# scatter(reduce(vcat, resOFOderivΦId)[:,1:2], lw=2, label="dΦ")

######Uncomment to get a plot of the solutions of the projection. Change here [:,1:2] which wells to plot
# scatter(reduce(hcat, allSolutions)'[:,1:2], lw=2, label="w")

# ######Uncomment to get a plot the heatmaps (NB: take A LOT of time)
# using GMT
# x = y = range(1, stop = tH, length = tH)

# for sd =1:noInputs ####Change here which well you want to plot, currently all five wells
#     G1 = mat2grid(resMatrix[:,sd:5:end], x, y);
#     if sd<=1 ####The first well requires more resolution
#         GMT.contourf(G1,show=true,contour=.0005,colorbar=true)
#     else
#         GMT.contourf(G1,show=true,contour=.02,colorbar=true)
#     end
# end