using JuMP, Ipopt, LinearAlgebra, Plots, MadNLP, InfiniteOpt, Zygote, DiffOpt, OSQP
using BlockDiagonals, Clarabel
############
function realSystem1D(contr)
    sol = contr.^2+4*contr
    return sol #####We return a measurement
end

function hMap1D(u)
    y= u.^2+4*u
    return y
end

function Φ1D(u,y)
    z = 0.1.*(u.^2.0.*y-4.0.*u.*y+5.0.*u).+5
    return z
end
################

function internalProjectionFO(αval, ucurrent, ycurrent,mapp,objFcn,dudβcurrent,dudαcurrent,bval)
    α = αval
    β = bval
    ycurrent = ycurrent
    noInputs = length(ucurrent)
    timecurrent = length(dudαcurrent)
    ∇h = u -> jacobian(x -> mapp(x), u)[1]
    Adoublebar = vcat(Am.* α, Cm.* α.* (∇h(ucurrent).+β))
    bdoublebar = vcat(Bm.- Am.*ucurrent, Dm .- Cm .* realSystem1D(ucurrent)) #####Dots are needed because I have only one output

    ΦmatT = (a, b) -> [jacobian((u,y) -> objFcn(u,y), a, b)[1] jacobian((u,y) -> objFcn(u,y), a, b)[2]]'
    solver = optimizer_with_attributes(Clarabel.Optimizer);

    m = Model(() -> DiffOpt.diff_optimizer(solver))
    set_silent(m)
    @variable(m, w[1:noInputs])
    @expression(m,gradMapping, ∇h(ucurrent).+β)
    @expression(m,gradMappingH, [ I transpose(gradMapping)])
    @expression(m,objExp,(w+Gmat^(-1)* gradMappingH * ΦmatT(ucurrent, ycurrent))'*Gmat*(w+Gmat^(-1)* gradMappingH * ΦmatT(ucurrent, ycurrent)))
    @objective(m, Min, objExp[1])

    @constraint(m,cstIn,Adoublebar.*w.<= bdoublebar)

    optimize!(m)
    results = value.(m[:w])
    # ##############For manual computations
    gradhT = [2*ucurrent[1]+4]

    Adoublebar = vcat(Am.* α, Cm.* α.* (gradhT.+β))
    bdoublebar = vcat(Bm.- Am.*ucurrent, Dm .- Cm .* realSystem1D(ucurrent)) #####Dots are needed because I have only one output

    gradhM = gradhT .+ β ###########This line defines how the mismatched gradient is parametrised as a function of β
    gradPhi = [0.1.*(2.0.*ucurrent.*ycurrent.-4.0.*ycurrent.+5) 0.1.*(ucurrent.^2.0.-4.0.*ucurrent)]
    Hbig = [1 gradhM']
    cdoublebar = 2.0 .* Hbig * transpose(gradPhi)[1:noInputs+noOutputs]
    Gdoublebar = 2 * Gmat

    LHM = [
        Gdoublebar -transpose(Adoublebar)
        -Diagonal(dual.(cstIn)[1:size(Adoublebar)[1]])*Adoublebar -Diagonal(Adoublebar.*value.(w).-bdoublebar) 
    ]

    LHMinv = inv(LHM)
    k1 = length(LHM[:,1])
    k2 = length(Diagonal(dual.(cstIn)[1:size(Adoublebar)[1]])[1,:])
    RHMb = [zeros(k1-k2,k2)
    -Diagonal(dual.(cstIn)[1:size(Adoublebar)[1]])] #Matrix(1.0I, size(bdoublebar)[1])  
    RHMc = [-Matrix(1.0I, 1,1)
    zeros(k2,1)]

    dallb = LHMinv*RHMb    ####both w and λ
    dallc = LHMinv*RHMc    ####both w and λ

    dwb = dallb[1,:] ####picking only w
    dwc = dallc[1,:] ####picking only w

    dwA = []
    for i = 1:size(Adoublebar)[1]
            dAm = zeros(size(Adoublebar))
            dAm[i] = 1.0 ######Do for every element of Adoublebar

            RHMA = [
                transpose(dAm)* dual.(cstIn)[1:size(Adoublebar)[1]]
                Diagonal(dual.(cstIn)[1:size(Adoublebar)[1]])* dAm.* value.(w)
            ]
            dallAtemp = LHMinv * RHMA
            dwAtemp=dallAtemp[1,1:end] ####picking only w
            push!(dwA, dwAtemp)
    end
    dwA = reduce(hcat,dwA)

    dwG = []
    for i = 1:1 ######Set correct G size
            dGm = zeros(size(Gdoublebar))
            dAm = zeros(size(Adoublebar))

            dGm[i] = 1.0 ######Do for every element of Adoublebar
            RHMG = -[
                dGm.*value.(w)
                0.0.*Diagonal(dual.(cstIn)[1:size(Adoublebar)[1]])* dAm.* value.(w)
            ]
            dallGtemp = LHMinv * RHMG
            dwGtemp=dallGtemp[1,1:end] ####picking only w
            push!(dwG, dwGtemp)
    end
    dwG = reduce(hcat,dwG)

    # ####Zygote computations
    gradhTf = (u,β,α) -> 2*u.+4
    gradhMf = (u,β,α) -> gradhTf(u,β,α) .+ β ###########This line defines how the mismatched gradient is parametrised as a function of β

    Adoublebarf = (u,β,α) -> vcat(Am.* α, Cm.* α .* gradhMf(u,β,α))
    bdoublebarf = (u,β,y,α) -> vcat(Bm - Am.* u, Dm .- Cm .* y) #####Dots are needed because I have only one output
    
    gradPhif = (u,β,y,α) -> [  0.1.*(2.0.*u.*y.-4.0.*y.+5) 0.1.*(u.^2.0.-4.0.*u)]
    Hbigf = (u,β,α) -> [1 gradhMf(u,β,α)']
    cdoublebarf = (u,β,ycurrent,α) -> 2.0 .* Hbigf(u,β,α) * transpose(gradPhif(u,β,ycurrent,α))[1:noInputs+noOutputs]

    Gdoublebarf = (u,β,α) -> 2 * Gmat

    ####################Derivatives wrt α
    dcα = jacobian( b -> cdoublebarf(ucurrent,β,ycurrent,b), α)
    dcu = jacobian( a -> cdoublebarf(a,β,ycurrent,αval), ucurrent)
    dcy = jacobian( c -> cdoublebarf(ucurrent,β,c,αval), ycurrent)
    dyu = jacobian( x -> realSystem1D(x), ucurrent)

    dbα = jacobian( b -> bdoublebarf(ucurrent,β,ycurrent,b), α)
    dbu = jacobian( a -> bdoublebarf(a,β,ycurrent,αval), ucurrent)
    dby = jacobian( c -> bdoublebarf(ucurrent,β,c,αval), ycurrent)
    dyu = jacobian( x -> realSystem1D(x), ucurrent)

    dAα = jacobian( b -> Adoublebarf(ucurrent,β,b), α)
    dAu = jacobian( a -> Adoublebarf(a,β,αval), ucurrent)
    
    dQPdα = (dwb'*(dbα[1]).+dwc.*(dcα[1]).+dwA*(dAα[1]))
    dQPdu = (dwb'*(dbu[1].+dby[1].*dyu[1]).+dwc.*(dcu[1].+dcy[1].*dyu[1]).+dwA*(dAu[1]))
    
    dQPdα = reshape(dQPdα,:,1)
    dQPdu = reshape(dQPdu,:,1)
    BigBlock = dQPdα
    
    if timecurrent >1
        BigBlock = BlockDiagonal([dQPdα,kron(Matrix(1I, timecurrent-1, timecurrent-1),dQPdu)])
    end
    dwdα = BigBlock*dudαcurrent
    return results,dwdα


end


function OFOoptim(α, ΔT, tuningHorizon, ucurrent, ycurrent,maps,objFcn,bval)
    kMax = tuningHorizon ./ ΔT - 1

    yAll = [ycurrent]
    uAll = [ucurrent]
    derivucurrent = 1
    derivwcurrent = 0
    derivuAll = derivucurrent
    derivuAllα = derivucurrent
    derivwAll = []
    derivΦAll = []
    derivΦuAll = []
    allres = zeros(Int(kMax+1),1)
    derivwmatrix = zeros(Int(kMax+1),Int(kMax+1))
    derivumatrix = zeros(Int(kMax+1),Int(kMax+1))

    testAll = []
    for k = 0:kMax
        projResult, derivwcurrent = internalProjectionFO(α, ucurrent, ycurrent, maps, objFcn, derivuAll,derivuAllα, bval)
        push!(derivwAll, derivwcurrent)
        allres[Int(k+1)] = projResult[1]

        derivwmatrix[Int(k+1),1:Int(k+1)] = reverse(derivwAll[Int(k+1)])
        derivumatrix[Int(k+1),:]  = allres .+ α*sum(derivwmatrix;dims=2)
        derivuAllα = derivumatrix[Int(k+1),1:Int(k+1)]
        derivuAllα = vcat([1.0],reverse(derivuAllα))
        dΦu = jacobian(a -> Φ1D(a,ycurrent), ucurrent)[1]
        dΦy = jacobian( b -> Φ1D(ucurrent,b), ycurrent)[1]
        dyu = jacobian( x -> realSystem1D(x), ucurrent)[1]


        dPhidu = (dΦu.+dΦy.*dyu)
        dPhidu = reshape(dPhidu,:,1)
        BigBlockPhi = kron(Matrix(1I, Int(k+1), Int(k+1)),dPhidu)

        dΦαAll = BigBlockPhi*derivuAllα[2:end]
        dΦα = sum(dΦαAll[1:end])
        push!(derivΦAll, dΦα)
        push!(derivΦuAll, dΦu)

        uucstr = ucurrent .+ α *  projResult
        unew = uucstr
        ySol = realSystem1D(unew)
        ycurrent = ySol
        ucurrent = unew

        yAll = push!(yAll, ycurrent[1])
        uAll = push!(uAll, ucurrent[1])
        testAll = dΦαAll
    end
    return yAll, uAll, derivuAllα, derivwAll, derivΦAll, derivΦuAll, testAll

end

distVal = [1;1]

Am = [ 1 ; -1]
Bm = [5;5] #####Unconstrained case
# Bm = [2;2] #####Uncomment for the constrained case

Cm = [1.0;-1.0]
Dm = [3; 5.0]

β = 0.0

noInputs = 1
noOutputs=1

Gmat = 1

uC = -0.63

yC = realSystem1D(uC)

ΔT = 1
tHVector = [50,100,150]

rsC = u -> realSystem1D(u)
hmC = u -> hMap1D(u)
PhiC = (u,y) -> Φ1D(u,y)

αVector1 = range(0.0001,0.025, step=1e-4)

αVector = collect(αVector1)

normTime = []

plot()
for tInst in eachrow(tHVector)
    tH = tInst[1]
    ΦAllFinalDer = []
    ΦAllFinal = []

    for alp in eachrow(αVector)
        α = alp[1]
        resOFOyId, resOFOuId, resOFOderivuId, resOFOderivwId, resOFOderivΦId, resOFOderivΦuId, allPhilast = OFOoptim(α, ΔT, tH, uC, yC, rsC, PhiC, 0.0)

        uSolId = i -> resOFOuId[i]
        ySolId = i -> resOFOyId[i]
        push!(ΦAllFinalDer, reduce(vcat, resOFOderivΦId)'[end])
        push!(ΦAllFinal, PhiC.(uSolId.(tH), ySolId.(tH)))

    end
    finiteDif = diff(ΦAllFinal) ./ diff(αVector)
    scatter!(αVector[1:end-1], finiteDif)
    plot!(αVector, ΦAllFinalDer,lw=4)
    MSE = sum((ΦAllFinalDer[2:end] - finiteDif) .^ 2) / size(finiteDif)[1]
    MSEnorm = norm(ΦAllFinalDer[2:end] - finiteDif)
    MSEpercent = maximum(abs.(ΦAllFinalDer[2:end] - finiteDif) ./ finiteDif * 100)
    push!(normTime, MSEpercent)
end

xlabel!("alpha")
ylabel!("dPhidalpha")
