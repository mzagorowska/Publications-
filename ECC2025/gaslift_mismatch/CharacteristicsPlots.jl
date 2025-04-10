######Script file to obtain the characteristics from the paper "Sensitivity of Online Feedback Optimization to time-varying parameters" by M. Zagorowska and L. Imsland
######Contact person: m.a.zagorowska@tudelft.nl

using Plots

function hMap(u) 
    normalisedU = (u.-muAll'[:,1])./muAll'[:,2]
    uMatrix = ones(size(normalisedU))
    for ni=1:4
        uMatrix = hcat(normalisedU.^ni,uMatrix)
    end
    allWells = diag(pAll*uMatrix')
    y= [ 
        sum(allWells[1:2]);
        sum(allWells[3:end])
    ]
    return y,allWells
end

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

######Lower and upper bound for each well
ub = [9576;11745;5972;7377;9043]
lb = [1157;6819;2714;2399;4125]

######Computing and plotting the characteristics of individual wells
allWells = zeros(5,100)
plot()
for ss = 1:size(pAll)[1]
    uLong = collect(range(lb[ss],ub[ss],100))
    normalisedU = (uLong.-muAll'[ss,1])./muAll'[ss,2]
    uMatrix = ones(size(normalisedU))
    for ni=1:4
        uMatrix = hcat(normalisedU.^ni,uMatrix)
    end

    for k=1:100
    test = reshape(pAll[ss,:],1,:)*uMatrix[k,:]
    allWells[ss,k] = test[1]
    end
    plot!(uLong,allWells[ss,:],lw=2, label="u_"*string(ss))
end
xlabel!("Gas rate Sm^3 h^-1")
ylabel!("Oil rate Sm^3 day^-1")

plot!()