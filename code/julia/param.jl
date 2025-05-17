module ParamModule

using Random

function modular_uptake(N::Int, M::Int, N_modules::Int, s_ratio::Float64)
    @assert N_modules ≤ M && N_modules ≤ N "N_modules must be less than or equal to both M and N"

    sR = div(M, N_modules)
    dR = M - (N_modules * sR)
    sC = div(N, N_modules)
    dC = N - (N_modules * sC)

    diffR = fill(sR, N_modules)
    diffR[randperm(N_modules)[1:dR]] .+= 1
    mR = [collect(x:y) for (x, y) in zip(cumsum(diffR) .- diffR .+ 1, cumsum(diffR))]

    diffC = fill(sC, N_modules)
    diffC[randperm(N_modules)[1:dC]] .+= 1
    mC = [collect(x:y) for (x, y) in zip(cumsum(diffC) .- diffC .+ 1, cumsum(diffC))]

    u = rand(N, M)
    for (x, y) in zip(mC, mR)
        u[x, y] .*= s_ratio
    end
    u = u ./ sum(u, dims=2)
    return u
end

function modular_leakage(M::Int, N_modules::Int, s_ratio::Float64, λ::Float64)
    @assert N_modules ≤ M "N_modules must be less than or equal to M"

    sR = div(M, N_modules)
    dR = M - (N_modules * sR)

    diffR = fill(sR, N_modules)
    diffR[randperm(N_modules)[1:dR]] .+= 1
    mR = [collect(x:y) for (x, y) in zip(cumsum(diffR) .- diffR .+ 1, cumsum(diffR))]

    l = rand(M, M)
    for (i, x) in enumerate(mR), (j, y) in enumerate(mR)
        if i == j || i + 1 == j
            l[x, y] .*= s_ratio
        end
    end
    l = λ .* l ./ sum(l, dims=2)
    return l
end

function generate_l_tensor(N::Int, M::Int, N_modules::Int, s_ratio::Float64, λ::Float64)
    return [modular_leakage(M, N_modules, s_ratio, λ) for _ in 1:N]
end

end  # module
