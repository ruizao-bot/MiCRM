using Distributions
import Pkg
Pkg.add("DifferentialEquations")
using DifferentialEquations
using LinearAlgebra
include("sim_frame.jl")
N = 100
M = 5
L = 0.3
var_Bv = 0.01
niche = rand(M, N)
diri = zeros(N, M)
var_B = [0.01, 0.1, 0.2, 0.5, 1.0]

function generate_var_param(N, M, L, var_Bv, niche)
    diri = zeros(Float64, N, M)
    for i in 1:N
        diri[i,:] = rand(Dirichlet(niche[:,i]),1)
    end

    u_sum = rand(Truncated(Normal(2.5, var_Bv), 0, Inf), N)
    u = diri.*u_sum
    m = rand(Truncated(Normal(1, var_Bv), 0, Inf), N)
    l = def_l(N, M, L)
    ρ = ones(M); ω = zeros(M)
    λ = fill(L, 1, M)

    return (N=N, M=M, u=u, m=m, l=l, ρ=ρ, ω=ω, λ=λ)
end

function def_l(N, M, L)
    l = zeros(N, M, M)
    for i in 1:N, α in 1:M
        weights = rand(Dirichlet(ones(M)))
        l[i, α, :] .= L * weights
    end
    return l
end

function dxx!(dx, x, p, t)

    for i =1:p.N
        dx[i] = 0.0
        dx[i] = -p.m[i]*x[i]

        for α = 1:p.M
            dx[i] += x[i]*x[α + p.N]*p.u[i, α]
            for β=1:p.M
                dx[i] += -x[i]*x[α + p.N]*p.u[i, α]*p.l[i, α, β]
            end
        end
    end
    for α = 1:p.M
        dx[α + p.N] = 0.0
        dx[α + p.N] = p.ρ[α] - (x[α + p.N] * p.ω[α])

        for i=1:p.N
            dx[α + p.N] += -p.u[i, α]*x[α+p.N]*x[i]
            for β=1:p.M
                dx[α + p.N] += x[β + p.N] * x[i] * p.u[i, β] * p.l[i, β, α]
            end
        end
    end
end
tspan = (0.0, 1.5e8)
x0 = vcat(fill(0.1, N), fill(1, M))
Ci = fill(0.1, N)
condition(du, t, integrator) = norm(integrator(t, Val{1})) <= eps()
affect!(integrator) = terminate!(integrator)
cb = DiscreteCallback(condition, affect!)

p = generate_var_param(N, M, L, 0.1, niche)
        ϵ = (p.u * x0[N+1:N+M] .* (1 - 0.3) .- p.m) ./ (p.u * x0[N+1:N+M])
        ## run simulation
        prob = ODEProblem(dxx!, x0, tspan, p)
        sol =solve(prob, AutoVern7(Rodas5()), save_everystep = false, callback=cb)

        p_lv = Eff_LV_params(p=p, sol=sol);
        r = p_lv.r
        C = sol.u[length(sol.t)][1:N]
	 sur = (1:N)[C .> 1.0e-7]

C = sol.u[length(sol.t)][1:N]
f = Figure(fontsize = 35, resolution = (1200, 900));
ax = Axis(f[1,1], xlabel = "CUE", ylabel = "r", xlabelsize = 50, ylabelsize = 50)
scatter!(ax, ϵ,r, color = ("#4F363E",0.5), markersize = 15, label = "All")
scatter!(ax, ϵ[sur],r[sur], color = ("#EF8F8C",1), markersize = 15, label = "Survivor")
axislegend(position = :lt)
f

using CairoMakie
N = p.N
ts = sol.t
us = sol.u

f2 = Figure(fontsize=30, resolution=(1200, 800))
ax2 = Axis(f2[1,1], xlabel="Time", ylabel="Species Abundance", title="Biomass Dynamics of All Species")

for i in 1:N
    lines!(ax2, ts, [u[i] for u in us], linewidth=1, alpha=0.5)
end

f2


########################
using LinearAlgebra
using Random
using Distributions
using DifferentialEquations
using PyPlot
using .ParamModule  # 假设 param 模块已经用 Julia 实现
using .CUEModule    # 假设 CUE 模块也已实现

########## New elv with indice selection #########
using LinearAlgebra
using Random
using Distributions
using DifferentialEquations
using PyPlot
using .ParamModule  # 假设 param 模块已经用 Julia 实现
using .CUEModule    # 假设 CUE 模块也已实现

Random.seed!(37)
N_pool = 1000
M_pool = 50
λ = 0.3
N_modules = 1
s_ratio = 1.0
N = 100
M = 5

m = rand(Truncated(Normal(1.0, 0.01), 0.0, Inf), N)

u_pool = modular_uptake(N_pool, M_pool, N_modules, s_ratio)
l_pool = generate_l_tensor(N_pool, M_pool, N_modules, s_ratio, λ)

rho_pool = fill(0.5, M_pool)
omega_pool = fill(0.5, M_pool)

species_indices = randperm(N_pool)[1:N]
resource_indices = randperm(M_pool)[1:M]

u = u_pool[species_indices, resource_indices]
u .= 2.5 .* u ./ sum(u, dims=2)

l = l_pool[species_indices, resource_indices, resource_indices]

lambda_alpha = fill(λ, M)
rho = rho_pool[resource_indices]
omega = omega_pool[resource_indices]

function dCdt_Rdt!(du, u, p, t)
    N, M, uptake, leak, lambda_alpha, rho, omega, m = p
    C = u[1:N]
    R = u[N+1:end]
    dCdt = zeros(N)
    dRdt = zeros(M)

    for i in 1:N
        dCdt[i] = sum(C[i] * R[α] * uptake[i, α] * (1 - lambda_alpha[α]) for α in 1:M) - C[i] * m[i]
    end

    for α in 1:M
        dRdt[α] = rho[α] - omega[α] * R[α]
        dRdt[α] -= sum(C[i] * R[α] * uptake[i, α] for i in 1:N)
        dRdt[α] += sum(C[i] * R[β] * uptake[i, β] * leak[i, β, α] for i in 1:N, β in 1:M)
    end

    du[1:N] = dCdt
    du[N+1:end] = dRdt
end

C0 = fill(0.01, N)
R0 = fill(1.0, M)
Y0 = vcat(C0, R0)
params = (N, M, u, l, lambda_alpha, rho, omega, m)
tspan = (0.0, 600.0)

prob = ODEProblem(dCdt_Rdt!, Y0, tspan, params)
sol_mcm = solve(prob, saveat=300)

C_hat = sol_mcm[end][1:N]
R_hat = sol_mcm[end][N+1:end]

# Compute D matrix
D = zeros(M, M)
for a in 1:M, γ in 1:M
    if a == γ
        D[a, a] = omega[a] + sum(C_hat[i] * u[i, a] for i in 1:N)
    else
        D[a, γ] = -sum(C_hat[i] * u[i, γ] * l[i, γ, a] for i in 1:N)
    end
end

# ∂R/∂C
partial_R_C = zeros(M, N)
for j in 1:N
    v_j = [-R_hat[α] * u[j, α] + sum(R_hat[β] * u[j, β] * l[j, β, α] for β in 1:M) for α in 1:M]
    partial_R_C[:, j] .= D \ v_j
end

# α_ij
alpha = zeros(N, N)
for i in 1:N, j in 1:N
    alpha[i, j] = sum(u[i, a] * (1 - lambda_alpha[a]) * partial_R_C[a, j] for a in 1:M)
end

# r_i
r = zeros(N)
for i in 1:N
    growth = sum(u[i, a] * (1 - lambda_alpha[a]) * R_hat[a] for a in 1:M)
    interaction = sum(alpha[i, j] * C_hat[j] for j in 1:N)
    r[i] = growth - m[i] - interaction
end

function dCdt_elv!(dC, C, p, t)
    alpha, r = p
    for i in 1:length(C)
        dC[i] = C[i] * (r[i] + sum(alpha[i, j] * C[j] for j in 1:length(C)))
    end
end

prob_elv = ODEProblem(dCdt_elv!, C0, tspan, (alpha, r))
sol_elv = solve(prob_elv, saveat=300)
