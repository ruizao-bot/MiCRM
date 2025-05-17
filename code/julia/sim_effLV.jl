include("./sim_frame.jl");

N=100
M=50
L = 0.3
### Temp params 
# T=15+273.15; 
ρ_t= [-0.3500, -0.3500]; # realistic covariance
Tr=273.15+13; Ed=3.5 #[-0.1384 -0.1384]
###################################
# Generate MiCRM parameters
tspan = (0.0, 15000.0)
x0 = vcat(fill(0.1, N), fill(1, M))
# here we define a callback that terminates integration as soon as system reaches steady state
condition(du, t, integrator) = norm(integrator(t, Val{1})) <= eps()
affect!(integrator) = terminate!(integrator)
cb = DiscreteCallback(condition, affect!)


################# Template for eff_LV calc ##################
T = 273.15+15
p = generate_params(N, M; f_u=F_u, f_m=F_m, f_ρ=F_ρ, f_ω=F_ω, L=L, T=T, ρ_t=ρ_t, Tr=Tr, Ed=Ed)
## run simulation
prob = ODEProblem(dxx!, x0, tspan, p)
sol =solve(prob, AutoVern7(Rodas5()), save_everystep = false, callback=cb)

@time p_lv = Eff_LV_params(p=p, sol=sol);
all_ℵ = p_lv.ℵ
all_r = p_lv.r

N_sur = sum(all_r .> 0)
sur_ℵ = all_ℵ[all_r.>0, all_r.>0]
mean_ℵ = mean([sur_ℵ[i, j]/diag(sur_ℵ)[i] for i in 1:N_sur for j in 1:N_sur if i != j])
sur_r = all_r[all_r.>0]
mean_r = mean(sur_r) 

pred = sum(sur_r .> ((N_sur - 1)* mean_ℵ)*mean_r/(1+(N_sur-1)*mean_ℵ))

all_CUE = all_r ./(all_r .+ p.m) 
sur_CUE = all_CUE[all_CUE.>0]

m = p.m[all_r.>0]
ϵ_s = ((N_sur - 1)* mean_ℵ)*mean_r./((m.*(1+(N_sur - 1)* mean_ℵ)).+(((N_sur - 1)* mean_ℵ)*mean_r))
pred_ϵ = sum(sur_CUE .> ϵ_s)

bio = sol.u[length(sol.t)][1:N]

CUE = (p.u * x0[N+1:N+M] .* (1-L) .- p.m) ./ (p.u * x0[N+1:N+M])
e = CUE[bio .> 1e-7]
em = p.m[bio .> 1e-7]
u = sum(p.u, dims =2)
eu = u[bio .> 1e-7]


using LsqFit
fitting(x, pr) = pr[1] ./ (pr[2] .* x .+ pr[3])
pr0 = [1.0, 1.0, 1.0]
fit = curve_fit(fitting, m,ϵ_s, pr0)
a, b, c = fit.param
mx = range(minimum(m), stop = maximum(m), length = 100)
ϵy = fitting(mx, [a, b, c])

pred_ϵv = sur_CUE[sur_CUE .> ϵ_s]
pred_ϵm = m[sur_CUE .> ϵ_s]
f = Figure(fontsize = 30, resolution = (1200, 800));
ax = Axis(f[1,1], xlabel = "m", ylabel = "ϵ")
lines!(mx, ϵy, label = "calculated boundary")
# scatter!(m, ϵ_s, label = "")
scatter!(m, sur_CUE, label = "all species")
scatter!(pred_ϵm, pred_ϵv, label = "prediction")
scatter!(em, e, label = "simulation")
axislegend(position = :rt)
f



# # Set initial conditions
# Ci = fill(0.0, N)
# for i in 1:N
#     Ci[i] = 0.1
# end
# # Define ODE problem
# # prob_LV = ODEProblem(LV_dx!, Ci, (0.0, t_span), LV1)
# t_span = 15000
# prob_LV = ODEProblem(LV_dx!, Ci, (0.0, t_span), p_lv)

# # Integrate ODE with callback and terminate if it fails to converge more than 3 times (this is to avoid unncesessary overhead)
# sol_LV = solve(prob_LV, AutoVern7(Rodas5()), save_everystep = false, callback=cb)
# bm = sol_LV.u[length(sol_LV.t)][1:N]
# length(bm[bm.>1e-7])

###############################################################

all = Float64[]; all_pred = Float64[]
# richness = Float64[]; richness_err = Float64[]
@time for i in range(0, stop = 25, length = 26)
        T = 273.15 + i 
        # all = Float64[]
        for j in 1:50
            p = generate_params(N, M; f_u=F_u, f_m=F_m, f_ρ=F_ρ, f_ω=F_ω, L=0.3, T=T, ρ_t=ρ_t, Tr=Tr, Ed=Ed)
            x0 = vcat(fill(0.1, N), fill(1, M))
            prob = ODEProblem(dxx!, x0, tspan, p)
            sol =solve(prob, AutoVern7(Rodas5()), save_everystep = false, callback=cb)
            bm = sol.u[length(sol.t)][1:N]
            ### Effective LV
            p_lv = Eff_LV_params(p=p, sol=sol)
            all_ℵ = p_lv.ℵ; all_r = p_lv.r
            N_sur = sum(all_r .> 0) 
            sur_ℵ = all_ℵ[all_r.>0, all_r.>0]  # eliminate all species with r<0
            mean_ℵ = mean([sur_ℵ[i, j]/diag(sur_ℵ)[i] for i in 1:N_sur for j in 1:N_sur if i != j])
            sur_r = all_r[all_r.>0] 
            mean_r = mean(sur_r) 
            pred = sum(sur_r .> ((N_sur - 1)* mean_ℵ)*mean_r/(1+(N_sur-1)*mean_ℵ)) # upper bound for the number of survivors

            push!(all, length(bm[bm.>1e-7]))
            push!(all_pred, pred)
            print(i, " °C Completed ",j*2, "% ", "\n")
        end
        # rich = mean(all);rich_err = std(all)/sqrt(length(all))
        # push!(richness, rich); push!(richness_err, rich_err)
        # print(i, " °C Complete, ", "richness ",rich,"\n")
    end 


f = Figure(fontsize = 30, resolution = (1200, 800));
ax = Axis(f[1,1], xlabel = "Relative Richness (LV)", ylabel = "Relative Richness (MiCRM)")
scatter!(ax, all_pred./maximum(all_pred), all./maximum(all), color = "#601210", markersize = 15)
f

