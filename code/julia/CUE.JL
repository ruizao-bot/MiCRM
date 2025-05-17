module CUEModule

function compute_CUE(u::Array{Float64,2}, R0::Vector{Float64}, lambda_alpha::Array{Float64,2}, m::Vector{Float64})
    total_uptake = sum(u .* R0', dims=2)[:]
    net_uptake = sum(u .* R0' .* (1 .- lambda_alpha), dims=2)[:] .- m
    return net_uptake ./ total_uptake
end

function compute_community_CUE1(sol, N::Int, rho::Vector{Float64}; num_points::Int=50, return_all::Bool=false)
    C_values = sol.u[end][1:N, :]
    R_values = sol.u[end][N+1:end, :]

    t = sol.t
    time_indices = round.(Int, range(1, length(t), length=num_points))
    C_selected = C_values[:, time_indices]
    R_selected = R_values[:, time_indices]

    dR = sum(R_selected; dims=1)[1:end-1] .- sum(R_selected; dims=1)[2:end]
    dt = t[time_indices[2:end]] .- t[time_indices[1:end-1]]
    dR += sum(rho) .* dt

    dC = sum(C_selected; dims=1)[2:end] .- sum(C_selected; dims=1)[1:end-1]
    community_cue = dC ./ dR
    integral = trapz(t[time_indices[2:end]], community_cue)

    return return_all ? (time_indices, community_cue, integral) : integral
end

function compute_community_CUE2(sol, N::Int, u::Matrix{Float64}, R0::Vector{Float64}, l::Array{Float64,3}, m::Vector{Float64})
    C_final = sol.u[end][1:N]
    total_uptake = sum(u .* R0', dims=2)[:]
    net_uptake = sum(u .* R0' .* (1 .- sum(l, dims=2)), dims=2)[:] .- m
    species_CUE = net_uptake ./ total_uptake
    community_CUE = dot(C_final, species_CUE) / sum(C_final)
    return community_CUE, species_CUE
end

end  # module
