#include("init_dmrg.jl")
#include("fidelity_susceptibility.jl")
#include("fidelity_susceptibility.jl")
using ITensors
using CSV, DataFrames, Tables
using Plots
gr()

#==================#
# Input parameters #
#==================#
base_output_path = "spinone_heisenberg/output"
L = 128
α = 2
d_min = 0.0
d_max = 2.0
step_size = 0.25
#n_steps = 30
eps = 1e-4

# Extract command line arguments
function parse_args()
    args = ARGS
    α = 0
    L = 0

    # Loop through the arguments and extract values
    for (i, arg) in enumerate(args)
        if arg == "--alpha"
            α = parse(Int, args[i + 1])
        elseif arg == "--size"
            L = parse(Int, args[i + 1])
        end
    end
    return alpha, size
end


# TODO
# input parameters: sigma, hc_guess, L, n_steps, scaling range on and off
# save to file after every h, so that nothing lost!
# write script creates joblist -> probably bash script that can be made into a slurm_script

#======================#
# function definitions #
#======================#




# creates operator string needed to init mpo
function create_op_sum(sites, D, α)
    N = length(sites)
    
    # Single ion anisotropy
    os = OpSum()
    for i=1:N
        os += D,"Sz2",i
        #print(i, ' ')
    end

	# AF Heisenberg interactions
    #=for i in 1:N-1
		os += 0.5,"S+",i,"S-",i+1
		os += 0.5,"S-",i,"S+",i+1
		os += 1.0,"Sz",i,"Sz",i+1
    end=#

	
    # staggered long-range AF Heisenberg interactions 
    for i=1:N-1
        for δ=1:N-i
			coupling = (-1.0)^(δ+1)*δ^(-float(α))
			os += 0.5*coupling,"S+",i,"S-",i+δ
			os += 0.5*coupling,"S-",i,"S+",i+δ
			os += 1.0*coupling,"Sz",i,"Sz",i+δ
        end
    end
    
	return os

end


# calculates fidelity susceptibility
function calc_fidelity(psi, psi_eps, eps)
    overlap = abs(inner(psi, psi_eps))  # contract the two mps wave functions
    
    return -2*log(overlap)/(eps^2) # fidelity susceptiblity
end


# calculates von Neumann entanglement entropy
function calc_entropy(psi, cut_idx)
    psi = orthogonalize(psi, cut_idx)
    U,S,V = svd(psi[cut_idx], (linkinds(psi, cut_idx-1)..., siteinds(psi, cut_idx)...))
    SvN = 0.0
    for n=1:dim(S, 1)
      p = S[n,n]^2
      SvN -= p * log(p)
    end

    return SvN

end

function two_op_correlator(psi, sites, i, j)
    if i > j
        i,j = j,i
    end

    # make site i center of orthogonality s.t. everything to the left gives identity upon contraction
    # and everything to the right (of j when inserting Szs)
    orthogonalize!(psi, i)
    Szi = op("Sz", sites[i])
    Szj = op("Sz", sites[j])

    # contraction procedure strating from left
    C = psi[i]
    C *= Szi  # contract Sz with psi2
    idx_right = commonindex(psi[i],psi[i+1]) # virtual index (link) to the right between 
    # prime physical (site) index and right virtual link + contract with psi(i)^†; link to the left is contracted
    C *= dag(prime(prime(psi[i], "Site"), idx_right)) 

    δmax = j-i-1
    if δmax > 0
        for δ in 1:δmax
            idx = i+δ
            C *= psi[idx]
            C *= dag(prime(psi[idx], "Link"))  # prime both links left and right
        end
    end

    # end contraction procedure
    C *= psi[j]
    C *= Szj
    idx_left = commonindex(psi[j],psi[j-1])  # virtual index (link) to the left
    # prime physical (site) index and left virtual index + contract with psi(j)^†; link to the right is contracted
    C *= dag(prime(prime(psi[j], "Site"), idx_left))
    
    return scalar(C)

end

# calculate string-order parameter
function calc_string_order(psi, sites, i, j)
    if i > j
        i,j = j,i
    end

    # make site i center of orthogonality s.t. everything to the left gives identity upon contraction
    # and everything to the right (of j when inserting Szs)
    orthogonalize!(psi, i)

    # contraction procedure strating from left
    C = psi[i]
    C *= op("Sz", sites[i])  # contract Sz with psi2
    idx_right = commonindex(psi[i],psi[i+1]) # virtual index (link) to the right between 
    # prime physical (site) index and right virtual link + contract with psi(i)^†; link to the left is contracted
    C *= dag(prime(prime(psi[i], "Site"), idx_right)) 

    δmax = j-i-1
    if δmax > 0
        for δ in 1:δmax
            idx = i+δ
            C *= psi[idx]
            C *= exp(im*π*op("Sz",sites[idx])) # multiply with exponential
            C *= dag(prime(psi[idx]))  # prime links and site index
            #@show inds(exp(im*π*op("Sz",sites[idx])))
            #@show inds(prime(psi[idx]))

        end
    end

    # end contraction procedure
    C *= psi[j]
    C *= op("Sz", sites[j])
    idx_left = commonindex(psi[j],psi[j-1])  # virtual index (link) to the left
    # prime physical (site) index and left virtual index + contract with psi(j)^†; link to the right is contracted
    C *= dag(prime(prime(psi[j], "Site"), idx_left))
    
    return -1*real(scalar(C)) 
end


# calculate magnetization
#function calc_magnetization(psi)
#
#end

#==============#
# main routine #
#==============#
let

    t1 = time();

    α, L = parse_args()

    println("===================================")
    println("L = $(L)")
    println("-----------------------------------")

    # FIXME different results?
    sites = siteinds("S=1", L; conserve_sz=true)
    #sites = siteinds("S=1", L)

    fidelity_list = Float64[]
    entropy_list = Float64[]
    mag_list = Float64[]

    header = ["D" "fidelity"]
    CSV.write("$(base_output_path)/spinone_heisenberg_fidelity_sigma$(α)_L$(L).csv",  Tables.table(header), header=false)
    
    header = ["D" "SvN"]
    CSV.write("$(base_output_path)/spinone_heisenberg_svn_sigma$(α)_L$(L).csv",  Tables.table(header), header=false)

    header = ["D" "mag"]
    CSV.write("$(base_output_path)/spinone_heisenberg_magnetization_sigma$(α)_L$(L).csv",  Tables.table(header), header=false)

    # iterate over all
    Ds = d_min:step_size:d_max

    for D in Ds
        println("D=$(D)")

        # create OpSum
        os1 = create_op_sum(sites, D, α)
        os2 = create_op_sum(sites, D+eps, α)

        # constructing the Hamiltonian parts
        H1 = MPO(os1, sites)
        H2 = MPO(os2, sites)

        # dmrg parameters
        nsweeps = 30

        # maxdim dominated schedule
        #maxdim = [10 20 80 200 300 400 800]
        #cutoff = [1E-5 1E-8 1E-12 1E-12 1E-12]
        #noise = [0 0 0 0 0 0 0]

        # cutoff domintaed schedule
        #maxdim = [20 80 200 400 800 800]
        #cutoff = [1E-5 1E-6 1E-8 1E-9 1E-10 1E-12]
        #noise = [0 0 0 0 0 0]

        # noise schedule
        #maxdim = [10 20 80 200 300 400 400]
        #cutoff = [1E-5 1E-6 1E-7 1E-8 1E-8 1E-8 1E-8]
        #noise = [1E-5 1E-5 1E-8 1E-9 1E-10 1E-10 1E-10]

        # individual
        maxdim = [10 20 80 200 300 400 500]
        cutoff = [1E-5 1E-5 1E-6 1E-7 1E-7 1E-8]        
        noise = [0 0 0 0 0 0 0 0]
        #noise = [1E-5 1E-5 1E-8 1E-9 1E-10 1E-10 1E-10]
        
        #maxdim = [10 10 20 50 50 200 200 300 300]
        #cutoff = [1E-5 1E-6 1E-6 1E-7 1E-7 6E-8]        
        #noise = [0 0 0 0 0 0 0 0]

        # init wavefunction
        states = [isodd(n) ? "Up" : "Dn" for n in 1:L]
        psi0 = MPS(sites, states)

        # observer to 
        observer = DMRGObserver(;energy_tol=1E-8,minsweeps=5)

        # calc ground-state wave functions
        # TODO: For long-range sytems it might be sensible to increase niter! Not available anymore?
        # Noise can help convegence, introduces peturbation at each step 1E-5->1E-12
        energy,psi = dmrg(H1,psi0; nsweeps,maxdim,cutoff,observer=observer,outputlevel=1)
        energy_eps,psi_eps = dmrg(H2,psi0; nsweeps,maxdim,cutoff,observer=observer,outputlevel=1)

        #energy,psi = dmrg(H1,psi0; nsweeps,maxdim,cutoff,noise,outputlevel=1)
        #energy_eps,psi_eps = dmrg(H2,psi0; nsweeps,maxdim,cutoff,noise,outputlevel=1)

        # calc fidelity susceptibility
        fidelity = calc_fidelity(psi, psi_eps, eps)
        push!(fidelity_list,fidelity)
        CSV.write("$(base_output_path)/spinone_heisenberg_fidelity_sigma$(α)_L$(L).csv",  Tables.table([D fidelity]), append=true)


        # calc von Neumann entropy
        SvN = calc_entropy(psi, L÷2)
        push!(entropy_list, SvN)
        CSV.write("$(base_output_path)/spinone_heisenberg_svn_sigma$(α)_L$(L).csv",  Tables.table([D SvN]), append=true)

        i = L÷4
        j = i + L÷2
        #zzcorr = correlation_matrix(psi, "Sz", "Sz")
        #@show zzcorr[i,j]

        #zzcorr_test = two_op_correlator(psi, sites, i, j)
        #@show zzcorr_test

        string_order = calc_string_order(psi, sites, i, j)
        @show string_order

        # append to csvs
        zzcorr = correlation_matrix(psi,"Sz","Sz")
        stag_zzcorr = [(-1)^(i + j) * 3 * zzcorr[i, j] for i in axes(zzcorr, 1), j in axes(zzcorr, 2)]
        mag = sum(stag_zzcorr)/L^2
        @show mag
        push!(mag_list, mag)
        CSV.write("$(base_output_path)/spinone_heisenberg_magnetization_sigma$(α)_L$(L).csv",  Tables.table([D mag]), append=true)
        #println("mx = $(mag)")
        
        #push!(fid_sus_lists, fid_sus_L)
        #push!(mag_lists, mag_L)
        #header_mag = ["h", "m_squared"]
        #header_fid = ["h", "fid_suscept"]
        #CSV.write("$(base_output_path)/spinone_heisenberg_mag_sigma$(σ)_L$(L).csv", Tables.table(cat(hzs_L,mag_L,dims=2)), header = header_mag)
        #CSV.write("$(base_output_path)/spinone_heisenberg_fid_suscept_sigma$(σ)_L$(L).csv", Tables.table(cat(hzs_L,fid_sus_L,dims=2)), header = header_fid)

    end

    elapsed_time = time() - t1
    println("Elapsed time: $(elapsed_time) seconds")

    #labels = ["L=$(L)" for L in Ls]
    #plot(hzs_list, fid_sus_lists, label=labels, yaxis=("fidelity suscept."), xaxis=("h"))
    #plot(hzs_list, mag_lists, label=labels, yaxis=("m^2"), xaxis=("h"))

end