using ITensors
using CSV, DataFrames, Tables
using LinearAlgebra

BLAS.set_num_threads(1)

#==================#
# Input parameters #
#==================#
base_output_path = "output"
infflag = true
d_min = -1.0
d_max = 1.0
step_size = 0.02


#======================#
# function definitions #
#======================#

# Extract command line arguments
function parse_args()
    args = ARGS
    
    α = 0
    L = 0

    # Loop through the arguments and extract values
    for (i, arg) in enumerate(args)
        if arg == "--alpha"
            αstring = args[i + 1]
            if αstring == "inf" || αstring == "Inf"
                global infflag = true
            else
                global infflag = false
            end
            α = parse(Float64, αstring)
        
        elseif arg == "--size"
            L = parse(Int, args[i + 1])
        end
    end
    return α, L
end

# creates operator string needed to init mpo
function create_op_sum(sites, D, α)
    N = length(sites)
    
    # Single ion anisotropy
    os = OpSum()
    for i=1:N
        os += D,"Sz2",i
        #print(i, ' ')
    end

    if infflag
	    # AF Heisenberg NN interactions
        for i in 1:N-1
		    os += 0.5,"S+",i,"S-",i+1
		    os += 0.5,"S-",i,"S+",i+1
		    os += 1.0,"Sz",i,"Sz",i+1
        end

    else
        # staggered long-range AF Heisenberg interactions 
        for i=1:N-1
            for δ=1:N-i
                coupling = (-1.0)^(δ+1)*δ^(-α)
                os += 0.5*coupling,"S+",i,"S-",i+δ
                os += 0.5*coupling,"S-",i,"S+",i+δ
                os += 1.0*coupling,"Sz",i,"Sz",i+δ
            end
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

    # write headers of csv files
    header = ["D" "fidelity"]
    CSV.write("$(base_output_path)/spinone_heisenberg_fidelity_alpha$(α)_L$(L).csv",  Tables.table(header), header=false)
    header = ["D" "SvN"]
    CSV.write("$(base_output_path)/spinone_heisenberg_svn_alpha$(α)_L$(L).csv",  Tables.table(header), header=false)
    header = ["D" "str_order"]
    CSV.write("$(base_output_path)/spinone_heisenberg_stringorder_alpha$(α)_L$(L).csv",  Tables.table(header), header=false)
    header = ["D" "mx" "my" "mz" "m_tot"]
    CSV.write("$(base_output_path)/spinone_heisenberg_magnetization_alpha$(α)_L$(L).csv",  Tables.table(header), header=false)

    # iterate over all
    Ds = d_min:step_size:d_max

    # file lock for multithreating
    file_lock = ReentrantLock()
    
        # TODO: Implement dynamic scheduling
    # iterate over Ds in multiple threads
    Threads.@threads for D in Ds
        println("D=$(D)")

        # create OpSum
		@show α
        os = create_op_sum(sites, D, α)

	    # constructing the Hamiltonian parts
	    t2 = time()
	    H = MPO(os, sites)
    	construction_time = time() - t2
    	println("construction time: $(construction_time) seconds")

        # dmrg parameters
        nsweeps = 50

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
        maxdim = [10 20 80 200 300 400 500 600]
        cutoff = [1E-5 1E-5 1E-6 1E-7 1E-7 1E-8]        
        noise = [0 0 0 0 0 0 0 0]
        #noise = [1E-5 1E-5 1E-8 1E-9 1E-10 1E-10 1E-10]
        
        #maxdim = [10 10 20 50 50 200 200 300 300]
        #cutoff = [1E-5 1E-6 1E-6 1E-7 1E-7 6E-8]        
        #noise = [0 0 0 0 0 0 0 0]

        # init wavefunction
        if D<=0.0 
        # Haldane like init states
        #remainder = L%3
        #Leff = L - remainder
        #pattern = ["Up", "Dn", "Z0"]
        #states = [pattern[(i)%3+1] for i in 0:Leff-1]
        #append!(states, fill("Z0",remainder)) 
        
            # AF init state
            states = [isodd(n) ? "Up" : "Dn" for n in 1:L]
        else
            # large D like GS
            states = ["Z0" for n in 1:L]
        end

        psi0 = MPS(sites, states)

        # observer to 
        observer = DMRGObserver(;energy_tol=1E-10,minsweeps=5)

        # calc ground-state wave functions
        # TODO: For long-range sytems it might be sensible to increase niter! Not available anymore?
        # Noise can help convegence, introduces peturbation at each step 1E-5->1E-12
        energy,psi = dmrg(H,psi0; nsweeps,maxdim,cutoff,observer=observer,outputlevel=1)
        
        # calc von Neumann entropy
        SvN = calc_entropy(psi, L÷2)

        # calc non-local string order parameter
        i = L÷4
        j = i + L÷2
        string_order = calc_string_order(psi, sites, i, j)

        # calc magnetizaiton
        xxcorr = correlation_matrix(psi,"Sx","Sx")
        stag_xxcorr = [(-1)^(i + j) * 3 * xxcorr[i, j] for i in axes(xxcorr, 1), j in axes(xxcorr, 2)]
        magx_sq = sum(stag_xxcorr)/L^2
        yycorr = correlation_matrix(psi,"Sy","Sy")
        stag_yycorr = [(-1)^(i + j) * 3 * yycorr[i, j] for i in axes(yycorr, 1), j in axes(yycorr, 2)]
        magy_sq = sum(stag_yycorr)/L^2
        zzcorr = correlation_matrix(psi,"Sz","Sz")
        stag_zzcorr = [(-1)^(i + j) * 3 * zzcorr[i, j] for i in axes(zzcorr, 1), j in axes(zzcorr, 2)]
        magz_sq = sum(stag_zzcorr)/L^2

        #mag = 3*abs.(expect(psi,"Sz"))[L÷2]
        

	    lock(file_lock)
	    CSV.write("$(base_output_path)/spinone_heisenberg_fidelity_alpha$(α)_L$(L).csv",  Tables.table([D fidelity]), append=true)
        CSV.write("$(base_output_path)/spinone_heisenberg_svn_alpha$(α)_L$(L).csv",  Tables.table([D SvN]), append=true)
        CSV.write("$(base_output_path)/spinone_heisenberg_stringorder_alpha$(α)_L$(L).csv",  Tables.table([D string_order]), append=true)
        CSV.write("$(base_output_path)/spinone_heisenberg_magnetization_alpha$(α)_L$(L).csv",  Tables.table([D np.sqrt(magx_sq) np.sqrt(magy_sq) np.sqrt(magz_sq) np.sqrt(magx_sq + magy_sq + magz_sq)]), append=true)
	    unlock(file_lock)

    end

    elapsed_time = time() - t1
    println("Elapsed time: $(elapsed_time) seconds")

end
