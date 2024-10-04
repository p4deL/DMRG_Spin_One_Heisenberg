using ITensors
using CSV, DataFrames, Tables
using LinearAlgebra

BLAS.set_num_threads(1)

#==================#
# Input parameters #
#==================#
base_output_path = "output"
infflag = true
d_min = 0.0
d_max = 1.0
step_size = 0.025
eps = 1e-4

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
        os1 = create_op_sum(sites, D, α)
        os2 = create_op_sum(sites, D+eps, α)


	    # constructing the Hamiltonian parts
	    t2 = time()
	    H1 = MPO(os1, sites)
        H2 = MPO(os2, sites)
    	construction_time = time() - t2
    	println("construction time: $(construction_time) seconds")

        # dmrg parameters
        nsweeps = 200

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
        #maxdim = [10 20 80 200 300 400 500 600]
        maxdim = [10 20 80 200 300]
        cutoff = [1E-5 1E-5 1E-6 1E-7 1E-7 1E-8]        
        noise = [0 0 0 0 0 0 0 0]
        #noise = [1E-5 1E-5 1E-8 1E-9]
        
        #maxdim = [10 10 20 50 50 200 200 300 300]
        #cutoff = [1E-5 1E-6 1E-6 1E-7 1E-7 6E-8]        
        #noise = [0 0 0 0 0 0 0 0]

        # init wavefunction

        # Haldane like init states
        #remainder = L%3
        #Leff = L - remainder
        #pattern = ["Up", "Dn", "Z0"]
        #states = [pattern[(i)%3+1] for i in 0:Leff-1]
        #append!(states, fill("Z0",remainder)) 
        
        # AF init state
        states = [isodd(n) ? "Up" : "Dn" for n in 1:L]

        # large D like GS
        #states = ["Z0" for n in 1:L]
        
        
        psi0 = MPS(sites, states)

        # observer to 
        observer = DMRGObserver(;energy_tol=1E-12,minsweeps=5)

        # calc ground-state wave functions
        # TODO: For long-range sytems it might be sensible to increase niter! Not available anymore?
        # Noise can help convegence, introduces peturbation at each step 1E-5->1E-12
        energy,psi = dmrg(H1,psi0; nsweeps,maxdim,cutoff,noise=noise,observer=observer,outputlevel=1)
        energy_eps,psi_eps = dmrg(H2,psi0; nsweeps,maxdim,cutoff,noise=noise,observer=observer,outputlevel=1)

        # calc fidelity susceptibility
        fidelity = calc_fidelity(psi, psi_eps, eps)
        

	    lock(file_lock)
	    CSV.write("$(base_output_path)/spinone_heisenberg_fidelity_alpha$(α)_L$(L).csv",  Tables.table([D fidelity]), append=true)
	    unlock(file_lock)

    end

    elapsed_time = time() - t1
    println("Elapsed time: $(elapsed_time) seconds")

end
