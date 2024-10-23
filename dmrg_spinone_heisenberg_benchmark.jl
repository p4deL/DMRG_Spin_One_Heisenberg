using ITensors
using CSV, DataFrames, Tables
using LinearAlgebra
using Base.Threads

#BLAS.set_num_threads(4)

#==================#
# Input parameters #
#==================#
base_output_path = "output"
infflag = true



#======================#
# function definitions #
#======================#

# Extract command line arguments
function parse_args()
    args = ARGS
    
    α = 0
    D = 0
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
        
        elseif arg == "--D"
            D = parse(Float64, args[i + 1])
        
        elseif arg == "--size"
            L = parse(Int, args[i + 1])
        end
    end
    return α, D, L
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

#==============#
# main routine #
#==============#
let

    α, D, L = parse_args()

    t1 = time();
    
    println("===================================")
    println("L = $(L)")
    println("α = $(α)")
    println("D = $(D)")
    println("-----------------------------------")

    #sites = siteinds("S=1", L; conserve_sz=true)
    sites = siteinds("S=1", L)

    # create OpSum
    os = create_op_sum(sites, D, α)

    # constructing the Hamiltonian parts
    #t2 = time()
    H = MPO(os, sites)
    construction_time = time() - t2
    #println("construction time: $(construction_time) seconds")

    # dmrg parameters
    nsweeps = 25

    # individual
    maxdim = [50 50 50 50 100 100 100 100 200 200 200 200 300 300 300 300]
    cutoff = [1E-8]        
    noise = [0 0 0 0 0 0 0 0]
    #noise = [1E-5 1E-5 1E-8 1E-9 1E-10 1E-10 1E-10]
    

    # init wavefunction
    if D<=0.45 || α <= 3 
    # Haldane like init states
    # FIXME I could try a Haldane state that does not break symmetry -> superposition
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
    observer = DMRGObserver(;energy_tol=1E-8,minsweeps=5)

    # calc ground-state wave functions
    energy,psi = dmrg(H,psi0; nsweeps,maxdim,cutoff,observer=observer,outputlevel=1)
    
    println("energy E=$(energy)")

    elapsed_time = time() - t1
    println("Elapsed time: $(elapsed_time) seconds")

end
