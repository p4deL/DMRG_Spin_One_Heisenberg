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
d_max = 1.5
step_size = 0.1

#======================#
# function definitions #
#======================#

# Extract command line arguments
function parse_args()
    args = ARGS
    
    α = 0
    L1 = 0
    L2 = 0

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
        
        elseif arg == "--size1"
            L1 = parse(Int, args[i + 1])
        elseif arg == "--size2"
            L2 = parse(Int, args[i + 1])
        end
    end
    return α, L1, L2
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


#==============#
# main routine #
#==============#
let

    t1 = time();

    α, L1, L2 = parse_args()

    println("===================================")
    println("L1 = $(L1)")
    println("L2 = $(L2)")
    println("-----------------------------------")

    # FIXME different results?
    sites1 = siteinds("S=1", L1; conserve_sz=true)
    sites2 = siteinds("S=1", L2; conserve_sz=true)
    #sites = siteinds("S=1", L1)


    # write headers of csv files
    header = ["D" "L1" "L2" "SvN1" "SvN2" "ceff"]
    CSV.write("$(base_output_path)/spinone_heisenberg_centralcharge_alpha$(α)_1L$(L1)_2L$(L2).csv",  Tables.table(header), header=false)

    # iterate over all
    Ds = d_min:step_size:d_max

    # file lock for multithreating
    file_lock = ReentrantLock()
    
    # iterate over Ds in multiple threads
    Threads.@threads for D in Ds
        println("D=$(D)")

        # create OpSum
		@show α
        os1 = create_op_sum(sites1, D, α)
        os2 = create_op_sum(sites2, D, α)


	    # constructing the Hamiltonian parts
	    t2 = time()
	    H1 = MPO(os1, sites1)
        H2 = MPO(os2, sites2)
    	construction_time = time() - t2
    	println("construction time: $(construction_time) seconds")

        # dmrg parameters
        nsweeps = 100

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
        states = [isodd(n) ? "Up" : "Dn" for n in 1:L1]
        psi01 = MPS(sites1, states)
        
        states = [isodd(n) ? "Up" : "Dn" for n in 1:L2]
        psi02 = MPS(sites2, states)

        # observer to 
        observer = DMRGObserver(;energy_tol=1E-8,minsweeps=5)

        # calc ground-state wave functions
        # TODO: For long-range sytems it might be sensible to increase niter! Not available anymore?
        # Noise can help convegence, introduces peturbation at each step 1E-5->1E-12
        energy1,psi1 = dmrg(H1,psi01; nsweeps,maxdim,cutoff,observer=observer,outputlevel=1)
        energy2,psi2 = dmrg(H2,psi02; nsweeps,maxdim,cutoff,observer=observer,outputlevel=1)

        # calc von Neumann entropy
        SvN1 = calc_entropy(psi1, L1÷2)
        SvN2 = calc_entropy(psi2, L2÷2)

        # calc approx central charge 
        ceff = 6 * (SvN1 - SvN2)/(log(L1)-log(L2))
        

	    lock(file_lock)
        CSV.write("$(base_output_path)/spinone_heisenberg_centralcharge_alpha$(α)_1L$(L1)_2L$(L2).csv",  Tables.table([D L1 L2 SvN1 SvN2 ceff]), append=true)
	    unlock(file_lock)

    end

    elapsed_time = time() - t1
    println("Elapsed time: $(elapsed_time) seconds")

end
