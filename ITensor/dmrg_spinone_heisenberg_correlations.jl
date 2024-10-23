using ITensors
using CSV, DataFrames, Tables
using LinearAlgebra
using Base.Threads

BLAS.set_num_threads(4)

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
#=function calc_string_order(psi, sites, i, j)
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
end=#

# calculate string-order correlation
function calc_string_order(psi, sites)    
    L = length(sites)
    
    # make site 1 center of orthogonality s.t. everything to the left gives identity upon contraction
    # and everything to the right (of j when inserting Szs)
    orthogonalize!(psi, 1)

    # contraction procedure strating from left
    C1 = psi[1]
    C1 *= op("Sz", sites[1])  # contract Sz with psi2
    idx_right = commonindex(psi[1],psi[2]) # virtual index (link) to the right between 
    # prime physical (site) index and right virtual link + contract with psi(i)^†; link to the left is contracted
    C1 *= dag(prime(prime(psi[1], "Site"), idx_right)) 

    #  
    str_order = vec(zeros(L-1, 1))

    # TODO check 
    for jmax in 2:L
        
        C = C1
        for j in 2:jmax-1
            C *= psi[j]
            C *= exp(im*π*op("Sz",sites[j])) # multiply with exponential
            C *= dag(prime(psi[j]))  # prime links and site index
            #@show inds(exp(im*π*op("Sz",sites[idx])))
            #@show inds(prime(psi[idx]))

        end

        # end contraction procedure
        C *= psi[jmax]
        C *= op("Sz", sites[jmax])
        idx_left = commonindex(psi[jmax],psi[jmax-1])  # virtual index (link) to the left
        # prime physical (site) index and left virtual index + contract with psi(j)^†; link to the right is contracted
        C *= dag(prime(prime(psi[jmax], "Site"), idx_left))

        str_order[jmax-1] = -1.0*real(scalar(C))

    end


    return str_order
end


#==============#
# main routine #
#==============#
let

    t1 = time();

    α, D, L = parse_args()

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
    t2 = time()
    H = MPO(os, sites)
    construction_time = time() - t2
    println("construction time: $(construction_time) seconds")

    # dmrg parameters
    nsweeps = 25

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
    maxdim = [10 20 80 200 300 400]
    cutoff = [1E-5 1E-5 1E-6 1E-7 1E-7 1E-8]        
    noise = [0 0 0 0 0 0 0 0]
    #noise = [1E-5 1E-5 1E-8 1E-9 1E-10 1E-10 1E-10]
    
    #maxdim = [10 10 20 50 50 200 200 300 300]
    #cutoff = [1E-5 1E-6 1E-6 1E-7 1E-7 6E-8]        
    #noise = [0 0 0 0 0 0 0 0]

    # init wavefunction
    if D<=0.0 
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
    observer = DMRGObserver(;energy_tol=1E-10,minsweeps=5)

    # calc ground-state wave functions
    # TODO: For long-range sytems it might be sensible to increase niter! Not available anymore?
    # Noise can help convegence, introduces peturbation at each step 1E-5->1E-12
    energy,psi = dmrg(H,psi0; nsweeps,maxdim,cutoff,observer=observer,outputlevel=1)
    
    # calc non-local string order 
    strcorr = calc_string_order(psi, sites)

    # calc correlations
    xxcorr = correlation_matrix(psi,"Sx","Sx")[1,2:end]
    yycorr = real(correlation_matrix(complex(psi),"Sy","Sy")[1,2:end])
    zzcorr = correlation_matrix(psi,"Sz","Sz")[1,2:end]

    #@show length(xxcorr)
    #@show length(zzcorr)
    #@show length(strcorr)
    #@show collect(2:L)

    header = ["pos", "corrxx", "corryy", "corrzz", "corrstr"]
    CSV.write("$(base_output_path)/spinone_heisenberg_correlations_D$(D)_alpha$(α)_L$(L).csv",  Tables.table(cat(collect(2:L),xxcorr,yycorr,zzcorr,strcorr,dims=2)), header=header)



    elapsed_time = time() - t1
    println("Elapsed time: $(elapsed_time) seconds")

end
