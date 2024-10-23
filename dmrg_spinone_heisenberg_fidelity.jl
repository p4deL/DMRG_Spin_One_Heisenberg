using ITensors
using CSV, DataFrames, Tables
using LinearAlgebra

BLAS.set_num_threads(1)

#==================#
# Input parameters #
#==================#
base_output_path = "output"
infflag = true
d_min = -1.5
d_max = -1.5
step_size = 0.02
eps = 1e-4 ############

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

#=function simple_warmup(D, α, Lmax, nsweeps, maxdim, cutoff)
    
    #sites = siteinds("S=1", 2; conserve_sz=true)
    sites = siteinds("S=1", 2)
    os = create_op_sum(sites, D, α)
    H = MPO(os, sites)
    
    af_state = ["Up" "Dn"]
    psi = randomMPS(sites, af_state, linkdims=3)
    energy,psi = dmrg(H, psi; nsweeps,maxdim)
    
    L = 4
    while L <= Lmax
        #sites = siteinds("S=1", L; conserve_sz=true)
        Lold = L-2
        middle = Lold÷2
        sites = siteinds(psi)
        center_site1 = siteinds("S=1",1)
        center_site2 = siteinds("S=1",1)
        center_site1 = replacetags(center_site1, "n=1"=>"n=$(middle+1)")
        center_site2 = replacetags(center_site2, "n=1"=>"n=$(middle+2)")
        new_sites = vcat(sites[1:middle], [center_site1[1],center_site2[1]], sites[middle+1:end])
        for i in middle+3:L
            #println("$(i-2)->$(i)")
            new_sites[i] = replacetags(new_sites[i], "n=$(i-2)"=>"n=$(i)")
        end 

        #@show sites
        #@show new_sites


        af_state = [isodd(n) ? "Up" : "Dn" for n in 1:L]
        #@show psi
        psi_new = randomMPS(new_sites, af_state; linkdims=3)
        #@show psi_new
        # copy

        for i in 1:middle
            println("left")
            println("i=$(i)")
            psi_new[i] = psi[i] 
        end
        #psi_new[middle+1] = psi[middle]
        #psi_new[middle+2] = psi[middle]
        
        #FIXME: Use randomITensor instead
        #FIXME: How to guarantee Sz conservation?

        for i in middle+3:L
            println("right")
            println("i=$(i)")
            psi_new[i] = psi[i-2]
            replacetags!(psi_new[i], "n=$(i-2)"=>"n=$(i)")
            replacetags!(psi_new[i], "l=$(i-3)"=>"l=$(i-1)")
        end
        @show psi_new


        os = create_op_sum(new_sites, D, α) # use add! or something
        H = MPO(os, new_sites) # can I also use add" or something here?
        @show H
        energy,psi = dmrg(H, psi_new; nsweeps,maxdim,cutoff)
        
        L += 2
    end

    return psi
end=#


#=function dmrg_warmup(Lmax, maxdim, D, α)
    sites = siteinds("S=1", 2; conserve_sz=true)
    os = create_op_sum(sites, D, α)
    H = MPO(os, sites)
    #orthogonalize!(H,1)
    @show siteinds(H)

    exit()
    Linds = (i, k)
    Rinds = (j, l)
    #D, U = eigen(H, Linds, Rinds)
    #dl, dr = uniqueind(D, U), commonind(D, U)
    #Ul = replaceinds(U, (Rinds..., dr) => (Linds..., dl))
    #A * U ≈ Ul * D # true


    #Sz1 = op("Sz", sites[1])
    #Sz2 = op("Sz", sites[2])
    #Sz1Sz2 = Sz1*Sz2
    #@show Sz1
    #@show Sz2
    #@show Sz1Sz2




    

end=#

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
  
    #=
    states = [isodd(n) ? "Up" : "Dn" for n in 1:L]
    psi0 = randomMPS(sites, states; linkdims=L)


    psi0_representatives = []
    D_int = [-1.5 -0.1 1.5]
    for D in D_int

        os = create_op_sum(sites, D, α)
	    H = MPO(os, sites)
        
        nsweeps = 25
        maxdim = [10 20 80 200 300 400 500 600]
        cutoff = [1E-5 1E-5 1E-6 1E-7 1E-7 1E-8]        
        noise = [0]

        if D < -0.1
            states = [isodd(n) ? "Up" : "Dn" for n in 1:L]
        else
            states = ["Z0" for n in 1:L]
        end

        psi0 = randomMPS(sites, states; linkdims=L)

        observer = DMRGObserver(;energy_tol=1E-11,minsweeps=10)

        energy,psi = dmrg(H,psi0; nsweeps,maxdim,cutoff,noise=noise,eigsolve_krylovdim=6, eigsolve_maxiter=3,observer=observer,outputlevel=1)

        push!(psi0_representatives, psi)
    end
    =#

    #@show psi0_representatives

    # TODO: Implement dynamic scheduling
    # iterate over Ds in multiple threads
    Threads.@threads for D in Ds
        println("D=$(D)")

        # create OpSum
		@show α
        os1 = create_op_sum(sites, D, α)
        os2 = create_op_sum(sites, D+eps, α)

        #simple_warmup(D, α, L, 2, 40, 1e-6)
        #exit()

	    # constructing the Hamiltonian parts
	    t2 = time()
	    H1 = MPO(os1, sites)
        H2 = MPO(os2, sites)
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
        #maxdim = [10 20 80 200 300 400 500 600]
        #maxdim = [10 20 80 200 300 400 500 600]
        maxdim = [600]
        cutoff = [1E-5 1E-5 1E-6 1E-7 1E-7 1E-8]        
        #noise = [0 0 0 0 0 0 0 0]
        #FIXME works good like that
        # L = 20
        #noise = [1E-3 1E-4 1E-5 1E-6 1E-7 1E-8 1E-9 1E-11]
        #noise = [1E-4 1E-5 1E-6]
        noise = [0]
        
        # L = 40
        #noise = [1E-2 1E-3 1E-4 1E-5 1E-6 1E-7]
        # L = 60
        #noise = [1E-2 1E-4 1E-5 1E-5 1E-6 1E-7]
        
        #maxdim = [10 10 20 50 50 200 200 300 300]
        #cutoff = [1E-5 1E-6 1E-6 1E-7 1E-7 6E-8]        
        #noise = [0 0 0 0 0 0 0 0]

        # init wavefunction
        if D<=0.3
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
        

       #=if D<-0.6
            psi0 = psi0_representatives[1]
        elseif D <= 0.3
            psi0 = psi0_representatives[2]
        else
            psi0 = psi0_representatives[3]
        end=#
        
        #states = ["Z0" for n in 1:L]
        states = [isodd(n) ? "Up" : "Dn" for n in 1:L]
        psi0 = randomMPS(sites, states; linkdims=L)

        # observer to 
        observer = DMRGObserver(;energy_tol=1E-10,minsweeps=5)

        # calc ground-state wave functions
        # TODO: For long-range sytems it might be sensible to increase niter! Not available anymore?
        # Noise can help convegence, introduces peturbation at each step 1E-5->1E-12
        energy,psi = dmrg(H1,psi0; nsweeps,maxdim,cutoff,noise=noise,eigsolve_krylovdim=6, eigsolve_maxiter=5,observer=observer,outputlevel=1)
        energy_eps,psi_eps = dmrg(H2,psi0; nsweeps,maxdim,cutoff,noise=noise,eigsolve_krylovdim=6,eigsolve_maxiter=5,observer=observer,outputlevel=1)

        #psi0 = psi_eps 
        print(energy)

        # calc fidelity susceptibility
        fidelity = calc_fidelity(psi, psi_eps, eps)
        

	    lock(file_lock)
	    CSV.write("$(base_output_path)/spinone_heisenberg_fidelity_alpha$(α)_L$(L).csv",  Tables.table([D fidelity]), append=true)
	    unlock(file_lock)

    end

    elapsed_time = time() - t1
    println("Elapsed time: $(elapsed_time) seconds")

end
