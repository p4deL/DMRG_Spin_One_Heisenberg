using ITensors
using CSV, DataFrames, Tables
using Plots
gr()

base_output_path = "iTensor/output"
koppa = 1.0
nu = 1.0
#hc_guess = 0.9875
#hc_guesses = [0.98 0.985 0.99 0.9925]
hc_guesses = [0.375 0.375 0.375 0.375]
#hc_guesses = [0.44 0.44 0.44 0.44]
nonuni_x = 1.5
σ = -0.4
n_steps = 30


function create_op_sum(sites, mag_field, σ)
    N = length(sites)
    
    # H: transverse field
    os = OpSum()
    for i=1:N
        os += mag_field,"Z",i
        #print(i, ' ')
    end
    #print('\n')

    # Ising interactions
    for i=1:N-1
        for δ=1:N-i
            os += 1.0*δ^(-1.0-σ),"X",i,"X",i+δ
        #    #print(i, i+δ,' ')
        end
        #print('\n')
        #os += 1,"X",i,"X",i+1
    end
    return os
end


let 
    #Ls = [8 12 16 20]
    #Ls = [6 10 14 18]
    #Ls = [12 14 16 18]
    #Ls = [120 160 200 240 280]
    Ls = [96 128 196 256]
    #Ls = [96]

    # center
    #scaling_factor = L^(koppa/nu)/Ls[begin]^(koppa/nu)
    #hzs = 0.0:0.05:2.0
    #hzs = 0.95:0.01:1.05
    

    t1 = time();
    eps = 1e-4
   
    fid_sus_lists = Vector[]
    mag_lists = Vector[]
    hzs_list = Vector[]

    for (L, hc_guess) in zip(Ls, hc_guesses)
       
        println("===================================")
        println("L = $(L)")
        println("-----------------------------------")

        # hc_guess scaling?
        #println(hc_guess*L^(koppa/nu)/Ls[end]^(koppa/nu))

        println(hc_guess)
        h_left = hc_guess - nonuni_x*L^(-koppa/nu)
        h_right = hc_guess + nonuni_x*L^(-koppa/nu)
        hzs_L = h_left:(h_right-h_left)/n_steps:h_right
        #hzs_L = 0.65:0.01:0.75
        push!(hzs_list, hzs_L)
        @show h_left
        @show h_right
        @show hzs_L

        #scaling_factor = L^(koppa/nu)/Ls[begin]^(koppa/nu)
        #scaling_factor = 1.
        #hzs = unscaled_hzs*scaling_factor

        sites = siteinds("S=1/2", L; conserve_szparity=true, qnname_szparity="SzP")
        #sites = siteinds("S=1/2", L)
        #@show sites
        #q = QN(("SzP",1))

        fid_sus_L = Float64[]
        mag_L = Float64[]

        for hz in hzs_L
        
            # create OpSum
            os1 = create_op_sum(sites, hz, σ)
            os2 = create_op_sum(sites, hz+eps, σ)

            # constructing the Hamiltonian parts
            H1 = MPO(os1, sites)
            H2 = MPO(os2, sites)

            # dmrg parameters
            nsweeps = 30
            maxdim = [300]
            cutoff = [1E-9]

            # init wavefunction
            # FIXME: random_mps doesn't really make sense as it is in an arbitrary parity sector
            #psi0 = random_mps(sites; linkdims=20, QN=q)
            #states = ["Up" n in 1:L]
            #states = [isodd(n) ? "Up" : "Dn" for n in 1:L]
            states = ["Up" for n in 1:L]
            psi0 = MPS(sites, states)

            # observer to 
            observer = DMRGObserver(;energy_tol=1E-10,minsweeps=5)

            # calc ground-state wave functions
            energy,psi1 = dmrg(H1,psi0; nsweeps,maxdim,cutoff,observer=observer,outputlevel=0)
            energy,psi2 = dmrg(H2,psi0; nsweeps,maxdim,cutoff,observer=observer,outputlevel=0)
            
            #println("hz = $(hz), ϵ = $(energy/L)")
            #push!(energies, energy/N)
        
            # calc fidelity susceptibility
            overlap = abs(inner(psi1,psi2))  # contract the two mps wave functions
            #fid_sus = -2*log(overlap)/(L*eps^2) # fidelity susceptiblity per site
            fid_sus = -2*log(overlap)/(eps^2) # fidelity susceptiblity
            push!(fid_sus_L,fid_sus)
            #println(fid_sus)
            
            # calc magnetization
            #mz = sum(abs.(expect(psi1,"Z")))/L
            #push!(mag_L, mz)

            xxcorr = correlation_matrix(psi1,"X","X")
            stag_xxcorr = [(-1)^(i + j) * xxcorr[i, j] for i in axes(xxcorr, 1), j in axes(xxcorr, 2)]
            mag = sum(stag_xxcorr)/L^2
            push!(mag_L, mag)
            println("mx = $(mag)")

        end
    
        
        push!(fid_sus_lists, fid_sus_L)
        push!(mag_lists, mag_L)
        header_mag = ["h", "m_squared"]
        header_fid = ["h", "fid_suscept"]
        CSV.write("$(base_output_path)/tfim_mag_sigma$(σ)_L$(L).csv", Tables.table(cat(hzs_L,mag_L,dims=2)), header = header_mag)
        CSV.write("$(base_output_path)/tfim_fid_suscept_sigma$(σ)_L$(L).csv", Tables.table(cat(hzs_L,fid_sus_L,dims=2)), header = header_fid)

    end

    elapsed_time = time() - t1
    println("Elapsed time: $(elapsed_time) seconds")
    
    labels = ["L=$(L)" for L in Ls]
    plot(hzs_list, fid_sus_lists, label=labels, yaxis=("fidelity suscept."), xaxis=("h"))
    #plot(hzs_list, mag_lists, label=labels, yaxis=("m^2"), xaxis=("h"))
    #println(fid_sus_list)

end
