using Distributions, Statistics, LinearAlgebra
function WaldStatistic(null_hyp,Mhat,OmegaW,q,samplesize)
    p = size(Mhat,1);
    r=p-q;
    #make left part of test stat
    ev,R = eigen(Mhat,sortby = x -> (-floor(real(x), digits = 6), floor(imag(x), digits = 6)));
    L = inv(R)';
    RI = R[:,1:q];
    RJ = R[:,(q+1):p];
    LI = L[:,1:q];
    LJ = L[:,(q+1):p];
    Lambda = Diagonal(ev);
    LambdaI = Lambda[1:q,1:q];
    LambdaJ = Lambda[(q+1):p,(q+1):p];
    PI = RI*LI'; #conjugate transpose
    Jacobianinv = inv(kron(transpose(LambdaI),I(r)) - kron(I(q),LambdaJ));
    BW = kron(LI,transpose(null_hyp)*RJ) * Jacobianinv * kron(transpose(RI),transpose(LJ));
    Omega = BW * kron(I(p),OmegaW) * BW';
    #C = cholesky(pinv(Omega));
    #Vp = vec(transpose(null_hyp) * PI)' * C.L;
    #result is always real but we want only real output
    return samplesize*real(vec(transpose(null_hyp) * PI)' * pinv(Omega) * vec(transpose(null_hyp) * PI));
end

function Dhat(Mhat,q,element1,element2)
    p = size(Mhat,1);
    r=p-q;
    #make left part of test stat
    #R = eigvecs(Mhat);
    #ev = eigvals(Mhat);
    ev,R = eigen(Mhat,sortby = x -> (-floor(real(x), digits = 6), floor(imag(x), digits = 6)));
    #L = inv(R)';
    #RI = R[:,1:q];
    Dp = R[1:r,1:q] * pinv(R[(r+1):p,1:q]); #this is Dprime
    return (Dp[element2,element1]);
end

function DInvSqrtCovMatEstimate(Mhat,OmegaW,q,element1,element2,BigCovMat=I(size(Mhat)[1]^2),homoskedastic=true)
    p = size(Mhat,1);
    r=p-q;
    #make left part of test stat
    #R = eigvecs(Mhat);
    #ev = eigvals(Mhat);
    ev,R = eigen(Mhat,sortby = x -> (-floor(real(x), digits = 6), floor(imag(x), digits = 6)));
    L = inv(R)';
    RI = R[:,1:q];
    RJ = R[:,(q+1):p];
    #LI = L[:,1:q];
    LJ = L[:,(q+1):p];
    Lambda = Diagonal(ev);
    LambdaI = Lambda[1:q,1:q];
    LambdaJ = Lambda[(q+1):p,(q+1):p];
    Jacobianinv = pinv(kron(transpose(LambdaI),I(r)) - kron(I(q),LambdaJ));
    Ri2inv = pinv(R[(r+1):p,1:q]);
    Dp = R[1:r,1:q] * Ri2inv;
    ups_perp = zeros(ComplexF64,p,r);
    ups_perp[1:r,1:r] = I(r);
    ups_perp[(r+1):p,1:r] = -Dp; #careful here
    ei = zeros(q,1);
    ej = zeros(r,1);
    ei[element1,1] = 1;
    ej[element2,1] = 1;
    BD = kron(ei'  * transpose(Ri2inv), ej' * transpose(ups_perp)*RJ) * Jacobianinv * kron(transpose(RI),transpose(LJ));
    if homoskedastic
        Omega = BD * kron(I(p),OmegaW) * BD';
    else
        Omega = BD * BigCovMat * BD';    
    end
    return (Omega)
end


function tStat(Mhat,M0,OmegaW,q,element1,element2,samplesize)
    numerator = Dhat(Mhat,q,element1,element2) - Dhat(M0,q,element1,element2);
    denominator = sqrt(DInvSqrtCovMatEstimate(Mhat,OmegaW,q,element1,element2)[1,1]);
    return ([sqrt(samplesize) * numerator/denominator sqrt(samplesize)*numerator denominator]);
end

function tStatFeasible(Mhat,D0,OmegaW,q,element1,element2,samplesize)
    numerator = Dhat(Mhat,q,element1,element2) .- D0;
    denominator = sqrt(DInvSqrtCovMatEstimate(Mhat,OmegaW,q,element1,element2)[1,1]);
    return ([sqrt(samplesize) * numerator/denominator sqrt(samplesize)*numerator denominator]);
end


function SimulateGraphtStat(mc_reps,sample_size,g,element1,element2)
    M4 = Matrix(adjacency_matrix(g));
    p = size(M4)[1];
    cov = rand(Wishart(p, Matrix{Float64}(1.0I, p, p)))/100
    #cov = rand(Normal(0,1),p,p)
    #c = cov' * cov / 100;
    noise = MvNormal(zeros(p),cov);
    q = 1;
    r = p - q;

    data_matrices = Array{Float64,3}(undef,sample_size,p,p);
    waldresults = Array{ComplexF64,2}(undef,mc_reps,3);
    fill!(waldresults,0.0+0.0im);

    for mcrep in 1:mc_reps
        
        cov_estimate = Array{Float64,2}(undef,p^2,p^2);
        fill!(cov_estimate,0.0);
        smaller_cov_estimate = Array{Float64,2}(undef,p,p);
        fill!(smaller_cov_estimate,0.0);
      
        for s in 1:sample_size
            #make noisy observation
            matrix_estimate = zeros(p,p);
            data_matrices[s,:,:] = M4 + rand(noise,p);
        end
        
        #find the mean along the first dimension
        matrix_estimate = mean(data_matrices,dims=1)[1,:,:];
        #find the covariance matrix estimate
        for s in 1:sample_size
            demeaned_matrix = data_matrices[s,:,:] .- matrix_estimate;
            cov_estimate += vec(demeaned_matrix)*vec(demeaned_matrix)';
        end
        
        #average across other dimension
        for maus in 1:p
            #1 p+1 2p+1 to p 2p 
            smaller_cov_estimate += cov_estimate[(1+ (maus-1)*p) : (maus*p) , (1+ (maus-1)*p) : (maus*p) ]
        end

        smaller_cov_estimate /= (p*sample_size);

        #ups = real(eigvecs(Matrix(M4),sortby = x -> (-floor(real(x), digits = 6), floor(imag(x), digits = 6)))[:,1]);
        #upsperp = nullspace(ups');

        waldresults[mcrep,:] = tStat(matrix_estimate,M4,smaller_cov_estimate,q,element1,element2,sample_size);

        #waldresults[mcrep] = WaldStatistic(upsperp[:,1:q],matrix_estimate,smaller_cov_estimate,q,sample_size);

    end

    
return waldresults
end