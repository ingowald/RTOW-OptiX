# if size(ARGS,1) != 3
  # println("Usage: julia process_tensors.jl <input_filename> <output_filename> <scale_factor>");
  # println("Assumes that input is a CSV file.");
  # exit();
# end

using DataFrames;
using LinearAlgebra;
using Printf;

#See equation 2 of the Kindlmann paper
function compute_anisotropy_coeffs(eigenvals::Array{Float64,1})
  lambdas = sort(eigenvals, rev=true)
  sum_l = lambdas[1] + lambdas[2] + lambdas[3] 
  c_l = (lambdas[1] - lambdas[2])/sum_l;
  c_p = 2*(lambdas[2] - lambdas[3])/sum_l;
  c_s = 3*lambdas[3]/sum_l;

  return c_l, c_p, c_s;
end

#mutates out_Q and out_D_modified
function get_tensor_warp!(T, e_min, e_max, out_Q, out_D_modified, filter_isotropy::Bool)
  S = (T + T')/2; #S is the symmetric part of T
  eigenDecomp = eigen(S);
  out_Q[:] = eigenDecomp.vectors;

  modified_eigvals = abs.(eigenDecomp.values) 
  modified_eigvals = clamp.(modified_eigvals, e_min, e_max);

  out_D_modified[:] = Diagonal(modified_eigvals)

  if filter_isotropy
    c_l, c_p, c_s = compute_anisotropy_coeffs(modified_eigvals)

    # println(c_l)
    # println(c_p)
    # println(c_s)
    println(eigenDecomp.vectors)
    println("*****")
    
    # See the last paragraph of Section 4 of Kindlmann's paper. The below
    # threshold hides isotropic tensors.
    if c_l + c_p <= 0.5 
      return false;
    end
  end

  return true;
end

# in_file: input CSV file
# out_file: CSV file with Q*D_clamped as the 3x3 matrix
# min_lambda: lower bound for eigenvalue (after taking absolute value) 
# max_lambda: upper bound for eigenvalue (after taking absolute value) 
function process_tensors(in_file, out_file, scale, min_lambda, max_lambda, filter_isotropy::Bool)
  println("Processing!");
  tensor_df = readtable(in_file);
  out_handle = open(out_file, "w");

  for i in 1:size(tensor_df,1)
    curr_row = tensor_df[i,1:9];
    T_flat = [x for x in curr_row];
    T = scale*transpose(reshape(T_flat, 3, 3))

    Q = Array{Float64,2}(undef, 3,3);
    fill!(Q,NaN);
    D_clamped = Array{Float64,2}(undef, 3,3);
    fill!(D_clamped,NaN);
    valid = get_tensor_warp!(T, min_lambda, max_lambda, Q, D_clamped, filter_isotropy);

    if !valid
      continue;
    end

    # If determinant is negative, Flip ONE of the columns. It will still be an
    # eigenvector with the same eigenvalue. (If v is an eigenvector, -v is also
    # an eigenvector)
    if(det(Q) < 0)
      Q[:,1] = -Q[:,1]
    end

    if(det(Q) < 0)
      println("Negative determinant!");
    end

    #take a transpose here because Julia operates
    #in row major order
    A = transpose(Q*D_clamped); 

    if(det(A) < 0)
      println("Negative determinant!");
    end

    out_s = @sprintf "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" A[1] A[2] A[3] A[4] A[5] A[6] A[7] A[8] A[9] tensor_df[i,10] tensor_df[i,11] tensor_df[i,12] ;       
    write(out_handle, out_s)
  end

  close(out_handle);
end

# process_tensors(ARGS[1], ARGS[2], parse(Float64,ARGS[3]), 5e-2, Inf, true)
