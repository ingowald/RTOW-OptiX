if size(ARGS,1) != 1
  println("Usage: julia process_tensors.jl <filename>.csv");
  exit();
end

using DataFrames;

tensor_df = readtable(ARGS[1]);
for i in 1:size(tensor_df,1) 
  curr_row = tensor_df[i,:]
  println(curr_row)
end
