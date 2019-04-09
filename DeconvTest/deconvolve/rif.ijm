/*
This macro runs Regularized Inverse Filter (RIF) from DeconvolutionLab2 plugin with given parameters.
*/
run("Misc...", "divide=Infinity save");
arg = getArgument();
args = split(arg, ' ');
path_input = args[0];
path_psf = args[1];
lambda = args[2];
path_output = args[3];
filename_output = args[4];

command = "-image file " + path_input + 
          " -psf file " + path_psf +
          " -algorithm RIF " + lambda + " " +
          " -monitor no " +
          " -out stack " + filename_output +
          " -path " + path_output;

run("DeconvolutionLab2 Run", command);
wait(1000);  // wait for 1 second before exiting the process
eval("script", "System.exit(0);");
