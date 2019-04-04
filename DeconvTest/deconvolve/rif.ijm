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
command = "-image file " + path_input + 
	" -psf file " + path_psf + 
	" -algorithm RIF " + lambda + " " + 
	" -display yes" + 
	" -monitor none";

run("DeconvolutionLab2 Run", command);
saveAs("tiff", path_output);
eval("script", "System.exit(0);");
