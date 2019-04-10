/*
This macro runs Richardson-Lucy with Total Variance (RLTV) from DeconvolutionLab2 plugin with given parameters.
*/
run("Misc...", "divide=Infinity save");
arg = getArgument();
arg = replace(arg, '___', "@");
args = split(arg, '@');
path_input = args[0];
path_psf = args[1];
iters = args[2];
lambda = args[3];
path_output = args[4];
filename_output = args[5];

command = "-image file " + path_input +
          " -psf file " + path_psf +
		  " -algorithm RLTV " + iters + " " + lambda + " " +
          " -monitor no " +
          " -out stack " + filename_output +
          " -path " + path_output;

run("DeconvolutionLab2 Run", command);
for (i=0; i<1000; i++){
if (File.exists(path_output + '/' + filename_output + '.tif')){
        eval("script", "System.exit(0);");
    }
    wait(500);
}

