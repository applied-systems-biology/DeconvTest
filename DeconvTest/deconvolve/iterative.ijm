/*
This macro runs Iterative Deconvolve 3D plugin with given parameters.
*/
run("Misc...", "divide=Infinity save");
arg = getArgument();
arg = replace(arg, '___', "@");
args = split(arg, '@');
path_input = args[0];
path_psf = args[1];
path_output = args[2];
normalize = args[3];
perform = args[4];
detect = args[5];
wiener = args[6];
low = args[7];
terminate = args[8];
iterations = args[9];

if (normalize=='TRUE'){
    normalize = "normalize";
}
else{
    normalize = "";
    }

if (perform=='TRUE'){
    perform = "perform";
}
else{
    perform = "";
}


if (detect=='TRUE'){
    detect = "detect";
}
else{
    detect = "";
    }

temp = split(path_input, '/');
title = temp[temp.length-1];

temp = split(path_psf, '/');
psf_title = temp[temp.length-1];

open(path_input);
open(path_psf);
selectWindow(psf_title);
rename("psf");
command = "image=title point=psf output=Deconvolved " +
			normalize + " show log " + perform + " " + detect +
			" wiener=" + wiener + " low=" + low + " z_direction=" + low +
			" maximum=" + iterations + " terminate=" + terminate;
run("Iterative Deconvolve 3D", command);

selectWindow(title);
close();
selectWindow("psf");
close();
saveAs("tiff", path_output);
close();
for (i=0; i<1000; i++){
    if (File.exists(path_output)){
        eval("script", "System.exit(0);");
    }
    wait(500);
}

