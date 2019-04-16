/*
This macro print the version of ImageJ in a `version.txt` file in a given directory.
*/
run("Misc...", "divide=Infinity save");
arg = getArgument();
args = split(arg, ' ');
path_output = args[0];
version = getVersion;
f = File.open(path_output + 'Fiji_version.txt');
print(f,version);
File.close(f);

eval("script", "System.exit(0);");