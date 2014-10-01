write("Fiji: Starting...");

// Get the input.
args = split(getArgument(), ":");
input = args[0];
output = args[1];
write("Fiji: input " + input);

// Open the file and process it.
open(input);
run("8-bit");

write("Fiji: convert to 8-bit");
run("Gaussian Blur...", "sigma=2");

write("Fiji: gaussian blur");
run("Auto Local Threshold", "method=Median radius=40 parameter_1=0 parameter_2=0 white");

write("Fiji: auto local threshold");
run("Skeletonize (2D/3D)");

write("Fiji: skeletonize");
run("Watershed");

// Write the results to a file.
write("Fiji: watershed");
saveAs('tif', output);
write("Fiji: output " + output);
write("Fiji: finished!");

// The command below speeds up exit of this script!
eval("script", "System.exit(0);");
