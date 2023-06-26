# obstruction-preprocessor-service
This service prepares images for characterization by obstruction algorithms
# gpu-integration-testing
This branch can be integrated into your build similar to the research server 
by first ensuring you have nvidia toolkit 10.0 installed. Download NsightEclipse.xml,
Makefile, and vectorAdd.cu. Add all three files into their own directory and cd 
inside that directory in your machine. Ensure that your selected g++ installation 
is below version 7 (we recommend version 6.4.0, or version 6). Simply run the "make"
command inside the directory and your project should build. You may run ./vectorAdd
to confirm that the gpu is integrated and that gpu block coding is functional.
