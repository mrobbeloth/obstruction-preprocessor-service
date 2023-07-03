# obstruction-preprocessor-service
This service prepares images for characterization by obstruction algorithms
# gpu-integration-testing
This branch can be integrated into your build similar to the research server 
by first ensuring you have nvidia toolkit 10.0 installed. Download the code 
from the .zip option. Ensure that your selected g++ installation is below 
version 7 (we recommend version 6.4.0, or version 6). Simply run "bash make.sh"
command inside the root directory of the extracted project and it should build. 
I must stress that any improvement on this codebase must use c++11 libraries
(The current compatible version of c++ with NVIDIA's nvcc compiler for GPU support)
