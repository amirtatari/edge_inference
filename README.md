# edge_inference
An inference application using Tensorflow-lite library as its backbone.

## Building the application
Use the following commands to build the project. To build with CPU delegate:

`mkdir -vp build && cmake -DCMAKE_BUILD_TYPE=RELEASE -DDELEGATE_TYPE=CPU -S . -B build && cmake --build build`

To build with GPU delegate:

`mkdir -vp build && cmake -DCMAKE_BUILD_TYPE=RELEASE -DDELEGATE_TYPE=GPU -S . -B build && cmake --build build`
