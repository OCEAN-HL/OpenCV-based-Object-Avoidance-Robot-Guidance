# OpenCV-based-Object-Avoidance-Robot-Guidance

<p align="center">
  <img src="https://user-images.githubusercontent.com/82140899/202213539-acadf116-4cfd-422b-99df-1ea56b622a94.png" width="30%">
</p>

*This work is supported by the European Commission's Horizon 2020 research and innovation program under grant agreement No 871428, 5G-CLARITY project (D-ACAD-052). A demonstration is deployed at M-SHED Museum Bristol*

**A Brief description of this repository:**

This repository provided a sample example to guide robot from point to point while avoiding people on it's path. Different from most existed works that deploy object avoiding functions on the robot, we transfer the video toke by cameras via 5G, Wifi and Lifi to a Sever located at Merchant Venturers Building, University of Bristol. Video stream processing and real-time routing provisioning decisions are packaged in a virtualize network function (VNF) on this server. The decision of the VNF will be sent back to the robot for guidance propose.

Here we only provide a piece of work in the entire system. For more information, you may check the manuscript 'Multi-RAT enhanced Private Wireless Networks with Intent-Based Network ManagementAutomation' accepted in Globecom 2023 workshop.

**Component**

*1. Camera-based Monitoring System:* three cameras setup that captures the live feed of the museum, ensuring maximum coverage of the area.

*2. Virtualized Network Function:* Receives the live feed and processes it in real-time.

*3. Robot:* A machine capable of moving point-to-point within the museum based on instructions from the server.

**Workflow**

*1. Video Conversion to Top-Down View:* The server processes the received video to convert it into a bird view.

*2. Generation of a Digital Map:* Using the bird view, a digital map corresponding to the original video is generated.
The map will be used as a reference for plotting real-time positions of objects and people.

*3. Real-time Position Plotting:* Based on OpenCV, the system identifies and tracks objects and people in the video.
These positions are then plotted on the digital map to give a real-time representation of the museum's interior.

*4. Optimal Path Selection:* Using pathfinding algorithms (e.g. Dijkstra's algorithm), the system computes the optimal path for the robot.
Consideration is given to real-time obstacles, ensuring the robot avoids any obstruction.

*5. Communication to the Robot:* The computed path is sent back to the robot for guiding it through the museum.
