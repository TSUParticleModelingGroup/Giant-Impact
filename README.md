# Giant-Impact

The giant impact hypothesis is the dominant theory explaining the formation of our Moon. However, its inability to produce an isotopically similar Earth-Moon system with correct angular momentum has cast a shadow on its validity. Computer-generated impacts have been successful in producing virtual systems that possess many of the physical properties we observe. Yet, addressing the isotopic similarities between the Earth and Moon coupled with correct angular momentum has proven to be challenging. Equilibration and evection resonance have been put forth as a means of reconciling the models. However, both were rejected in a meeting at The Royal Society in London. The main concern was that models were multi-staged and too complex. Here, we present initial impact conditions that produce an Earth-Moon system whose angular momentum and isotopic properties are correct. The model is straightforward and the results are a natural consequence of the impact.

## Requirements
* OS: Debian-based Linux distribution
* Hardware: An Nvidia GPU with semi-current drivers installed
* Software: The following dependencies are installed in the startMeUp script:
	* build-essential
 	* mesa-utils
  	* freeglut3-dev
   	* nvidia-cuda-toolkit
    * ffmpeg


## Instructions

Open a terminal in this folder (Giant-Impact). 

Go to this new terminal and type:
```sh
chmod 777 startMeUp
```
Then type in this new terminal:
```sh
./startMeUp
```

### Target and Impactor
-----The following steps set up the Target and Impactor bodies.-----


1. To compile all the code and place it in the correct locations, in the terminal, type:

        ./compileAll 
   
   (Note this only needs to be done once unless you modify the source code.)

2. Now, you should open the setupGeneratingTargetImpactor file and set the parameters for your run. 
   These parameters will set the parameters for the two bodies that will be created.
   Once you are happy with your setup parameters, save the file.

3. Next, you will generate the Target and Impactor with the parameters you set in the 
   setupGeneratingTargetImpactor file. First, the bodies will be created and allowed to settle into place.
   Then, the bodies will be spun, and again, allowed to settle. At this point, the bodies are completely
   independent and have no influence on one another. Once this process has been completed, the Target and 
   The impactor will be put in a time-stamped folder in the TargetImpactorBin. 
   (Note the time stamp only goes down to the minute so if you make more than one
   in a minute the program will throw an error.) To generate the bodies, in the terminal type:
	```sh
	./runGTI
	```
   You should note the date/time to identify the bodies you just created.

5. Open the TargetImpactorBin folder, and you will find the generated folder containing the two bodies.
   If you would like, you can rename this folder to a more descriptive name. Go into this folder. In this
   folder you will find two folders and three files, containing all the information you will need to create
   impacts using the two bodies you created.

### Initial Conditions
-----The following steps set up the initial conditions for an impact.-----

   
1. Open the setupImpactInitialization file and set the parameters for the impact.
   Note, that this just sets the initial positions and velocities of the two bodies to be impacted.
   Don't forget to save the file.

2. Next you will generate an initial impact (ie, time = 0) with the specified positions and velocities.
   To do this, type this in the terminal:
```sh
./runImpactInitialization
```
   This will then create a time-stamped folder in the ImpactBin folder which contains 1 folder and 4 files. 
   (Note the time stamp only goes down to the minute so	if you make more than one in a minute the program will 
   throw an error.).
   If you would like, you can change the name of this folder to something more descriptive.	
	
3. Go into your new folder and follow the instructions outlined below. 

### Adding time and Viewing
-----The following steps are for adding time to an impact or viewing an impact.-----

1. Open a terminal in this folder. 
   A shortcut is to right-click and choose "Open in Terminal".

2. Open the setupImpactAddTime file and set the time to add to the impact. You can also modify
   parameters to change the time step, draw rate, and record rate. Don't forget to save the file.

3. To actually add time to the impact, in the terminal, type 
```sh
	./runImpactAddTime
```	
   Note: this will add additional time to the impact starting from where it left off.

4. At any time, to view the entire impact, in the terminal, type
```sh
	./runImpactViewer
```	
   Note: this viewer does no n-body calculations. It simply reads the impactPosVel file and displays
   frames to the screen.

### ImpactBin info
-----The following is information about the two folders in the ImpactBin directory-----

- The ImpactBin folder contains all impacts created for this setup of bodies.

- The TargetImpactorInformation folder contains 4 files:

  	The setupGeneratingTargetImpactor file is just a copy of the setup file in the Giant-Impact folder 
  	that created this set of bodies.
  
  	The targetImpactorInitialPosVel file (a binary file) contains the positions and velocities of the 
  	two bodies.
  
  	The targetImpactorParameters file contains unit conversion and other pertinent information about 
  	the two bodies.

  	The targetImpactorStats file contains general stats about the two bodies created.
### Impact Information and runStatsFile
-----The following is information about the impactInformation folder and runStatsFile file-----

- The runStatsFile file contains stats about the impact up to the current time. It is updated every time
  runImpactAddTime is called.

- The impactInformation folder contains 3 files:

  	The impactPosVel file (a binary file) contains the positions and velocities of all elements in the impact.
  	
  	The initialStatsFile file contains the initial (ie. time = 0) stats of the impact. 
  	
  	The setupImpactInitialization file contains a copy of the setup file that set up the initial conditions of
  	the impact.

## Acknowledements
Upkeep of this repository is managed by the Tarleton Particle Modeling Group.
Special thanks to the Tarleton HPC Lab for the use of its hardware.
This ReadMe was written by Gavin McIntosh and Zachary Watson
