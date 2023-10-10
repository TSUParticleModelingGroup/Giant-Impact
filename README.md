# Giant-Impact
This is the old Moon code reworked


Instructions: 

Right-click and choose 
open in terminal

Go to this new terminal and type
chmod 777 startMeUp

Then type in this new terminal
./startMeUp

Then read the generateTargetImpactorReadMe that is created.

-----The following steps set up the Target and Impactor bodies.-----

1. Open a terminal in this folder (Giant-Impact). 
   A shortcut is to right-click and choose "Open in Terminal".

2. To compile all the code and place it in the correct locations, in the terminal, type:

        ./compileAll 
   
   (Note this only needs to be done once unless you modify the source code.)

3. Now, you should open the setupGeneratingTargetImpactor file and set the parameters for your run. 
   These parameters will set the parameters for the two bodies that will be created.
   Once you are happy with your setup parameters, save the file.

4. Next, you will generate the Target and Impactor with the parameters you set in the 
   setupGeneratingTargetImpactor file. First, the bodies will be created and allowed to settle into place.
   Then, the bodies will be spun, and again, allowed to settle. At this point, the bodies are completely
   independent and have no influence on one another. Once this process has been completed, the Target and 
   The impactor will be put in a time-stamped folder in the TargetImpactorBin. 
   (Note the time stamp only goes down to the minute so if you make more than one
   in a minute the program will throw an error.) To generate the bodies, in the terminal type:
	
	./runGTI
	
   You should note the date/time so you can identify the bodies you just created.

5. Open the TargetImpactorBin folder, and you will find the generated folder containing the two bodies.
   If you would like, you can rename this folder to a more descriptive name. Go into this folder. In this
   folder you will find two folders and three files, containing all the information you will need to create
   impacts using the two bodies you created.
   
6. Now for instructions on how to generate an impact using the two bodies you just created, follow the
   instructions provided in the setupImpactReadMe file.


-----The following steps set up the initial conditions for an impact.-----

1. Open a terminal in this folder. 
   A shortcut is to right-click and choose "Open in Terminal".
   
2. Open the setupImpactInitialization file and set the parameters for the impact.
   Note, that this just sets the initial positions and velocities of the two bodies to be impacted.
   Don't forget to save the file.

3. Next you will generate an initial impact (ie, time = 0) with the specified positions and velocities.
   To do this, type this in the terminal:
   
   ./runImpactInitialization

   This will then create a time-stamped folder in the ImpactBin folder which contains 1 folder and 4 files. 
   (Note the time stamp only goes down to the minute so	if you make more than one in a minute the program will 
   throw an error.).
   If you would like, you can change the name of this folder to something more descriptive.	
	
4. Go into your new folder and follow the instructions outlined in the addTimeAndViewerReadMe file. 


-----The following steps are for adding time to an impact or viewing an impact.-----

1. Open a terminal in this folder. 
   A shortcut is to right click and choose "Open in Terminal".

2. Open the setupImpactAddTime file and set the time to add to the impact. You can also modify
   parameters to change the time step, draw rate, and record rate. Don't forget to save the file.

3. To actually add time to the impact, in the terminal, type 

	./runImpactAddTime
	
   Note, this will add additional time to the impact starting from where it left off.

4. At any time, to view the entire impact, in the terminal, type

	./runImpactViewer
	
   Note, the viewer does no n-body calculations. It simply reads the impactPosVel file and displays
   frames to the screen.


-----The following is just information about the two folders in the current directory-----

- The ImpactBin folder contains all impacts created for this setup of bodies.

- The TargetImpactorInformation folder contains 4 files:

  	The setupGeneratingTargetImpactor file is just a copy of the setup file in the Giant-Impact folder 
  	that created this set of bodies.
  
  	The targetImpactorInitialPosVel file (a binary file) contains the positions and velocities of the 
  	two bodies.
  
  	The targetImpactorParameters file contains unit conversion and other pertinent information about 
  	the two bodies.

  	The targetImpactorStats file contains general stats about the two bodies created.

-----The following is just information about the impactInformation folder and runStatsFile file-----

- The runStatsFile file contains stats about the impact up to the current time. It is updated every time
  runImpactAddTime is called.

- The impactInformation folder contains 3 files:

  	The impactPosVel file (a binary file) contains the positions and velocities of all elements in the impact.
  	
  	The initialStatsFile file contains the initial (ie. time = 0) stats of the impact. 
  	
  	The setupImpactInitialization file contains a copy of the setup file that set up the initial conditions of
  	the impact.
