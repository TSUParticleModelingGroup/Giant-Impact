#include "../CommonCode/commonSetup.h"

//Globals to be read in from the collition setup file.
float4 InitialPosition1;
float4 InitialPosition2;
float4 InitialVelocity1;
float4 InitialVelocity2;

//Prototyping functions in this file
void createImpactFolder();
void readImpactSetupParameters();
void setupInitialConditions();

void createImpactFolder()
{   
	int returnStatus;
		
	//Creating the name for the folder that will store the collition branch.
	//This will tag it with a date and time stamp.
	time_t t = time(0); 
	struct tm * now = localtime( & t );
	int month = now->tm_mon + 1, day = now->tm_mday, curTimeHour = now->tm_hour, curTimeMin = now->tm_min;
	stringstream smonth, sday, stimeHour, stimeMin;
	smonth << month;
	sday << day;
	stimeHour << curTimeHour;
	stimeMin << curTimeMin;
	string monthday;
	if (curTimeMin <= 9)	monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":0" + stimeMin.str();
	else monthday = smonth.str() + "-" + sday.str() + "-" + stimeHour.str() + ":" + stimeMin.str();
	string foldernametemp = "Impact" + monthday;
	const char *foldername = foldernametemp.c_str();
	
	returnStatus = chdir("./ImpactBin");
	if(returnStatus != 0)
	{
		printf("\n TSU Error: moving into ImpactBin folder");
		exit(0);
	}
	
	//Creating the new folder.
	returnStatus = mkdir(foldername , S_IRWXU|S_IRWXG|S_IRWXO);
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: creating impact folder");
		exit(0);
	}
	
	returnStatus = chdir(foldername);
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: moving into impact folder");
		exit(0);
	}
	
	//Creating the information folder.
	returnStatus = mkdir("ImpactInformation" , S_IRWXU|S_IRWXG|S_IRWXO);
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: creating ImpactInformation folder");
		exit(0);
	}
	
	//Copying setupImpactInitialization file into ImpactInformation folder.
	returnStatus = system("cp ../../setupImpactInitialization ./ImpactInformation");
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: Copying setupImpactInitialization file into ImpactInformation folder");
		exit(0);
	}
	
	//Copying the add time setup file into the main directory for a template to add time to the impact.
	returnStatus = system("cp ../../../../SetupTemplates/setupImpactAddTime .");
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: copying runImpactInitialize file into TargetImpactor folder \n");
		exit(0);
	}
	
	//Copying the script to add time to the impact.
	returnStatus = system("cp ../../../../LinuxScripts/runImpactAddTime .");
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: copying runImpactAddTime file into Impact folder \n");
		exit(0);
	}
	
	//Copying the script to view the impact.
	returnStatus = system("cp ../../../../LinuxScripts/runImpactViewer .");
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: copying runImpactAddTime file into Impact folder \n");
		exit(0);
	}
	
	//Copying the readMe file into the folder.
	returnStatus = system("cp ../../../../ReadMes/addTimeAndViewerReadMe .");
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: copying addTimeAndViewerReadMe file into Impact folder \n");
		exit(0);
	}
}

void readParametersFromGenerateBodies()
{
	ifstream data;
	string name;
	
	data.open("../../TargetImpactorInformation/targetImpactorParameters");
	if(data.is_open() == 1)
	{
		getline(data,name,'=');
		data >> UnitTime;
		getline(data,name,'=');
		data >> UnitLength;
		getline(data,name,'=');
		data >> UnitMass;
		
		getline(data,name,'=');
		data >> MassFe;
		getline(data,name,'=');
		data >> MassSi;
		getline(data,name,'=');
		data >> Diameter;
		
		getline(data,name,'=');
		data >> KFe;
		getline(data,name,'=');
		data >> KSi;
		getline(data,name,'=');
		data >> KRFe;
		getline(data,name,'=');
		data >> KRSi;
		getline(data,name,'=');
		data >> SDFe;
		getline(data,name,'=');
		data >> SDSi;
		
		getline(data,name,'=');
		data >> TotalNumberOfElements;
		getline(data,name,'=');
		data >> NFe1;
		getline(data,name,'=');
		data >> NSi1;
		getline(data,name,'=');
		data >> NFe2;
		getline(data,name,'=');
		data >> NSi2;
	}
	else
	{
		printf("\n TSU Error could not open targetImpactorParameters file\n");
		exit(0);
	}
	data.close();
	
	printf("\n **************************************************************");
	printf("\n These are the parameters (in our units) that were used to generate the target and impactor.");
	printf("\n UnitTime = %f", UnitTime);
	printf("\n UnitLength = %f", UnitLength);
	printf("\n UnitMass = %f", UnitMass);
	printf("\n MassFe = %f", MassFe);
	printf("\n MassSi = %f", MassSi);
	printf("\n Diameter = %f", Diameter);
	printf("\n KFe = %f", KFe);
	printf("\n KSi = %f", KSi);
	printf("\n KRFe = %f", KRFe);
	printf("\n KRSi = %f", KRSi);
	printf("\n SDFe = %f", SDFe);
	printf("\n SDFe = %f", SDFe);
	printf("\n TotalNumberOfElements = %d", TotalNumberOfElements);
	printf("\n NFe1 = %d", NFe1);
	printf("\n NSi1 = %d", NSi1);
	printf("\n NFe2 = %d", NFe2);
	printf("\n NSi2 = %d", NSi2);
	printf("\n **************************************************************");
	
	NFe = NFe1 + NFe2;
	NSi = NSi1 + NSi2;
}

void readImpactSetupParameters()
{
	ifstream data;
	string name;
	
	data.open("../../setupImpactInitialization");
	if(data.is_open() == 1)
	{
		getline(data,name,'=');
		data >> InitialPosition1.x;
		getline(data,name,'=');
		data >> InitialPosition1.y;
		getline(data,name,'=');
		data >> InitialPosition1.z;
		
		getline(data,name,'=');
		data >> InitialPosition2.x;
		getline(data,name,'=');
		data >> InitialPosition2.y;
		getline(data,name,'=');
		data >> InitialPosition2.z;
		
		getline(data,name,'=');
		data >> InitialVelocity1.x;
		getline(data,name,'=');
		data >> InitialVelocity1.y;
		getline(data,name,'=');
		data >> InitialVelocity1.z;
		
		getline(data,name,'=');
		data >> InitialVelocity2.x;
		getline(data,name,'=');
		data >> InitialVelocity2.y;
		getline(data,name,'=');
		data >> InitialVelocity2.z;
	}
	else
	{
		printf("\nTSU Error could not open setupImpactInitialization file\n");
		exit(0);
	}
	data.close();
	
	printf("\n **************************************************************");
	printf("\n These are the parameters that were read in from the setupImpactInitialization file.");
	printf("\n Initial Position on body 1  %f, %f, %f", InitialPosition1.x, InitialPosition1.y, InitialPosition1.z);
	printf("\n Initial Position on body 2  %f, %f, %f", InitialPosition2.x, InitialPosition2.y, InitialPosition2.z);
	printf("\n Initial Velocity on body 1  %f, %f, %f", InitialVelocity1.x, InitialVelocity1.y, InitialVelocity1.z);
	printf("\n Initial Velocity on body 2  %f, %f, %f", InitialVelocity2.x, InitialVelocity2.y, InitialVelocity2.z);
	printf("\n **************************************************************");
	
	// Putting initial conditions into our units.
	InitialPosition1.x /= UnitLength;
	InitialPosition1.y /= UnitLength;
	InitialPosition1.z /= UnitLength;
	
	InitialPosition2.x /= UnitLength;
	InitialPosition2.y /= UnitLength;
	InitialPosition2.z /= UnitLength;
	
	InitialVelocity1.x *= UnitTime/UnitLength;
	InitialVelocity1.y *= UnitTime/UnitLength;
	InitialVelocity1.z *= UnitTime/UnitLength;
	
	InitialVelocity2.x *= UnitTime/UnitLength;
	InitialVelocity2.y *= UnitTime/UnitLength;
	InitialVelocity2.z *= UnitTime/UnitLength;
	
	printf("\n **************************************************************");
	printf("\n These are the parameters after they were put into our units.");
	printf("\n Initial Position on body 1  %f, %f, %f", InitialPosition1.x, InitialPosition1.y, InitialPosition1.z);
	printf("\n Initial Position on body 2  %f, %f, %f", InitialPosition2.x, InitialPosition2.y, InitialPosition2.z);
	printf("\n Initial Velocity on body 1  %f, %f, %f", InitialVelocity1.x, InitialVelocity1.y, InitialVelocity1.z);
	printf("\n Initial Velocity on body 2  %f, %f, %f", InitialVelocity2.x, InitialVelocity2.y, InitialVelocity2.z);
	printf("\n **************************************************************");
}

void setupInitialConditions()
{
	int k;
	size_t returnValue;
	int totalNumberOfElements;
	float4 *tempPos = (float4*)malloc(TotalNumberOfElements*sizeof(float4));
	float4 *tempVel = (float4*)malloc(TotalNumberOfElements*sizeof(float4));
	Pos = (float4*)malloc(TotalNumberOfElements*sizeof(float4));
	Vel = (float4*)malloc(TotalNumberOfElements*sizeof(float4));
	
	// Opening and reading the positions and velocities.
	FILE *inputFile = fopen("../../TargetImpactorInformation/targetImpactorInitialPosVel","rb");
	if(inputFile == NULL)
	{
		printf("\nTSU error: Error opening targetImpactorInitialPosVel file \nn");
		exit(0);
	}
	fseek(inputFile,0,SEEK_SET);
	
	// The total number of elements has already been read in from the papameter file
	// but it is the lead number in this file we read it in and don't use it.
	returnValue = fread(&totalNumberOfElements, sizeof(int), 1, inputFile);
	if(returnValue != 1)
	{
		printf("\nTSU error: Error reading time from targetImpactorInitialPosVel \n");
		exit(0);
	}
	
	returnValue = fread(tempPos, sizeof(float4), TotalNumberOfElements, inputFile);
	if(returnValue != TotalNumberOfElements)
	{
		printf("\nTSU error: Error reading positions from targetImpactorInitialPosVel \n");
		exit(0);
	}
	
	returnValue = fread(tempVel, sizeof(float4), TotalNumberOfElements, inputFile);
	if(returnValue != TotalNumberOfElements)
	{
		printf("\nTSU error: Error reading velocities from targetImpactorInitialPosVel \n");
		exit(0);
	}
	fclose(inputFile);
	
	// Adding on the initial positions and velocities
	for(int i = 0; i < NFe1 + NSi1; i++)
	{
		tempPos[i].x += InitialPosition1.x;
		tempPos[i].y += InitialPosition1.y;
		tempPos[i].z += InitialPosition1.z;
		
		tempVel[i].x += InitialVelocity1.x;
		tempVel[i].y += InitialVelocity1.y;
		tempVel[i].z += InitialVelocity1.z;
	}
	
	for(int i = NFe1 + NSi1; i < TotalNumberOfElements; i++)
	{
		tempPos[i].x += InitialPosition2.x;
		tempPos[i].y += InitialPosition2.y;
		tempPos[i].z += InitialPosition2.z;
		
		tempVel[i].x += InitialVelocity2.x;
		tempVel[i].y += InitialVelocity2.y;
		tempVel[i].z += InitialVelocity2.z;
	}
	
	// The original positions and velocities were set so that the first so many were body1 and the second so many were body2.
	// From now on we will just treat it as one big body so the iron will be collected as the first some many elements
	// and the silicate elements will be the next so many. That is what we are doing here.
	k = 0;
	for(int i = 0; i < NFe1; i++)
	{
		Pos[k] = tempPos[i];
		k++;
	}
	for(int i = NFe1 + NSi1; i < NFe1 + NSi1 + NFe2; i++)
	{
		Pos[k] = tempPos[i];
		k++;
	}
	for(int i = NFe1; i < NFe1 + NSi1; i++)
	{
		Pos[k] = tempPos[i];
		k++;
	}
	for(int i = NFe1 + NSi1 + NFe2; i < TotalNumberOfElements; i++)
	{
		Pos[k] = tempPos[i];
		k++;
	}
	
	k = 0;
	for(int i = 0; i < NFe1; i++)
	{
		Vel[k] = tempVel[i];
		k++;
	}
	for(int i = NFe1 + NSi1; i < NFe1 + NSi1 + NFe2; i++)
	{
		Vel[k] = tempVel[i];
		k++;
	}
	for(int i = NFe1; i < NFe1 + NSi1; i++)
	{
		Vel[k] = tempVel[i];
		k++;
	}
	for(int i = NFe1 + NSi1 + NFe2; i < TotalNumberOfElements; i++)
	{
		Vel[k] = tempVel[i];
		k++;
	}
	
	FILE *outputFile = fopen("impactPosVel","wb");
	float time = 0.0;
	fwrite(&time, sizeof(float), 1, outputFile);
	fwrite(Pos, sizeof(float4), TotalNumberOfElements, outputFile);
	fwrite(Vel, sizeof(float4), TotalNumberOfElements, outputFile);
	fclose(outputFile);
	
	//Copying impactPosVel file into ImpactInformation folder.
	int returnStatus = system("mv ./impactPosVel ./ImpactInformation");
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: Copying impactPosVel file into ImpactInformation folder");
		exit(0);
	}
	
	printf("\n **************************************************************");
	printf("\n Initial conditions have been set.");
	printf("\n **************************************************************");
}

void recordInitialImpactStat()
{
	double totalMass = 0.0;
	double3 centerOfMass, linearVelocity, angularMomentum;
	
	for(int i = 0; i < TotalNumberOfElements; i++)
	{
		if(i < NFe) totalMass += MassFe;
		else totalMass += MassSi;
	}
	
	centerOfMass.x = 0.0;
	centerOfMass.y = 0.0;
	centerOfMass.z = 0.0;
	for(int i = 0; i < TotalNumberOfElements; i++)
	{
		if(i < NFe)
		{
	    	centerOfMass.x += Pos[i].x*MassFe;
			centerOfMass.y += Pos[i].y*MassFe;
			centerOfMass.z += Pos[i].z*MassFe;
		}
		else
		{
	    	centerOfMass.x += Pos[i].x*MassSi;
			centerOfMass.y += Pos[i].y*MassSi;
			centerOfMass.z += Pos[i].z*MassSi;
		}
	}
	centerOfMass.x /= totalMass;
	centerOfMass.y /= totalMass;
	centerOfMass.z /= totalMass;
	
	linearVelocity.x = 0.0;
	linearVelocity.y = 0.0;
	linearVelocity.z = 0.0;
	for(int i = 0; i < TotalNumberOfElements; i++)
	{
		if(i < NFe)
		{
	    	linearVelocity.x += Vel[i].x*MassFe;
			linearVelocity.y += Vel[i].y*MassFe;
			linearVelocity.z += Vel[i].z*MassFe;
		}
		else
		{
	    	linearVelocity.x += Vel[i].x*MassSi;
			linearVelocity.y += Vel[i].y*MassSi;
			linearVelocity.z += Vel[i].z*MassSi;
		}
	}
	linearVelocity.x /= totalMass;
	linearVelocity.y /= totalMass;
	linearVelocity.z /= totalMass;
		
	double3 r;
	double3 v;
	angularMomentum.x = 0.0;
	angularMomentum.y = 0.0;
	angularMomentum.z = 0.0;
	for(int i = 0; i < TotalNumberOfElements; i++)
	{
		r.x = Pos[i].x - centerOfMass.x;
		r.y = Pos[i].y - centerOfMass.y;
		r.z = Pos[i].z - centerOfMass.z;
	
		v.x = Vel[i].x - linearVelocity.x;
		v.y = Vel[i].y - linearVelocity.y;
		v.z = Vel[i].z - linearVelocity.z;
		if(i < NFe)
		{
	    	angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassFe;
			angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassFe;
			angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassFe;
		}
		else
		{
			angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassSi;
			angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassSi;
			angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassSi;
		}
	}
	
	double mag, x, y, z;
	double massConverter = UnitMass; 
	double lengthConverter = UnitLength;
	double velocityConverter = UnitLength/UnitTime; 
	double momentumConverter = UnitMass*UnitLength*UnitLength/UnitTime;
	
	FILE *statsFile = fopen("./ImpactInformation/initialStatsFile","wb");
	
	fprintf(statsFile,"\n\n\n*************************************************************************\n\n");
	fprintf(statsFile,"\nThe following are the initial stats of the impact\n");
	fprintf(statsFile,"\nDistance is measured in Kilometers");
	fprintf(statsFile,"\nMass is measured in Kilograms");
	fprintf(statsFile,"\nTime is measured in seconds");
	fprintf(statsFile,"\nVelocity is measured in Kilometers/second");
	fprintf(statsFile,"\nAngular momentun is measured in Kilograms*Kilometers*Kilometers/seconds\n");
	
	x = centerOfMass.x*lengthConverter;
	y = centerOfMass.y*lengthConverter;
	z = centerOfMass.z*lengthConverter;
	fprintf(statsFile,"\nCenter of mass of the entire system 		        = (%f, %f, %f)\n", x, y, z);
	fprintf(statsFile,"\nTotal mass of the entire system 	= %e\n", totalMass*massConverter);
	
	x = linearVelocity.x*velocityConverter;
	y = linearVelocity.y*velocityConverter;
	z = linearVelocity.z*velocityConverter;
	fprintf(statsFile,"\nLinear velocity of the entire system system 		= (%f, %f, %f)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	fprintf(statsFile,"\nMagnitude of the linear velocity of the entire system 	= %f\n", mag);
	
	x = angularMomentum.x*momentumConverter;
	y = angularMomentum.y*momentumConverter;
	z = angularMomentum.z*momentumConverter;
	fprintf(statsFile,"\nAngular momentum of the entire system system 		= (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	fprintf(statsFile,"\nMagnitude of the angular momentum of the entire system 	= %e\n", mag);
	
	fprintf(statsFile,"\n\n*************************************************************************\n\n\n");
	fclose(statsFile);
}

int main(int argc, char** argv)
{ 
	createImpactFolder();
	readParametersFromGenerateBodies();
	readImpactSetupParameters();
	setupInitialConditions();
	recordInitialImpactStat();	
}






