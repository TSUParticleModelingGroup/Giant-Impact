#include "../CommonCode/commonSetup.h"
//structures to hold constants needed in the kernals
struct forceSeperateKernalConstantsStruct
{
	float GMassFeFe;
	float GMassFeSi;    
	float KFeFe;
	float KSiSi;
	float KFeSi;
	float KRFeFe;
	float KRSiSi;
	float KRFeSi;
	float KRMix;
	float ShellBreakFe;
	float ShellBreakSi;
	float ShellBreakFeSi1;
	float ShellBreakFeSi2; 
	int boarder1; 
	int boarder2;
	int boarder3;  
};

struct moveSeperateKernalConstantsStruct
{
	float Dt;
	float DtOverMassFe;
	float DtOverMassSi;
	int boarder1; 
	int boarder2;
	int boarder3;
};

forceSeperateKernalConstantsStruct ForceSeperateConstant;
moveSeperateKernalConstantsStruct MoveSeperateConstant;

double MassOfBody1 = -1.0;
double MassOfBody2 = -1.0;

float4 InitialSpin1;
float4 InitialSpin2;

double FractionEarthMassOfBody1;	//Mass of body 1 as a proportion of the Earth's mass
double FractionEarthMassOfBody2;	//Mass of body 2 as a proportion of the Earth's mass

double FractionFeBody1;			//Percent by mass of iron in body 1
double FractionSiBody1;			//Percent by mass of silicate in body 1
double FractionFeBody2;			//Percent by mass of iron in body 2
double FractionSiBody2;			//Percent by mass of silicate in body 2

float DampRateBody1;
float DampRateBody2;

double DampTime;
double DampRestTime;
double SpinRestTime;

double DensityFe;			//Density of iron in kilograms meterE-3 (Canup science 2012)
double DensitySi;			//Density of silcate in kilograms meterE-3 (Canup science 2012)

int DampCheck;
int Rest1Check;
int SpinCheck;

//Prototypes for functions in generateTargetImpactor.cu
void createTargetImpactorFolder();
void readTargetImpactorParameters();
void setTargetImpactorParameters();
void loadKernalConstantStructures();
void allocateMemoryAndSetupGPU();
void generateTargetImpactor();
void preSetup();
void postSetup();

#include "../CommonCode/commonFunctions.h"
#include "./generateTargetImpactorBasicFunctions.h"

void createTargetImpactorFolder()
{   
	int returnStatus;
		
	//Creating the name for the folder that will store the Target and Impactor.
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
	string targetImpactorFolderName = "TargetImpactor" + monthday;
	//const char *temp = folderNametemp.c_str();
	const char *folderName = targetImpactorFolderName.c_str();
	
	returnStatus = chdir("./TargetImpactorBin");
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: moving into TargetImpactorBin folder \n");
		exit(0);
	}
	
	//Creating the new folder.
	returnStatus = mkdir(folderName , S_IRWXU|S_IRWXG|S_IRWXO);
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: creating TargetImpactor folder \n");
		exit(0);
	}
	
	returnStatus = chdir(folderName);
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: moving into TargetImpactor folder \n");
		exit(0);
	}

	//Creating the new folder.
	returnStatus = mkdir("TargetImpactorInformation" , S_IRWXU|S_IRWXG|S_IRWXO);
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: creating TargetImpactorInformation folder \n");
		exit(0);
	}
	
	//Creating the new folder.
	returnStatus = mkdir("ImpactBin" , S_IRWXU|S_IRWXG|S_IRWXO);
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: creating ImpactBin folder \n");
		exit(0);
	}
	
	//Copying the collide setup file into the main directory for a template to setup the collition.
	returnStatus = system("cp ../../setupGeneratingTargetImpactor ./TargetImpactorInformation");
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: copying setupGeneratingTargetImpactor file into TargetImpactorInformation folder \n");
		exit(0);
	}
	
	//Copying the collide setup file into the main directory for a template to setup the collition.
	returnStatus = system("cp ../../LinuxScripts/runImpactInitialization .");
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: copying runImpactInitialize file into TargetImpactor folder \n");
		exit(0);
	}
	
	//Copying the collide setup file into the main directory for a template to setup the collition.
	returnStatus = system("cp ../../SetupTemplates/setupImpactInitialization .");
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: copying setupImpactInitialization file into TargetImpactor folder \n");
		exit(0);
	}
	
	//Copying the setupImpactReadMe into this folder.
	returnStatus = system("cp ../../ReadMes/setupImpactReadMe .");
	if(returnStatus != 0) 
	{
		printf("\n TSU Error: copying setupImpactReadMe file into TargetImpactor folder \n");
		exit(0);
	}
}

void readTargetImpactorParameters()
{
	ifstream data;
	string name;
	
	data.open("../../setupGeneratingTargetImpactor");
	if(data.is_open() == 1)
	{
		getline(data,name,'=');
		data >> InitialSpin1.x;
		getline(data,name,'=');
		data >> InitialSpin1.y;
		getline(data,name,'=');
		data >> InitialSpin1.z;
		getline(data,name,'=');
		data >> InitialSpin1.w;
		
		getline(data,name,'=');
		data >> InitialSpin2.x;
		getline(data,name,'=');
		data >> InitialSpin2.y;
		getline(data,name,'=');
		data >> InitialSpin2.z;
		getline(data,name,'=');
		data >> InitialSpin2.w;
		
		getline(data,name,'=');
		data >> FractionEarthMassOfBody1;
		getline(data,name,'=');
		data >> FractionEarthMassOfBody2;
		
		getline(data,name,'=');
		data >> FractionFeBody1;
		getline(data,name,'=');
		data >> FractionSiBody1;
		getline(data,name,'=');
		data >> FractionFeBody2;
		getline(data,name,'=');
		data >> FractionSiBody2;
		
		getline(data,name,'=');
		data >> DampRateBody1;
		getline(data,name,'=');
		data >> DampRateBody2;
		
		getline(data,name,'=');
		data >> TotalNumberOfElements;
		
		getline(data,name,'=');
		data >> DampTime;
		getline(data,name,'=');
		data >> DampRestTime;
		getline(data,name,'=');
		data >> SpinRestTime;
		
		getline(data,name,'=');
		data >> Dt;
		
		getline(data,name,'=');
		data >> DensityFe;
		getline(data,name,'=');
		data >> DensitySi;
		
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
		data >> DrawRate;
	}
	else
	{
		printf("\n TSU Error: could not open setupGeneratingTargetImpactor file\n");
		exit(0);
	}
	data.close();
	
	printf("\n ***********************************************************************");
	printf("\n These are the parameters that were read in just for a spot check.");
	printf("\n Spin on body 1  %f, %f, %f, %f", InitialSpin1.x, InitialSpin1.y, InitialSpin1.z, InitialSpin1.w);
	printf("\n Spin on body 1  %f, %f, %f, %f", InitialSpin1.x, InitialSpin1.y, InitialSpin1.z, InitialSpin1.w);
	printf("\n FractionEarthMassOfBody1 = %f", FractionEarthMassOfBody1);
	printf("\n FractionEarthMassOfBody2 = %f", FractionEarthMassOfBody2);
	printf("\n FractionFeBody1 = %f", FractionFeBody1);
	printf("\n FractionSiBody1 = %f", FractionSiBody1);
	printf("\n FractionFeBody2 = %f", FractionFeBody2);
	printf("\n FractionSiBody2 = %f", FractionSiBody2);
	printf("\n DampRateBody1 = %f", DampRateBody1);
	printf("\n DampRateBody2 = %f", DampRateBody2);
	printf("\n TotalNumberOfElements = %d", TotalNumberOfElements);
	printf("\n DampTime = %f", DampTime);
	printf("\n DampRestTime = %f", DampRestTime);
	printf("\n SpinRestTime = %f", SpinRestTime);
	printf("\n Dt = %f", Dt);
	printf("\n DensityFe = %e", DensityFe);
	printf("\n DensitySi = %e", DensitySi);
	printf("\n KFe = %e", KFe);
	printf("\n KSi = %e", KSi);
	printf("\n KRFe = %e", KRFe);
	printf("\n KRSi = %e", KRSi);
	printf("\n SDFe = %e", SDFe);
	printf("\n SDSi = %e", SDSi);
	printf("\n DrawRate = %d", DrawRate);
	printf("\n ***********************************************************************");
}

//In this function we setup a lot of stuff but what is most important is the setting of units.
//Mass unit, length unit, and time unit. Divide by this unit to take kilograms, kilometers, seconds to our units
//multiply to take our units to kilograms, kilometers, seconds.
void setTargetImpactorParameters()
{
	MassOfBody1 = MassOfEarth*FractionEarthMassOfBody1;
	MassOfBody2 = MassOfEarth*FractionEarthMassOfBody2;
	
	//Checking to see if the silicate and iron percent of each body adds to 1.
	if(FractionFeBody1 + FractionSiBody1 != 1.0) 
	{
		printf("\n TSU Error: body1 fraction don't add to 1\n");
		exit(0);
	}
	if(FractionFeBody2 + FractionSiBody2 != 1.0) 
	{
		printf("\n TSU Error: body2 fraction don't add to 1\n");
		exit(0);
	}
	
	//Setting up the total masses of the bodies.
	double totalMassOfFeBody1 = FractionFeBody1*MassOfBody1;
	double totalMassOfSiBody1 = FractionSiBody1*MassOfBody1;
	double totalMassOfFeBody2 = FractionFeBody2*MassOfBody2;
	double totalMassOfSiBody2 = FractionSiBody2*MassOfBody2;
	double totalMassOfFe = totalMassOfFeBody1 + totalMassOfFeBody2;
	double totalMassOfSi = totalMassOfSiBody1 + totalMassOfSiBody2;
	
	double massFe;
	double massSi;
	double diameterOfElement;
	
	//Finding the number of each type of element in both bodies.
	if(totalMassOfFe != 0.0) NFe = (double)TotalNumberOfElements*(DensitySi/DensityFe)/(totalMassOfSi/totalMassOfFe + DensitySi/DensityFe);
	else NFe = 0;
	NSi = TotalNumberOfElements - NFe;
	
	if(totalMassOfFe != 0.0) NFe1 = NFe*totalMassOfFeBody1/totalMassOfFe; 
	else NFe1 = 0;
	
	NFe2 = NFe - NFe1;
	
	if(totalMassOfSi != 0.0) NSi1 = NSi*totalMassOfSiBody1/totalMassOfSi; 
	else NSi1 = 0;
	
	NSi2 = NSi - NSi1;
	
	//Finding the mass of each type element.
	if(NFe != 0) massFe = totalMassOfFe/NFe;
	else massFe = 0.0;
	if(NSi != 0) massSi = totalMassOfSi/NSi;
	else massSi = 0.0;
	
	//Finding the common diameter of each element. This will also be our unit of length.
	if(NSi != 0) diameterOfElement = pow((6.0*massSi)/(Pi*DensitySi), (1.0/3.0));
	else diameterOfElement = pow((6.0*massFe)/(Pi*DensityFe), (1.0/3.0));
	
	UnitLength = diameterOfElement;
	
	//Finding the mass of each type element. 
	//The mass of a silicate element will be our unit for mass unless it is zero then the mass of 
	//an iron element will be our unit for mass.
	if(NSi != 0) UnitMass = massSi;
	else UnitMass = massFe;
	
	//Using our length and mass units to find a time unit that will set the universal gravity constant to 1.
	if(NSi != 0) UnitTime = sqrt((6.0*massSi*(double)NSi)/(UniversalGravity*Pi*DensitySi*totalMassOfSi));
	else if(NFe != 0) UnitTime = sqrt((6.0*massFe*(double)NFe)/(UniversalGravity*Pi*DensityFe*totalMassOfFe));
	else 
	{
		printf("\n TSU Error: No mass, function setRunParameters\n");
		exit(0);
	}
	
	//If all went correctly the mass of a silicate element is 1 (iron if there is no silicate), the diameter of all elements is 1, 
	//and the universal gravity constant is 1.
	Diameter = 1.0;
	Gravity = 1.0;

	if(NSi != 0)
	{
		MassSi = 1.0;
		MassFe = DensityFe/DensitySi;
	}
	else if(NFe != 0)
	{
		MassFe = 1.0;
	}
	else 
	{
		printf("\n TSU Error: No mass, function setRunParameters\n");
		exit(0);
	}
	
	//Setting mass of bodies in our units
	MassOfBody1 /= UnitMass;
	MassOfBody2 /= UnitMass;

	//Putting Initial Angule Velocities into our units. Taking hours to seconds first then to our units.
	InitialSpin1.w *= UnitTime/3600.0;
	InitialSpin2.w *= UnitTime/3600.0;
	
	//Putting Run times into our units. Taking hours to seconds then to our units.
	DampTime *= 3600.0/UnitTime;
	DampRestTime *= 3600.0/UnitTime;
	SpinRestTime *= 3600.0/UnitTime;
	
	//Putting repultion constants into our units.
	KFe *= UnitTime*UnitTime*UnitLength/UnitMass;
	KSi *= UnitTime*UnitTime*UnitLength/UnitMass;
	
	RadiusOfEarth /= UnitLength;
	MassOfEarth /= UnitMass;
	
	printf("\n ***********************************************************************");
	printf("\n Parameters have been put into our units");
	printf("\n ***********************************************************************");
}

//These are the constants needed in the force and move functions. I put them into structures so they would be easier to pass to the functions.
void loadKernalConstantStructures()
{
	//Force kernal seperate
	ForceSeperateConstant.GMassFeFe = Gravity*MassFe*MassFe;
	ForceSeperateConstant.GMassFeSi = Gravity*MassFe*MassSi;
	
	ForceSeperateConstant.KFeFe = 2.0*KFe;
	ForceSeperateConstant.KSiSi = 2.0*KSi;
	ForceSeperateConstant.KFeSi = KFe + KSi;
	
	ForceSeperateConstant.KRFeFe = 2.0*KFe*KRFe;
	ForceSeperateConstant.KRSiSi = 2.0*KSi*KRSi;
	ForceSeperateConstant.KRFeSi = KFe*KRFe + KSi*KRSi;
	
	if(SDFe >= SDSi) 	ForceSeperateConstant.KRMix = KFe + KSi*KRSi; 
	else 			ForceSeperateConstant.KRMix = KFe*KRFe + KSi;
	
	ForceSeperateConstant.ShellBreakFe = Diameter - Diameter*SDFe;
	ForceSeperateConstant.ShellBreakSi = Diameter - Diameter*SDSi;
	if(SDFe >= SDSi)
	{
		ForceSeperateConstant.ShellBreakFeSi1 = Diameter - Diameter*SDSi;
		ForceSeperateConstant.ShellBreakFeSi2 = Diameter - Diameter*SDFe;
	} 
	else 
	{
		ForceSeperateConstant.ShellBreakFeSi1 = Diameter - Diameter*SDFe;
		ForceSeperateConstant.ShellBreakFeSi2 = Diameter - Diameter*SDSi;
	}
	
	ForceSeperateConstant.boarder1 = NFe1;
	ForceSeperateConstant.boarder2 = NFe1 + NSi1;
	ForceSeperateConstant.boarder3 = NFe1 + NSi1 + NFe2;
	
	//Move kernal seperate	
	MoveSeperateConstant.Dt = Dt;
	MoveSeperateConstant.DtOverMassFe = Dt/MassFe;
	MoveSeperateConstant.DtOverMassSi = Dt/MassSi;
	MoveSeperateConstant.boarder1 = NFe1;
	MoveSeperateConstant.boarder2 = NSi1 + NFe1;
	MoveSeperateConstant.boarder3 = NFe1 + NSi1 + NFe2;
	
	printf("\n ***********************************************************************");
	printf("\n Kernal structures have been loaded.");
	printf("\n ***********************************************************************");
}

void allocateMemoryAndSetupGPU()
{
	Pos = (float4*)malloc(TotalNumberOfElements*sizeof(float4));
	Vel = (float4*)malloc(TotalNumberOfElements*sizeof(float4));
	Force = (float4*)malloc(TotalNumberOfElements*sizeof(float4));
	
	BlockConfig.x = BLOCKSIZE;
	BlockConfig.y = 1;
	BlockConfig.z = 1;
	
	GridConfig.x = (TotalNumberOfElements-1)/BlockConfig.x + 1;
	GridConfig.y = 1;
	GridConfig.z = 1;
	
	cudaMalloc((void**)&Pos_DEV0, TotalNumberOfElements *sizeof(float4));
	cudaErrorCheck("cudaMalloc Pos");
	cudaMalloc((void**)&Vel_DEV0, TotalNumberOfElements *sizeof(float4));
	cudaErrorCheck("cudaMalloc Vel");
	cudaMalloc((void**)&Force_DEV0, TotalNumberOfElements *sizeof(float4));
	cudaErrorCheck("cudaMalloc Force");
	
	//The code is set to do runs with element that are a multiply of the block size.
	if(TotalNumberOfElements%BLOCKSIZE != 0)
	{
		printf("\n TSU Error: Number of Particles is not a multiple of the block size \n\n");
		exit(0);
	}
	
	printf("\n ***********************************************************************");
	printf("\n Memory has been allocated and GPU has been setup.");
	printf("\n ***********************************************************************");
}

__global__ void getForcesSeperate(float4 *pos, float4 *vel, float4 *force, forceSeperateKernalConstantsStruct constant)
{
	int id, ids;
	int i,j;
	int inout;
	float4 forceSum;
	float4 posMe;
	float4 velMe;
	int test;
	int materialSwitch;
	float force_mag;
	float4 dp;
	float4 dv;
	float r2;
	float r;
	float invr;

	__shared__ float4 shPos[BLOCKSIZE];
	__shared__ float4 shVel[BLOCKSIZE];

	id = threadIdx.x + blockDim.x*blockIdx.x;
		
	forceSum.x = 0.0f;
	forceSum.y = 0.0f;
	forceSum.z = 0.0f;
		
	posMe.x = pos[id].x;
	posMe.y = pos[id].y;
	posMe.z = pos[id].z;
	
	velMe.x = vel[id].x;
	velMe.y = vel[id].y;
	velMe.z = vel[id].z;
		
	for(j = 0; j < gridDim.x; j++)
	{
		shPos[threadIdx.x] = pos[threadIdx.x + blockDim.x*j];
		shVel[threadIdx.x] = vel[threadIdx.x + blockDim.x*j];
		__syncthreads();
	   
		for(i = 0; i < blockDim.x; i++)	
		{
			ids = i + blockDim.x*j;
			if((id < constant.boarder2 && ids < constant.boarder2) || (constant.boarder2 <= id && constant.boarder2 <= ids))
			{
    			if((id < constant.boarder2) && (ids < constant.boarder2)) materialSwitch = constant.boarder1;
    			if((constant.boarder2 <= id) && (constant.boarder2 <= ids)) materialSwitch = constant.boarder3;
    			
				dp.x = shPos[i].x - posMe.x;
				dp.y = shPos[i].y - posMe.y;
				dp.z = shPos[i].z - posMe.z;
				r2 = dp.x*dp.x + dp.y*dp.y + dp.z*dp.z;
				r = sqrt(r2);
				if(id == ids) invr = 0;
				else invr = 1.0f/r;

				test = 0;
				if(id < materialSwitch) test = 1;
				if(ids < materialSwitch) test++;
	
				if(test == 0) //silicate silicate force
				{
					if(1.0 <= r)
					{
						force_mag = 1.0/r2;  // G = 1 and mass of silicate elemnet =1
					}
					else if(constant.ShellBreakSi <= r)
					{
						force_mag = 1.0 - constant.KSiSi*(1.0 - r2);
					}
					else
					{
						dv.x = shVel[i].x - velMe.x;
						dv.y = shVel[i].y - velMe.y;
	 					dv.z = shVel[i].z - velMe.z;
						inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
						if(inout <= 0) 	force_mag  = 1.0 - constant.KSiSi*(1.0 - r2);
						else 		force_mag  = 1.0 - constant.KRSiSi*(1.0 - r2);
					}
				}
   	 			else if(test == 1) //Silicate iron force
				{
					if(1.0 <= r)
					{
						force_mag  = constant.GMassFeSi/r2;
					}
					else if(constant.ShellBreakFeSi1 <= r)
					{
						force_mag  = constant.GMassFeSi - constant.KFeSi*(1.0 - r2);
					}
					else if(constant.ShellBreakFeSi2 <= r)
					{
						dv.x = shVel[i].x - velMe.x;
						dv.y = shVel[i].y - velMe.y;
						dv.z = shVel[i].z - velMe.z;
						inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
						if(inout <= 0) 	force_mag = constant.GMassFeSi - constant.KFeSi*(1.0 - r2);
		 				else 		force_mag = constant.GMassFeSi - constant.KRMix*(1.0 - r2);
					}
					else
					{
						dv.x = shVel[i].x - velMe.x;
						dv.y = shVel[i].y - velMe.y;
						dv.z = shVel[i].z - velMe.z;
						inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
						if(inout <= 0) 	force_mag = constant.GMassFeSi - constant.KFeSi*(1.0 - r2);
						else 		force_mag = constant.GMassFeSi - constant.KRFeSi*(1.0 - r2);
		 			}
				}
				else //Iron iron force
				{
					if(1.0 <= r)
					{
						force_mag = constant.GMassFeFe/r2;
					}
					else if(constant.ShellBreakFe <= r)
					{
							force_mag = constant.GMassFeFe - constant.KFeFe*(1.0 - r2);
					}
					else
					{
						dv.x = shVel[i].x - velMe.x;
						dv.y = shVel[i].y - velMe.y;
						dv.z = shVel[i].z - velMe.z;
						inout = dp.x*dv.x + dp.y*dv.y + dp.z*dv.z;
		   				if(inout <= 0) 	force_mag = constant.GMassFeFe - constant.KFeFe*(1.0 - r2);
		  				else 		force_mag = constant.GMassFeFe - constant.KRFeFe*(1.0 - r2);
					}
				}

				forceSum.x += force_mag*dp.x*invr;
				forceSum.y += force_mag*dp.y*invr;
				forceSum.z += force_mag*dp.z*invr;
			}
		}
		force[id].x = forceSum.x;
		force[id].y = forceSum.y;
		force[id].z = forceSum.z;
		__syncthreads();
	}
}

__global__ void moveBodiesDampedSeperate(float4 *pos, float4 *vel, float4 * force, moveSeperateKernalConstantsStruct constant, float DampRateBody1, float DampRateBody2)
{
	float temp;
	float damp;
	int id;
	
    id = threadIdx.x + blockDim.x*blockIdx.x;
 
	if(constant.boarder3 <= id) 
	{
		temp = constant.DtOverMassSi;
		damp = DampRateBody2;
	}
	else if(constant.boarder2 <= id) 
	{
		temp = constant.DtOverMassFe;
		damp = DampRateBody2;
	}
	else if(constant.boarder1 <= id) 
	{
		temp = constant.DtOverMassSi;
		damp = DampRateBody1;
	}
	else 
	{
		temp = constant.DtOverMassFe;
		damp = DampRateBody1;
	}
	
	vel[id].x += (force[id].x-damp*vel[id].x)*temp;
	vel[id].y += (force[id].y-damp*vel[id].y)*temp;
	vel[id].z += (force[id].z-damp*vel[id].z)*temp;

	pos[id].x += vel[id].x*constant.Dt;
	pos[id].y += vel[id].y*constant.Dt;
	pos[id].z += vel[id].z*constant.Dt;
}

void generateTargetImpactor()
{ 
	if(Pause != 1)
	{	
		getForcesSeperate<<<GridConfig, BlockConfig>>>(Pos_DEV0, Vel_DEV0, Force_DEV0, ForceSeperateConstant);
		RunTime += Dt;
		DrawCount++;
		
		if(RunTime < DampTime) 
		{
			if(DampCheck == 0)
			{
				printf("\n ***********************************************************************");
				printf("\n Damping is on.");
				printf("\n ***********************************************************************");
				DampCheck = 1;
				DrawCount = 0;
			}
			moveBodiesDampedSeperate<<<GridConfig, BlockConfig>>>(Pos_DEV0, Vel_DEV0, Force_DEV0, MoveSeperateConstant, DampRateBody1, DampRateBody2);
		}
		else if(RunTime < DampTime + DampRestTime)
		{
			if(Rest1Check == 0)
			{
				printf("\n ***********************************************************************");
				printf("\n Damping rest stage is on.");
				printf("\n ***********************************************************************");
				Rest1Check = 1;
				DrawCount = 0;
			}
			moveBodiesDampedSeperate<<<GridConfig, BlockConfig>>>(Pos_DEV0, Vel_DEV0, Force_DEV0, MoveSeperateConstant, 0.0, 0.0);
		}
		else
		{
			if(SpinCheck == 0)
			{
				copyPosVelFromGPU();
				spinBody(1, InitialSpin1);
				spinBody(2, InitialSpin2);
				copyPosVelToGPU();
				printf("\n ***********************************************************************");
				printf("\n Bodies have been spun and spin rest stage is on");
				printf("\n ***********************************************************************");
				SpinCheck = 1;
			}
			moveBodiesDampedSeperate<<<GridConfig, BlockConfig>>>(Pos_DEV0, Vel_DEV0, Force_DEV0, MoveSeperateConstant, 0.0, 0.0);
		}

		if(DrawCount == DrawRate) 
		{
			copyPosVelFromGPU();
			drawStuff();
			//fflush(stdout); // Flush the output buffer to ensure the cursor is at the beginning of the line
			// Print spaces to clear the line
			// printf("\r\033[K");
			//fflush(stdout);
			//for (int i = 0; i < 40; i++) printf(" ");
			printf("\n Setup time in hours = %f", RunTime*UnitTime/3600.0);
			DrawCount = 0;
		}
		
		if(DampTime + DampRestTime + SpinRestTime < RunTime) 
		{
			zeroOutTargetImpactorDrift(); 
			recordTargetImpactorStats();  
			recordCarryForwardParameters();	
			recordTargetImpactorInitialPosVel();
			cleanUpGenerateTargetImpactor();
			Done = 1;
			Pause = 0;
			if(MovieOn == 1) 
			{
				pclose(MovieFile);
				free(MovieBuffer);
				MovieOn = 0;
			}
			printf("\n ***********************************************************************");
			printf("\n Bodies have been created and stored in a time stamped folder in the TargetImpactorBin folder.");
			printf("\n Have a great day!");
			printf("\n ***********************************************************************\n");
			exit(0);
		}
	}
}

void preSetup()
{	
	createTargetImpactorFolder();
	readTargetImpactorParameters();
	setTargetImpactorParameters();
	loadKernalConstantStructures();
	allocateMemoryAndSetupGPU();
	createRawTargetImpactor();
	copyPosVelToGPU();
	Done = 0;
	RunTime = 0.0;
	DrawCount = 0;
	DampCheck = 0;
	Rest1Check = 0;
	SpinCheck = 0;
	Pause = 0;  
	TranslateRotate = 1;
	MovieOn = 0;	
}

void postSetup()
{	
	drawStuff();	
}

int main(int argc, char** argv)
{ 
	preSetup();

	//Direction here your eye is located location
	EyeX = 0.0;
	EyeY = 0.0;
	EyeZ = 10.0*RadiusOfEarth;

	//Where you are looking
	CenterX = 0.0;
	CenterY = 0.0;
	CenterZ = 0.0;

	//Up vector for viewing
	UpX = 0.0;
	UpY = 1.0;
	UpZ = 0.0;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(5,5);
	Window = glutCreateWindow("Generating Target and Impactor");
	gluLookAt(EyeX, EyeY, EyeZ, CenterX, CenterY, CenterZ, UpX, UpY, UpZ);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	
	glutReshapeFunc(reshape);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(KeyPressed);
	glutIdleFunc(idle);
	
	postSetup();
	glutMainLoop();
	return 0;
}

