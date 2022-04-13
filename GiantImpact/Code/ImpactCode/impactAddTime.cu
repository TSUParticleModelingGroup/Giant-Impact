#include "../CommonCode/commonSetup.h"

//structures to hold constants needed in the kernals
struct forceCollisionKernalConstantsStruct
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
	int NFe;   
};

struct moveCollisionKernalConstantsStruct
{
	float Dt;
	float DtOverMassFe;
	float DtOverMassSi;
	int NFe;
};

//Globals to hold kernal constants
forceCollisionKernalConstantsStruct ForceCollisionConstant;
moveCollisionKernalConstantsStruct MoveCollisionConstant; 

//Globals to be set by the findEarthAndMoon function
int NumberOfEarthElements = -1;
int NumberOfMoonElements = -1;
int *EarthIndex;
int *MoonIndex;

double AdditionalTime;

int RecordRate;
int RecordCount;

FILE *ImpactPosVelFile;

//Prototyping functions in this file
void readSetupImpactAddTime();
void loadKernalConstantStructures();
void allocateMemoryAndSetupGPU();
void nBodyCollisionSingleGPU();
void preSetup();
void postSetup();

#include "../CommonCode/commonFunctions.h"
#include "./impactBasicFunctions.h"

void readSetupImpactAddTime()
{
	ifstream data;
	string name;

	data.open("./setupImpactAddTime");
	if(data.is_open() == 1)
	{
		getline(data,name,'=');
		data >> AdditionalTime;
		getline(data,name,'=');
		data >> Dt;
		getline(data,name,'=');
		data >> DrawRate;
		getline(data,name,'=');
		data >> RecordRate;
	}
	else
	{
		printf("\n TSU Error could not open setupImpactAddTime file\n");
		exit(0);
	}
	data.close();
	
	printf("\n ***********************************************************************");
	printf("\n These are the parameters that were read in from the setupImpactAddTime file.");
	printf("\n AdditionalTime = %lf", AdditionalTime);
	printf("\n Dt = %f", Dt);
	printf("\n DrawRate = %d", DrawRate);
	printf("\n RecordRate = %d", RecordRate);
	printf("\n ***********************************************************************\n");
	
	AdditionalTime *= 3600.0/UnitTime;
}

//These are the constants needed in the force and move functions. I put them into structures so they would be easier to pass to the functions.
void loadKernalConstantStructures()
{
	//Force kernal Earth Moon System
	ForceCollisionConstant.GMassFeFe = MassFe*MassFe; // Gravity is 1.0
	ForceCollisionConstant.GMassFeSi = MassFe;  // Gravity is 1.0 and MassSi is 1.0
	// No need for GMassSiSi because it is 1.0
	
	ForceCollisionConstant.KFeFe = 2.0*KFe;
	ForceCollisionConstant.KSiSi = 2.0*KSi;
	ForceCollisionConstant.KFeSi = KFe + KSi;
	
	ForceCollisionConstant.KRFeFe = 2.0*KFe*KRFe;
	ForceCollisionConstant.KRSiSi = 2.0*KSi*KRSi;
	ForceCollisionConstant.KRFeSi = KFe*KRFe + KSi*KRSi;
	
	if(SDFe >= SDSi) 	ForceCollisionConstant.KRMix = KFe + KSi*KRSi; 
	else 				ForceCollisionConstant.KRMix = KFe*KRFe + KSi;
	
	ForceCollisionConstant.ShellBreakFe = Diameter - Diameter*SDFe;
	ForceCollisionConstant.ShellBreakSi = Diameter - Diameter*SDSi;
	if(SDFe >= SDSi)
	{
		ForceCollisionConstant.ShellBreakFeSi1 = Diameter - Diameter*SDSi;
		ForceCollisionConstant.ShellBreakFeSi2 = Diameter - Diameter*SDFe;
	} 
	else 
	{
		ForceCollisionConstant.ShellBreakFeSi1 = Diameter - Diameter*SDFe;
		ForceCollisionConstant.ShellBreakFeSi2 = Diameter - Diameter*SDSi;
	}
	
	ForceCollisionConstant.NFe = NFe;
	
	//Move kernal Earth Moon System
	MoveCollisionConstant.Dt = Dt;
	MoveCollisionConstant.DtOverMassSi = Dt/MassSi;
	MoveCollisionConstant.DtOverMassFe = Dt/MassFe;
	MoveCollisionConstant.NFe = NFe;
	
	printf("\n ***********************************************************************");
	printf("\n Kernal structures have been loaded.");
	printf("\n ***********************************************************************\n");
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
		printf("\nTSU Error: Number of Particles is not a multiple of the block size \n\n");
		exit(0);
	}
	
	printf("\n ***********************************************************************");
	printf("\n Memory has been allocated and GPU has been setup.");
	printf("\n ***********************************************************************\n");
}

void readTimeAndLastPosVel()
{
	size_t returnValue;
	
	// Reading the positions and velocities.
	ImpactPosVelFile = fopen("./ImpactInformation/impactPosVel","rb+");
	if(ImpactPosVelFile == NULL)
	{
		printf("\nTSU error: Error opening impactPosVel file \nn");
		exit(0);
	}
	fseek(ImpactPosVelFile,-(sizeof(float) + TotalNumberOfElements*sizeof(float4) + TotalNumberOfElements*sizeof(float4)), SEEK_END);
	
	float time;
	returnValue = fread(&time, sizeof(float), 1, ImpactPosVelFile);
	if(returnValue != 1)
	{
		printf("\nTSU error: Error reading addedTime from impactPosVel \n");
		exit(0);
	}
	RunTime = (double)time;
	printf("\n Start time = %f hours \n", RunTime*UnitTime/3600.0);
	TotalRunTime = RunTime + AdditionalTime;
	
	returnValue = fread(Pos, sizeof(float4), TotalNumberOfElements, ImpactPosVelFile);
	if(returnValue != TotalNumberOfElements)
	{
		printf("\nTSU error: Error reading positions from targetImpactorInitialPosVel \n");
		exit(0);
	}
	
	returnValue = fread(Vel, sizeof(float4), TotalNumberOfElements, ImpactPosVelFile);
	if(returnValue != TotalNumberOfElements)
	{
		printf("\nTSU error: Error reading velocities from targetImpactorInitialPosVel \n");
		exit(0);
	}
}

__global__ void getForcesCollisionSingleGPU(float4 *pos, float4 *vel, float4 *force, forceCollisionKernalConstantsStruct constant)
{
	int id, ids;
	int inout;
	float4 forceSum;
	float4 posMe;
	float4 velMe;
	int test;
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
		    
	for(int j=0; j < gridDim.x; j++)
	{
    		shPos[threadIdx.x] = pos[threadIdx.x + blockDim.x*j];
    		shVel[threadIdx.x] = vel[threadIdx.x + blockDim.x*j];
    		__syncthreads();
   
		for(int i=0; i < blockDim.x; i++)	
		{
			ids = i + blockDim.x*j;
		    	dp.x = shPos[i].x - posMe.x;
			dp.y = shPos[i].y - posMe.y;
			dp.z = shPos[i].z - posMe.z;
			r2 = dp.x*dp.x + dp.y*dp.y + dp.z*dp.z;
			r = sqrt(r2);
			if(id == ids) invr = 0;
			else invr = 1.0f/r;

		    	test = 0;
		    	if(id < constant.NFe) test = 1;
		    	if(ids < constant.NFe) test++;
	    
			if(test == 0) //Silicate silicate force
			{
				if(1.0 <= r)
				{
	    				force_mag = 1.0/r2; // G = 1 and mass of silicate elemnet =1
				}
				else if(constant.ShellBreakSi <= r)
				{
					force_mag = 1.0 - constant.KSiSi*(1.0 - r2); // because D = 1 G = 1 and mass of silicate = 1
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
					force_mag  = constant.GMassFeSi -constant.KFeSi*(1.0 - r2);
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
		__syncthreads();
	}
	force[id].x = forceSum.x;
	force[id].y = forceSum.y;
	force[id].z = forceSum.z;
}

__global__ void moveBodiesCollisionSingleGPU(float4 *pos, float4 *vel, float4 * force, moveCollisionKernalConstantsStruct MoveCollisionConstant)
{
	float temp;
	int id;
    	id = threadIdx.x + blockDim.x*blockIdx.x;
    	if(id < MoveCollisionConstant.NFe) temp = MoveCollisionConstant.DtOverMassFe;
    	else temp = MoveCollisionConstant.DtOverMassSi;
	
	vel[id].x += (force[id].x)*temp;
	vel[id].y += (force[id].y)*temp;
	vel[id].z += (force[id].z)*temp;
	
	pos[id].x += vel[id].x*MoveCollisionConstant.Dt;
	pos[id].y += vel[id].y*MoveCollisionConstant.Dt;
	pos[id].z += vel[id].z*MoveCollisionConstant.Dt;
}

void nBodyCollisionSingleGPU()
{ 	
	if(Pause != 1)
	{
		getForcesCollisionSingleGPU<<<GridConfig, BlockConfig>>>(Pos_DEV0, Vel_DEV0, Force_DEV0, ForceCollisionConstant);
		moveBodiesCollisionSingleGPU<<<GridConfig, BlockConfig>>>(Pos_DEV0, Vel_DEV0, Force_DEV0, MoveCollisionConstant);
		DrawCount++;
		RecordCount++;
		RunTime += Dt;
		
		if(DrawCount == DrawRate) 
		{
			cudaMemcpy( Pos, Pos_DEV0, TotalNumberOfElements *sizeof(float4), cudaMemcpyDeviceToHost );
			cudaErrorCheck("cudaMemcpyAsync Pos");
			cudaMemcpy( Vel, Vel_DEV0, TotalNumberOfElements *sizeof(float4), cudaMemcpyDeviceToHost );
			cudaErrorCheck("cudaMemcpyAsync Vel");
			drawPictureCollision();
			DrawCount = 0;
			printf("\n Impact run time = %f hours\n", RunTime*UnitTime/3600.0);
		}
				
		if(RecordCount == RecordRate) 
		{
			cudaMemcpy( Pos, Pos_DEV0, TotalNumberOfElements *sizeof(float4), cudaMemcpyDeviceToHost );
			cudaErrorCheck("cudaMemcpyAsync Pos");
			cudaMemcpy( Vel, Vel_DEV0, TotalNumberOfElements *sizeof(float4), cudaMemcpyDeviceToHost );
			cudaErrorCheck("cudaMemcpyAsync Vel");
			float time = RunTime;
			fwrite(&time, sizeof(float), 1, ImpactPosVelFile);
			fwrite(Pos, sizeof(float4), TotalNumberOfElements, ImpactPosVelFile);
			fwrite(Vel, sizeof(float4), TotalNumberOfElements, ImpactPosVelFile);	
			RecordCount = 0;
		}
		
		if(TotalRunTime < RunTime) 
		{
			cudaMemcpy( Pos, Pos_DEV0, TotalNumberOfElements *sizeof(float4), cudaMemcpyDeviceToHost );
			cudaErrorCheck("cudaMemcpyAsync Pos");
			cudaMemcpy( Vel, Vel_DEV0, TotalNumberOfElements *sizeof(float4), cudaMemcpyDeviceToHost );
			cudaErrorCheck("cudaMemcpyAsync Vel");
			fclose(ImpactPosVelFile);
			recordFinalImpactStat(RunTime);
			Done = 1;
			Pause = 0;
			if(MovieOn == 1) 
			{
				pclose(MovieFile);
				free(MovieBuffer);
				MovieOn = 0;
			}
			printf("\n ***********************************************************************");
			printf("\n Time has been added to the impact.");
			printf("\n If you want to add more time, run the program again.");
			printf("\n ***********************************************************************\n");
			exit(0);
		}
	}
}

void preSetup()
{	
	readParametersFromGenerateBodies();
	readSetupImpactAddTime();
	loadKernalConstantStructures();
	allocateMemoryAndSetupGPU();
	readTimeAndLastPosVel();
	copyPosVelToGPU();
	DrawCount = 0;
	DrawType = 1;
	DrawQuality = 1;
	RecordCount = 0;
	Pause = 0;  
	TranslateRotate = 1;
	MovieOn = 0;	
}

void postSetup()
{	
	drawPictureCollision();	
}

int main(int argc, char** argv)
{ 
	preSetup();
	
	XWindowSize = 1000;
	YWindowSize = 1000; 
	Near = 0.2;
	Far = 600.0;

	double	viewBoxSize = 100.0;
	Left = -viewBoxSize;
	Right = viewBoxSize;
	Bottom = -viewBoxSize;
	Top = viewBoxSize;
	Front = viewBoxSize;
	Back = -viewBoxSize;

	//Direction here your eye is located location
	RadiusOfEarth /= UnitLength;
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
	
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	
	glutReshapeFunc(reshape);
	glutDisplayFunc(Display);
	glutKeyboardFunc(KeyPressed);
	glutIdleFunc(idle);
	postSetup();
	glutMainLoop();
	return 0;
}






