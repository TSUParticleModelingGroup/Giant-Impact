#include "../CommonCode/commonSetup.h"

//Globals to be set by the findEarthAndMoon function
int NumberOfEarthElements = -1;
int NumberOfMoonElements = -1;
int *EarthIndex;
int *MoonIndex;

FILE *ImpactPosVelFile;

//Prototyping functions in this file
void allocateMemory();
void openFiles();
void readFrame();
void nBodyCollisionSingleGPU();
void preSetup();
void postSetup();

#include "../CommonCode/commonFunctions.h"
#include "./impactBasicFunctions.h"

void allocateMemory()
{
	Pos = (float4*)malloc(TotalNumberOfElements*sizeof(float4));
	Vel = (float4*)malloc(TotalNumberOfElements*sizeof(float4));
	
	printf("\n ***********************************************************************");
	printf("\n Memory has been allocated.");
	printf("\n ***********************************************************************\n");
}

void openFiles()
{
	ImpactPosVelFile = fopen("./ImpactInformation/impactPosVel", "rb");
	if(ImpactPosVelFile == NULL)
	{
		printf("\n\n The impactPosVel file does not exist\n\n");
		exit(0);
	}
	fseek(ImpactPosVelFile,0,SEEK_SET); // Setting pointer to begining of the file.
	
	printf("\n\n ***********************************************************************");
	printf("\n Files have been opened");
	printf("\n *************************************************************************\n");
}

void readFrame()
{
	int elementsRead;
	int seekReturn;
	float time;
	
	if(ForwardBackward == 1)
	{
		elementsRead = fread(&time, sizeof(float), 1, ImpactPosVelFile);
		
		if(elementsRead != 1)
		{
			printf("\n Error reading frame\n");
			Pause = 1;
		}
		terminalPrint();
		// printf("\n Time = %f", time);
		
		elementsRead = fread(Pos, sizeof(float4), TotalNumberOfElements, ImpactPosVelFile);
		/*if(elementsRead != TotalNumberOfElements)
		{
			printf("\n End of file reached\n");
			Pause = 1;
		}*/
		elementsRead = fread(Vel, sizeof(float4), TotalNumberOfElements, ImpactPosVelFile);
		if(elementsRead != TotalNumberOfElements)
		{
			printf("\n End of file reached\n");
			Pause = 1;
		}
	}
	else
	{
		seekReturn = fseek(ImpactPosVelFile, -2.0*(2.0*TotalNumberOfElements*sizeof(float4) + sizeof(float)) , SEEK_CUR);
		if(seekReturn != 0)
		{
			printf("\n Beginning of file reached\n");
			fseek(ImpactPosVelFile, 0, SEEK_SET);
			Pause = 1;
		}
		else
		{
			elementsRead = fread(&time, sizeof(float), 1, ImpactPosVelFile);
			if(elementsRead != 1)
			{
				printf("\n Error reading frame\n");
				Pause = 1;
			}
			// printf("\n RunTime = %f", RunTime);
			terminalPrint();
			elementsRead = fread(Pos, sizeof(float4), TotalNumberOfElements, ImpactPosVelFile);
			if(elementsRead != TotalNumberOfElements)
			{
				printf("\n Error reading frame\n");
				Pause = 1;
			}
			elementsRead = fread(Vel, sizeof(float4), TotalNumberOfElements, ImpactPosVelFile);
			if(elementsRead != TotalNumberOfElements)
			{
				printf("\n Error reading frame\n");
				Pause = 1;
			}
		}
	}
	RunTime = time;
}

void nBodyCollisionSingleGPU()
{
	if(Pause != 1)
	{
		readFrame();
		drawStuff();
	}
}

void preSetup()
{	
	readParametersFromGenerateBodies();
	allocateMemory();
	openFiles();
	ForwardBackward = 1;
	DrawType = 1;
	DrawQuality = 2;
	Pause = 1;  
	TranslateRotate = 1;
	MovieOn = 0;	
}

void postSetup()
{
	readFrame();	
	drawStuff();
}

int main(int argc, char** argv)
{
	ViewingImpact = 1;
	GeneratingImpact = 0;
	 
	preSetup();
	
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
	glClearColor(0.1, 0.1, 0.1, 0.0);
	
	glShadeModel(GL_SMOOTH);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	
	GLfloat light_position[] = {1.0, 0.0, 0.0, 0.0};
	GLfloat light_position2[] = {1.0, 0.0, 0.0, 0.0};

	//GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_ambient[]  = {0.2, 0.2, 0.2, 1.0};
	GLfloat light_diffuse[]  = {0.8, 0.8, 0.8, 1.0};
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

	glLightfv(GL_LIGHT1, GL_POSITION, light_position2);
	glEnable(GL_LIGHT1);
	
	glutReshapeFunc(reshape);
	glutDisplayFunc(Display);
	glutKeyboardFunc(KeyPressed);
	glutIdleFunc(idle);
	postSetup();
	printf("\n *********** The program is paused. Press p to start *********** \n");
	glutMainLoop();
	return 0;
}






