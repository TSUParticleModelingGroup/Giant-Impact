#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include "stdio.h"
#include <stdlib.h>
#include <cuda.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>
#include <signal.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
using namespace std;

#define BLOCKSIZE 256
#define ASSUMEZEROFLOAT 0.0000001
#define ASSUMEZERODOUBLE 0.00000000001

FILE* MovieFile;

double Pi = 3.141592654;
// Universal gravitational constant in kilometersE3 kilogramsE-1 and secondsE-2 (??? source)
double UniversalGravity = 6.67408e-20;
// The total mass of the Earth in kilograms
double MassOfEarth = 5.97219e24;
// The radius of the Earth in kilometers
double RadiusOfEarth = 6378.0;

//Globals to hold positions, velocities, and forces on both the GPU and CPU
float4 *Pos, *Vel, *Force;
float4 *Pos_DEV0, *Vel_DEV0, *Force_DEV0;

//Globals to setup the kernals
dim3 BlockConfig, GridConfig;

double UnitLength = -1.0;
double Diameter = -1.0;
double UnitMass = -1.0;
double MassSi = -1.0;
double MassFe = -1.0;
double UnitTime = -1.0;
double Gravity = -1.0;

int TotalNumberOfElements;

int NSi1 = -1; 
int NSi2 = -1;
int NFe1 = -1; 
int NFe2 = -1;

int NSi = -1; 
int NFe = -1; 

double KFe;
double KSi;
double KRFe;
double KRSi;
double SDFe;
double SDSi;

double TotalRunTime;
double RunTime;
double Dt;

int DrawRate;
int DrawCount;
int DrawType;
int DrawQuality;

int Pause = 0;
int Done = 0;
int TranslateRotate = 1;
int ForwardBackward = 1;
int MovieOn = 0;

int* MovieBuffer;

//Globals for setting up the viewing window 
static int Window;
int XWindowSize, YWindowSize; 
double Near, Far;

GLdouble Left, Right, Bottom, Top, Front, Back;

//Direction here your eye is located location
double EyeX, EyeY, EyeZ;

//Where you are looking
double CenterX, CenterY, CenterZ;

//Up vector for viewing
double UpX, UpY, UpZ;

void doStuff();
void drawStuff();

float4 ColorBody1Core = {1.0, 1.0, 0.0, 0.0};
float4 ColorBody1Mantle = {0.39, 0.26, 0.13, 0.0};
float4 ColorBody2Core = {1.0, 0.0, 0.0, 0.0};
float4 ColorBody2Mantle = {0.71, 0.40, 0.11, 0.0};






































