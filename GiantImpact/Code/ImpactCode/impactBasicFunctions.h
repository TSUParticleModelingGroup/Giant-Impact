// Prototypes of all the functions in this file
int findEarthAndMoon();
float getMassCollision(int);
float3 getCenterOfMassCollision(int);
float3 getLinearVelocityCollision(int);
float3 getAngularMomentumCollision(int);
void drawPictureCollision();
void drawPictureNormal();
void drawAnalysisPicture();
void setupInitialConditions();
void recordInitialImpactStat();
void recordFinalCollisionStat(double);
void recordPosAndVel();

//void nBodyCollisionSingleGPU();

// These first two functions are so that the callback functions have a common function to call
// (drawpicture and runSimulation) that is redirected to the correct funtion for this code.
// They are prototyped in the commonFunctions.h file.
void doStuff()
{
	nBodyCollisionSingleGPU();
}

void drawStuff()
{
	drawPictureCollision();
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
	
	printf("\n ***********************************************************************");
	printf("\n These are the parameters that were read in from the collideParameters file.");
	printf("\n UnitTime = %e seconds", UnitTime);
	printf("\n UnitLength = %e kilomaters", UnitLength);
	printf("\n UnitMass = %e kilograms", UnitMass);
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
	printf("\n ***********************************************************************\n");
	
	NFe = NFe1 + NFe2;
	NSi = NSi1 + NSi2;
}

int findEarthAndMoon()
{
	int groupId[TotalNumberOfElements], used[TotalNumberOfElements];
	float mag, dx, dy, dz;
	float touch = Diameter*1.5;
	int groupNumber, numberOfGroups;
	int k;
	
	for(int i = 0; i < TotalNumberOfElements; i++)
	{
		groupId[i] = -1;
		used[i] = 0;
	}
	
	groupNumber = 0;
	for(int i = 0; i < TotalNumberOfElements; i++)
	{
		if(groupId[i] == -1)
		{
			groupId[i] = groupNumber;
			//find all from this group
			k = i;
			while(k < TotalNumberOfElements)
			{
				if(groupId[k] == groupNumber && used[k] == 0)
				{
					for(int j = i; j < TotalNumberOfElements; j++)
					{
						dx = Pos[k].x - Pos[j].x;
						dy = Pos[k].y - Pos[j].y;
						dz = Pos[k].z - Pos[j].z;
						mag = sqrt(dx*dx + dy*dy + dz*dz);
						if(mag < touch)
						{
							groupId[j] = groupNumber;
						}
					}
					used[k] = 1;
					k = i;
				}
				else k++;	
			}
			
		}
		groupNumber++;
	}
	numberOfGroups = groupNumber;
	
	if(numberOfGroups == 1)
	{
		printf("\n No Moon found\n");
	}
	
	int count;
	int *groupSize = (int *)malloc(numberOfGroups*sizeof(int));
	for(int i = 0; i < numberOfGroups; i++)
	{
		count = 0;
		for(int j = 0; j < TotalNumberOfElements; j++)
		{
			if(i == groupId[j]) count++;
		}
		groupSize[i] = count;
	}
	
	int earthGroupId = -1;
	NumberOfEarthElements = 0;
	for(int i = 0; i < numberOfGroups; i++)
	{
		if(groupSize[i] > NumberOfEarthElements)
		{
			NumberOfEarthElements = groupSize[i];
			earthGroupId = i;
		}
	}
	
	int moonGroupId = -1;
	NumberOfMoonElements = 0;
	for(int i = 0; i < numberOfGroups; i++)
	{
		if(groupSize[i] > NumberOfMoonElements && i != earthGroupId)
		{
			NumberOfMoonElements = groupSize[i];
			moonGroupId = i;
		}
	}
	
	free(groupSize);
	EarthIndex = (int *)malloc(NumberOfEarthElements*sizeof(int));
	MoonIndex = (int *)malloc(NumberOfMoonElements*sizeof(int));
	
	int earthCount = 0;
	int moonCount = 0;
	for(int j = 0; j < TotalNumberOfElements; j++)
	{
		if(groupId[j] == earthGroupId) 
		{
			EarthIndex[earthCount] = j;
			earthCount++;
		}
		else if(groupId[j] == moonGroupId)  
		{
			MoonIndex[moonCount] = j;
			moonCount++;
		}
	}
	
	return(1);	
}

float getMassCollision(int scope)
{
	float mass = 0.0;
	
	if(scope == 0) // entire system
	{
		for(int i = 0; i < TotalNumberOfElements; i++)
		{
			if(i < NFe) mass += MassFe;
			else mass += MassSi;
		}
	}
	else if(scope == 1) // earth-moon syatem
	{
		for(int i = 0; i < NumberOfEarthElements; i++)
		{
			if(EarthIndex[i] < NFe) mass += MassFe;
			else mass += MassSi;
		}
		for(int i = 0; i < NumberOfMoonElements; i++)
		{
			if(MoonIndex[i] < NFe) mass += MassFe;
			else mass += MassSi;
		}
	}
	else if(scope == 2) // earth
	{
		for(int i = 0; i < NumberOfEarthElements; i++)
		{
			if(EarthIndex[i] < NFe) mass += MassFe;
			else mass += MassSi;
		}
	}
	else if(scope == 3) // moon
	{
		for(int i = 0; i < NumberOfMoonElements; i++)
		{
			if(MoonIndex[i] < NFe) mass += MassFe;
			else mass += MassSi;
		}
	}
	else
	{
		printf("\nTSU Error: In getMassCollision function bodyId invalid\n");
		exit(0);
	}
	return(mass);
}

float3 getCenterOfMassCollision(int scope)
{
	float totalMass;
	float3 centerOfMass;
	centerOfMass.x = 0.0;
	centerOfMass.y = 0.0;
	centerOfMass.z = 0.0;
	
	if(scope == 0) // Entire System
	{
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
		totalMass = getMassCollision(0);
		centerOfMass.x /= totalMass;
		centerOfMass.y /= totalMass;
		centerOfMass.z /= totalMass;
	}
	else if(scope == 1) // Earth-Moon System
	{
		for(int i = 0; i < NumberOfEarthElements; i++)
		{
			if(EarthIndex[i] < NFe)
			{
		    		centerOfMass.x += Pos[EarthIndex[i]].x*MassFe;
				centerOfMass.y += Pos[EarthIndex[i]].y*MassFe;
				centerOfMass.z += Pos[EarthIndex[i]].z*MassFe;
			}
			else
			{
		    		centerOfMass.x += Pos[EarthIndex[i]].x*MassSi;
				centerOfMass.y += Pos[EarthIndex[i]].y*MassSi;
				centerOfMass.z += Pos[EarthIndex[i]].z*MassSi;
			}
		}
		for(int i = 0; i < NumberOfMoonElements; i++)
		{
			if(MoonIndex[i] < NFe)
			{
		    		centerOfMass.x += Pos[MoonIndex[i]].x*MassFe;
				centerOfMass.y += Pos[MoonIndex[i]].y*MassFe;
				centerOfMass.z += Pos[MoonIndex[i]].z*MassFe;
			}
			else
			{
		    		centerOfMass.x += Pos[MoonIndex[i]].x*MassSi;
				centerOfMass.y += Pos[MoonIndex[i]].y*MassSi;
				centerOfMass.z += Pos[MoonIndex[i]].z*MassSi;
			}
		}
		totalMass = getMassCollision(1);
		centerOfMass.x /= totalMass;
		centerOfMass.y /= totalMass;
		centerOfMass.z /= totalMass;
		
	}
	else if(scope == 2) // Earth
	{
		for(int i = 0; i < NumberOfEarthElements; i++)
		{
			if(EarthIndex[i] < NFe)
			{
		    		centerOfMass.x += Pos[EarthIndex[i]].x*MassFe;
				centerOfMass.y += Pos[EarthIndex[i]].y*MassFe;
				centerOfMass.z += Pos[EarthIndex[i]].z*MassFe;
			}
			else
			{
		    		centerOfMass.x += Pos[EarthIndex[i]].x*MassSi;
				centerOfMass.y += Pos[EarthIndex[i]].y*MassSi;
				centerOfMass.z += Pos[EarthIndex[i]].z*MassSi;
			}
		}
		totalMass = getMassCollision(2);
		centerOfMass.x /= totalMass;
		centerOfMass.y /= totalMass;
		centerOfMass.z /= totalMass;
	}
	else if(scope == 3) // Moon
	{
		for(int i = 0; i < NumberOfMoonElements; i++)
		{
			if(MoonIndex[i] < NFe)
			{
		    		centerOfMass.x += Pos[MoonIndex[i]].x*MassFe;
				centerOfMass.y += Pos[MoonIndex[i]].y*MassFe;
				centerOfMass.z += Pos[MoonIndex[i]].z*MassFe;
			}
			else
			{
		    		centerOfMass.x += Pos[MoonIndex[i]].x*MassSi;
				centerOfMass.y += Pos[MoonIndex[i]].y*MassSi;
				centerOfMass.z += Pos[MoonIndex[i]].z*MassSi;
			}
		}
		totalMass = getMassCollision(3);
		centerOfMass.x /= totalMass;
		centerOfMass.y /= totalMass;
		centerOfMass.z /= totalMass;
	}
	else
	{
		printf("\nTSU Error: In getCenterOfMassCollision function scope invalid\n");
		exit(0);
	}
	return(centerOfMass);
}

float3 getLinearVelocityCollision(int scope)
{
	float totalMass;
	float3 linearVelocity;
	linearVelocity.x = 0.0;
	linearVelocity.y = 0.0;
	linearVelocity.z = 0.0;
	
	if(scope == 0) // entire system
	{
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
		totalMass = getMassCollision(0);
		linearVelocity.x /= totalMass;
		linearVelocity.y /= totalMass;
		linearVelocity.z /= totalMass;
	}	
	else if(scope == 1) // earth-moon system
	{
		for(int i = 0; i < NumberOfEarthElements; i++)
		{
			if(EarthIndex[i] < NFe)
			{
		    	linearVelocity.x += Vel[EarthIndex[i]].x*MassFe;
				linearVelocity.y += Vel[EarthIndex[i]].y*MassFe;
				linearVelocity.z += Vel[EarthIndex[i]].z*MassFe;
			}
			else
			{
		    	linearVelocity.x += Vel[EarthIndex[i]].x*MassSi;
				linearVelocity.y += Vel[EarthIndex[i]].y*MassSi;
				linearVelocity.z += Vel[EarthIndex[i]].z*MassSi;
			}
		}
		for(int i = 0; i < NumberOfMoonElements; i++)
		{
			if(MoonIndex[i] < NFe)
			{
		    	linearVelocity.x += Vel[MoonIndex[i]].x*MassFe;
				linearVelocity.y += Vel[MoonIndex[i]].y*MassFe;
				linearVelocity.z += Vel[MoonIndex[i]].z*MassFe;
			}
			else
			{
		    	linearVelocity.x += Vel[MoonIndex[i]].x*MassSi;
				linearVelocity.y += Vel[MoonIndex[i]].y*MassSi;
				linearVelocity.z += Vel[MoonIndex[i]].z*MassSi;
			}
		}
		totalMass = getMassCollision(1);
		linearVelocity.x /= totalMass;
		linearVelocity.y /= totalMass;
		linearVelocity.z /= totalMass;
	}
	else if(scope == 2) //earth
	{
		for(int i = 0; i < NumberOfEarthElements; i++)
		{
			if(EarthIndex[i] < NFe)
			{
		    	linearVelocity.x += Vel[EarthIndex[i]].x*MassFe;
				linearVelocity.y += Vel[EarthIndex[i]].y*MassFe;
				linearVelocity.z += Vel[EarthIndex[i]].z*MassFe;
			}
			else
			{
		    	linearVelocity.x += Vel[EarthIndex[i]].x*MassSi;
				linearVelocity.y += Vel[EarthIndex[i]].y*MassSi;
				linearVelocity.z += Vel[EarthIndex[i]].z*MassSi;
			}
		}
		totalMass = getMassCollision(2);
		linearVelocity.x /= totalMass;
		linearVelocity.y /= totalMass;
		linearVelocity.z /= totalMass;
	}
	else if(scope == 3) //moon
	{
		for(int i = 0; i < NumberOfMoonElements; i++)
		{
			if(MoonIndex[i] < NFe)
			{
		    	linearVelocity.x += Vel[MoonIndex[i]].x*MassFe;
				linearVelocity.y += Vel[MoonIndex[i]].y*MassFe;
				linearVelocity.z += Vel[MoonIndex[i]].z*MassFe;
			}
			else
			{
		    	linearVelocity.x += Vel[MoonIndex[i]].x*MassSi;
				linearVelocity.y += Vel[MoonIndex[i]].y*MassSi;
				linearVelocity.z += Vel[MoonIndex[i]].z*MassSi;
			}
		}
		totalMass = getMassCollision(3);
		linearVelocity.x /= totalMass;
		linearVelocity.y /= totalMass;
		linearVelocity.z /= totalMass;
	}
	else
	{
		printf("\nTSU Error: in getlinearVelocityEarthMoonSystem function scope invalid\n");
		exit(0);
	}
	return(linearVelocity);
}

float3 getAngularMomentumCollision(int scope)
{
	float3 centerOfMass, linearVelocity, angularMomentum;
	float3 r;
	float3 v;
	angularMomentum.x = 0.0;
	angularMomentum.y = 0.0;
	angularMomentum.z = 0.0;
	
	if(scope == 0) //Entire system
	{
		centerOfMass = getCenterOfMassCollision(0);
		linearVelocity = getLinearVelocityCollision(0);
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
	}
	else if(scope == 1) //Earth-Moon system
	{
		centerOfMass = getCenterOfMassCollision(1);
		linearVelocity = getLinearVelocityCollision(1);
		for(int i = 0; i < NumberOfEarthElements; i++)
		{
			r.x = Pos[EarthIndex[i]].x - centerOfMass.x;
			r.y = Pos[EarthIndex[i]].y - centerOfMass.y;
			r.z = Pos[EarthIndex[i]].z - centerOfMass.z;
		
			v.x = Vel[EarthIndex[i]].x - linearVelocity.x;
			v.y = Vel[EarthIndex[i]].y - linearVelocity.y;
			v.z = Vel[EarthIndex[i]].z - linearVelocity.z;
			if(EarthIndex[i] < NFe)
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
		for(int i = 0; i < NumberOfMoonElements; i++)
		{
			r.x = Pos[MoonIndex[i]].x - centerOfMass.x;
			r.y = Pos[MoonIndex[i]].y - centerOfMass.y;
			r.z = Pos[MoonIndex[i]].z - centerOfMass.z;
		
			v.x = Vel[MoonIndex[i]].x - linearVelocity.x;
			v.y = Vel[MoonIndex[i]].y - linearVelocity.y;
			v.z = Vel[MoonIndex[i]].z - linearVelocity.z;
			if(MoonIndex[i] < NFe)
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
	}
	else if(scope == 2) //Earth
	{
		centerOfMass = getCenterOfMassCollision(2);
		linearVelocity = getLinearVelocityCollision(2);
		for(int i = 0; i < NumberOfEarthElements; i++)
		{
			r.x = Pos[EarthIndex[i]].x - centerOfMass.x;
			r.y = Pos[EarthIndex[i]].y - centerOfMass.y;
			r.z = Pos[EarthIndex[i]].z - centerOfMass.z;
		
			v.x = Vel[EarthIndex[i]].x - linearVelocity.x;
			v.y = Vel[EarthIndex[i]].y - linearVelocity.y;
			v.z = Vel[EarthIndex[i]].z - linearVelocity.z;
			if(EarthIndex[i] < NFe)
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
	}
	else if(scope == 3) //Moon
	{
		centerOfMass = getCenterOfMassCollision(3);
		linearVelocity = getLinearVelocityCollision(3);
		for(int i = 0; i < NumberOfMoonElements; i++)
		{
			r.x = Pos[MoonIndex[i]].x - centerOfMass.x;
			r.y = Pos[MoonIndex[i]].y - centerOfMass.y;
			r.z = Pos[MoonIndex[i]].z - centerOfMass.z;
		
			v.x = Vel[MoonIndex[i]].x - linearVelocity.x;
			v.y = Vel[MoonIndex[i]].y - linearVelocity.y;
			v.z = Vel[MoonIndex[i]].z - linearVelocity.z;
			if(MoonIndex[i] < NFe)
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
	}
	else
	{
		printf("\nTSU Error: in getAngularMomentumCollision function scope invalid\n");
		exit(0);
	}
	return(angularMomentum);
}

void drawPictureCollision()
{
	if(DrawType == 1) drawPictureNormal();
	if(DrawType == 2) 
	{
		drawAnalysisPicture();
	}
}

void drawPictureNormal()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	if(DrawQuality == 1)
	{
		glPointSize(2.0);
		glBegin(GL_POINTS);
		 	for(int i=0; i<TotalNumberOfElements; i++)
			{
				if(i < NFe1) 
				{
					glColor3d(ColorBody1Core.x,ColorBody1Core.y,ColorBody1Core.z);
				}
				else if(i < NFe1 + NFe2)
				{
					glColor3d(ColorBody2Core.x,ColorBody2Core.y,ColorBody2Core.z);
				}
				else if(i < NFe1 + NFe2 + NSi1) 
				{
					glColor3d(ColorBody1Mantle.x,ColorBody1Mantle.y,ColorBody1Mantle.z);
				}
				else
				{
					glColor3d(ColorBody2Mantle.x,ColorBody2Mantle.y,ColorBody2Mantle.z);
				}
				
				glVertex3f(Pos[i].x, Pos[i].y, Pos[i].z);
			}
		glEnd();
	}
	
	if(DrawQuality == 2)
	{
		for(int i = 0; i < TotalNumberOfElements; i++)
		{
			if(i < NFe1) 
			{
				glColor3d(ColorBody1Core.x,ColorBody1Core.y,ColorBody1Core.z);
			}
			else if(i < NFe1 + NFe2)
			{
				glColor3d(ColorBody2Core.x,ColorBody2Core.y,ColorBody2Core.z);
			}
			else if(i < NFe1 + NFe2 + NSi1) 
			{
				glColor3d(ColorBody1Mantle.x,ColorBody1Mantle.y,ColorBody1Mantle.z);
			}
			else
			{
				glColor3d(ColorBody2Mantle.x,ColorBody2Mantle.y,ColorBody2Mantle.z);
			}
				
			glPushMatrix();
				glTranslatef(Pos[i].x, Pos[i].y, Pos[i].z);
				glutSolidSphere(0.5,20,20);
			glPopMatrix();
		}
	}

	glutSwapBuffers();
	
	// Making a video of the run.
	if(MovieOn == 1)
	{
		glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, MovieBuffer);
		fwrite(MovieBuffer, sizeof(int)*XWindowSize*YWindowSize, 1, MovieFile);
	}
}

void drawAnalysisPicture()
{
	findEarthAndMoon();
	float massSystem = getMassCollision(1);
	float massEarth = getMassCollision(2);
	float massMoon = getMassCollision(3);
	float3 centerOfMassSystem = getCenterOfMassCollision(1);
	float3 centerOfMassEarth = getCenterOfMassCollision(2);
	float3 centerOfMassMoon = getCenterOfMassCollision(3);
	float3 linearVelocitySystem = getLinearVelocityCollision(1);
	float3 linearVelocityEarth = getLinearVelocityCollision(2);
	float3 linearVelocityMoon = getLinearVelocityCollision(3);
	float3 angularMomentumSystem = getAngularMomentumCollision(1);
	float3 angularMomentumEarth = getAngularMomentumCollision(2);
	float3 angularMomentumMoon = getAngularMomentumCollision(3);
	float Stretch;
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	if(DrawQuality == 1)
	{
		glPointSize(2.0);
		//Recoloring the Earth elements blue
		glColor3d(0.0,0.0,1.0);
		glBegin(GL_POINTS);
			for(int i = 0; i < NumberOfEarthElements; i++)
			{	
					glVertex3f(Pos[EarthIndex[i]].x, Pos[EarthIndex[i]].y, Pos[EarthIndex[i]].z);
			}
		glEnd();
		
		//Recoloring the Moon elements white
		glColor3d(1.0,1.0,1.0);
		glBegin(GL_POINTS);
			for(int i = 0; i < NumberOfMoonElements; i++)
			{	
				glVertex3f(Pos[MoonIndex[i]].x, Pos[MoonIndex[i]].y, Pos[MoonIndex[i]].z);
			}
		glEnd();
	}
	if(DrawQuality == 2)
	{
		//Recoloring the Earth elements blue
		glColor3d(0.0,0.0,1.0);
		glBegin(GL_POINTS);
			for(int i = 0; i < NumberOfEarthElements; i++)
			{	
				glPushMatrix();
					glTranslatef(Pos[EarthIndex[i]].x, Pos[EarthIndex[i]].y, Pos[EarthIndex[i]].z);
					glutSolidSphere(0.5,20,20);
				glPopMatrix();
			}
		glEnd();
		
		//Recoloring the Moon elements white
		glColor3d(1.0,1.0,1.0);
		glBegin(GL_POINTS);
			for(int i = 0; i < NumberOfMoonElements; i++)
			{	
				glPushMatrix();
					glTranslatef(Pos[MoonIndex[i]].x, Pos[MoonIndex[i]].y, Pos[MoonIndex[i]].z);
					glutSolidSphere(0.5,20,20);
				glPopMatrix();
			}
		glEnd();
	}
	
	glLineWidth(3.0);
	glPointSize(10.0);

	//Place a yellow point at the center of mass of the Earth-Moon system
	glColor3d(1.0,1.0,0.0);
	glBegin(GL_POINTS);
		glVertex3f(centerOfMassSystem.x, centerOfMassSystem.y, centerOfMassSystem.z);
	glEnd();
	
	//Placing yellow vectors in the direction of the angular momentum of the Earth-Moon system
	glColor3f(1.0,1.0,0.0);
	Stretch = 1.0;
	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMassSystem.x, centerOfMassSystem.y, centerOfMassSystem.z);
		glVertex3f(	centerOfMassSystem.x + angularMomentumSystem.x*Stretch/massSystem, 
				centerOfMassSystem.y + angularMomentumSystem.y*Stretch/massSystem, 
				centerOfMassSystem.z + angularMomentumSystem.z*Stretch/massSystem);
	glEnd();
	
	//Place a blue point at the center of mass of the Earth
	glColor3d(0.0,0.0,1.0);
	glBegin(GL_POINTS);
		glVertex3f(centerOfMassEarth.x, centerOfMassEarth.y, centerOfMassEarth.z);
	glEnd();
	
	//Placing blue vectors in the direction of the angular momentum of the Earth
	Stretch = 1.0;
	glBegin(GL_LINE_LOOP);
	glColor3f(0.0,0.0,1.0);
		glVertex3f(centerOfMassEarth.x, centerOfMassEarth.y, centerOfMassEarth.z);
		glVertex3f(	centerOfMassEarth.x + angularMomentumEarth.x*Stretch/massEarth, 
				centerOfMassEarth.y + angularMomentumEarth.y*Stretch/massEarth, 
				centerOfMassEarth.z + angularMomentumEarth.z*Stretch/massEarth);
	glEnd();
	
	//Place a white point at the center of mass of the Moon
	glColor3d(1.0,1.0,1.0);
	glBegin(GL_POINTS);
		glVertex3f(centerOfMassMoon.x, centerOfMassMoon.y, centerOfMassMoon.z);
	glEnd();
	
	//Placing white vectors in the direction of the angular momentum of the Moon
	Stretch = 1.0;
	glColor3f(1.0,1.0,1.0);
	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMassMoon.x, centerOfMassMoon.y, centerOfMassMoon.z);
		glVertex3f(	centerOfMassMoon.x + angularMomentumMoon.x*Stretch/massMoon, 
				centerOfMassMoon.y + angularMomentumMoon.y*Stretch/massMoon, 
				centerOfMassMoon.z + angularMomentumMoon.z*Stretch/massMoon);
	glEnd();

	//Placing green vectors in the direction of linear velocity of the Moon
	Stretch = 1.0;
	glColor3f(0.0,1.0,0.0);
	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMassMoon.x, centerOfMassMoon.y, centerOfMassMoon.z);
		glVertex3f(	centerOfMassMoon.x + linearVelocityMoon.x*Stretch, 
				centerOfMassMoon.y + linearVelocityMoon.y*Stretch, 
				centerOfMassMoon.z + linearVelocityMoon.z*Stretch);
	glEnd();

	glutSwapBuffers();
	
	// Making a video of the run.
	if(MovieOn == 1)
	{
		glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, MovieBuffer);
		fwrite(MovieBuffer, sizeof(int)*XWindowSize*YWindowSize, 1, MovieFile);
	}
	
	free(EarthIndex);
	free(MoonIndex);
}

void recordFinalImpactStat(double time)
{
	double mag, size, angle, x, y, z;
	
	double timeConverter = UnitTime;
	double lengthConverter = UnitLength;
	double massConverter = UnitMass; 
	double velocityConverter = UnitLength/UnitTime; 
	double momentumConverter = UnitMass*UnitLength*UnitLength/UnitTime;
	
	findEarthAndMoon();
	int earthFeCountBody1 = 0;
	int earthFeCountBody2 = 0;
	int earthSiCountBody1 = 0;
	int earthSiCountBody2 = 0;
	int moonFeCountBody1 = 0;
	int moonFeCountBody2 = 0;
	int moonSiCountBody1 = 0;
	int moonSiCountBody2 = 0;
	
	float massUniversalSystem = getMassCollision(0);
	float massEarthMoonSystem = getMassCollision(1);
	float massEarth = getMassCollision(2);
	float massMoon = getMassCollision(3);
	
	float3 centerOfMassUniversalSystem = getCenterOfMassCollision(0);
	float3 centerOfMassEarthMoonSystem = getCenterOfMassCollision(1);
	float3 centerOfMassEarth = getCenterOfMassCollision(2);
	float3 centerOfMassMoon = getCenterOfMassCollision(3);
	
	float3 linearVelocityUniversalSystem = getLinearVelocityCollision(0);
	float3 linearVelocityEarthMoonSystem = getLinearVelocityCollision(1);
	float3 linearVelocityEarth = getLinearVelocityCollision(2);
	float3 linearVelocityMoon = getLinearVelocityCollision(3);
	
	float3 angularMomentumUniversalSystem = getAngularMomentumCollision(0);
	float3 angularMomentumEarthMoonSystem = getAngularMomentumCollision(1);
	float3 angularMomentumEarth = getAngularMomentumCollision(2);
	float3 angularMomentumMoon = getAngularMomentumCollision(3);
	
	for(int i = 0; i < NumberOfEarthElements; i++)
	{
		if(EarthIndex[i] < NFe1) 			earthFeCountBody1++;
		else if(EarthIndex[i] < NFe1 + NFe2) 		earthFeCountBody2++;
		else if(EarthIndex[i] < NFe1 + NFe2 + NSi1) 	earthSiCountBody1++;
		else 						earthSiCountBody2++;
	}
	
	for(int i = 0; i < NumberOfMoonElements; i++)
	{
		if(MoonIndex[i] < NFe1) 			moonFeCountBody1++;
		else if(MoonIndex[i] < NFe1 + NFe2) 		moonFeCountBody2++;
		else if(MoonIndex[i] < NFe1 + NFe2 + NSi1) 	moonSiCountBody1++;
		else 						moonSiCountBody2++;
	}
	
	FILE *RunStatsFile = fopen("./runStatsFile","wb");
	
	fprintf(RunStatsFile,"\n\n\n*************************************************************************\n\n");
	fprintf(RunStatsFile,"\nThe following are the final stats of the run when time = %f hours\n", time*timeConverter/3600.0);
	fprintf(RunStatsFile,"\nDistance is measured in Kilometers");
	fprintf(RunStatsFile,"\nMass is measured in Kilograms");
	fprintf(RunStatsFile,"\nTime is measured in seconds");
	fprintf(RunStatsFile,"\nVelocity is measured in Kilometers/second");
	fprintf(RunStatsFile,"\nAngular momentun is measured in Kilograms*Kilometers*Kilometers/seconds\n");
	
	fprintf(RunStatsFile,"\nThe mass of Earth 		= %e", massEarth*massConverter);
	fprintf(RunStatsFile,"\nThe mass of Moon 		= %e", massMoon*massConverter);
	if(massMoon != 0.0) fprintf(RunStatsFile,"\nThe mass ratio Earth/Moon 	= %f\n", massEarth/massMoon);
	
	fprintf(RunStatsFile,"\nMoon iron from body 1 		= %d", moonFeCountBody1);
	fprintf(RunStatsFile,"\nMoon silicate from body 1 	= %d", moonSiCountBody1);
	fprintf(RunStatsFile,"\nMoon iron from body 2 		= %d", moonFeCountBody2);
	fprintf(RunStatsFile,"\nMoon silicate from body 2 	= %d", moonSiCountBody2);
	if((moonFeCountBody2 + moonSiCountBody2) == 0)
	{
		fprintf(RunStatsFile,"\nThe Moon is only composed of elements from body 1\n");
	}
	else if((moonFeCountBody1 + moonSiCountBody1) == 0)
	{
		fprintf(RunStatsFile,"\nThe Moon is only composed of elements from body 2\n");
	}
	else
	{
		fprintf(RunStatsFile,"\nMoon ratio body1/body2 		= %f\n", (float)(moonFeCountBody1 + moonSiCountBody1)/(float)(moonFeCountBody2 + moonSiCountBody2));
	}
	
	fprintf(RunStatsFile,"\nEarth iron from body 1 		= %d", earthFeCountBody1);
	fprintf(RunStatsFile,"\nEarth silicate from body 1 	= %d", earthSiCountBody1);
	fprintf(RunStatsFile,"\nEarth iron from body 2 		= %d", earthFeCountBody2);
	fprintf(RunStatsFile,"\nEarth silicate from body 2 	= %d", earthSiCountBody2);
	if((earthFeCountBody2 + earthSiCountBody2) == 0)
	{
		fprintf(RunStatsFile,"\nThe Earth is only composed of elements from body 1\n");
	}
	else if((earthFeCountBody1 + earthSiCountBody1) == 0)
	{
		fprintf(RunStatsFile,"\nThe Earth is only composed of elements from body 2\n");
	}
	else
	{
		fprintf(RunStatsFile,"\nEarth ratio body1/body2 		= %f\n", (float)(earthFeCountBody1 + earthSiCountBody1)/(float)(earthFeCountBody2 + earthSiCountBody2));
	}
	
	//It is always assumed that the ecliptic plane is the xz-plane.
	x = angularMomentumEarthMoonSystem.x*momentumConverter;
	y = angularMomentumEarthMoonSystem.y*momentumConverter;
	z = angularMomentumEarthMoonSystem.z*momentumConverter;
	fprintf(RunStatsFile,"\nAngular momentum of the Earth Moon system 		= (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	fprintf(RunStatsFile,"\nMagnitude of the angular momentum of the system 	= %e", mag);
	size = sqrt(x*x + y*y + z*z) * sqrt(x*x + z*z);
	angle = acos((x*x + z*z)/size);
	fprintf(RunStatsFile,"\nAngle off ecliptic plane of the system's rotation 	= %f\n", 90.0 - angle*180.0/Pi);
	
	x = angularMomentumEarth.x*momentumConverter;
	y = angularMomentumEarth.y*momentumConverter;
	z = angularMomentumEarth.z*momentumConverter;
	fprintf(RunStatsFile,"\nAngular momentum of the Earth 				= (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	fprintf(RunStatsFile,"\nMagnitude of the angular momentum of the Earth 		= %e", mag);
	size = sqrt(x*x + y*y + z*z) * sqrt(x*x + z*z);
	angle = acos((x*x + z*z)/size);
	fprintf(RunStatsFile,"\nAngle off ecliptic plane of the Earth's rotation 	= %f\n", 90.0 - angle*180.0/Pi);
	
	x = angularMomentumMoon.x*momentumConverter;
	y = angularMomentumMoon.y*momentumConverter;
	z = angularMomentumMoon.z*momentumConverter;
	fprintf(RunStatsFile,"\nAngular momentum of the Moon 				= (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	fprintf(RunStatsFile,"\nMagnitude of the angular momentum of the Moon 		= %e", mag);
	size = sqrt(x*x + y*y + z*z) * sqrt(x*x + z*z);
	angle = acos((x*x + z*z)/size);
	fprintf(RunStatsFile,"\nAngle off ecliptic plane of the Moon's rotation 	= %f\n", 90.0 - angle*180.0/Pi);
	
	x = centerOfMassEarthMoonSystem.x*lengthConverter;
	y = centerOfMassEarthMoonSystem.y*lengthConverter;
	z = centerOfMassEarthMoonSystem.z*lengthConverter;
	fprintf(RunStatsFile,"\nCenter of mass of the Earth-Moon system 		= (%f, %f, %f)", x, y, z);
	
	x = centerOfMassEarth.x*lengthConverter;
	y = centerOfMassEarth.y*lengthConverter;
	z = centerOfMassEarth.z*lengthConverter;
	fprintf(RunStatsFile,"\nCenter of mass of the Earth system 			= (%f, %f, %f)", x, y, z);
	
	x = centerOfMassMoon.x*lengthConverter;
	y = centerOfMassMoon.y*lengthConverter;
	z = centerOfMassMoon.z*lengthConverter;
	fprintf(RunStatsFile,"\nCenter of mass of the Moon system 			= (%f, %f, %f)\n", x, y, z);
	
	x = linearVelocityEarthMoonSystem.x*velocityConverter;
	y = linearVelocityEarthMoonSystem.y*velocityConverter;
	z = linearVelocityEarthMoonSystem.z*velocityConverter;
	fprintf(RunStatsFile,"\nLinear Velocity of the Earth-Moon system 		= (%f, %f, %f)", x, y, z);
	
	x = linearVelocityEarth.x*velocityConverter;
	y = linearVelocityEarth.y*velocityConverter;
	z = linearVelocityEarth.z*velocityConverter;
	fprintf(RunStatsFile,"\nLinear Velocity of the Earth system 			= (%f, %f, %f)", x, y, z);
	
	x = linearVelocityMoon.x*velocityConverter;
	y = linearVelocityMoon.y*velocityConverter;
	z = linearVelocityMoon.z*velocityConverter;
	fprintf(RunStatsFile,"\nLinear Velocity of the Moon system 			= (%f, %f, %f)\n", x, y, z);
	
	fprintf(RunStatsFile,"\n*****Stats of the entire system to check the numerical scheme's validity*****\n");
	
	x = centerOfMassUniversalSystem.x*lengthConverter;
	y = centerOfMassUniversalSystem.y*lengthConverter;
	z = centerOfMassUniversalSystem.z*lengthConverter;
	fprintf(RunStatsFile,"\nCenter of mass of the entire system 		        = (%f, %f, %f)\n", x, y, z);
	fprintf(RunStatsFile,"\nTotal mass of the entire system 	= %f\n", massUniversalSystem);
	
	x = linearVelocityUniversalSystem.x*velocityConverter;
	y = linearVelocityUniversalSystem.y*velocityConverter;
	z = linearVelocityUniversalSystem.z*velocityConverter;
	fprintf(RunStatsFile,"\nLinear velocity of the entire system system 		= (%f, %f, %f)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	fprintf(RunStatsFile,"\nMagnitude of the linear velocity of the entire system 	= %f\n", mag);
	
	x = angularMomentumUniversalSystem.x*momentumConverter;
	y = angularMomentumUniversalSystem.y*momentumConverter;
	z = angularMomentumUniversalSystem.z*momentumConverter;
	fprintf(RunStatsFile,"\nAngular momentum of the entire system system 		= (%e, %e, %e)", x, y, z);
	mag = sqrt(x*x + y*y + z*z);
	fprintf(RunStatsFile,"\nMagnitude of the angular momentum of the entire system 	= %e\n", mag);
	
	fprintf(RunStatsFile,"\n\n*************************************************************************\n\n\n");
	fclose(RunStatsFile);
}

void recordContinuePosAndVel(double time)
{
	size_t returnValue;
	
	FILE *ContinueRunPosAndVelFile = fopen("./continueRunPosAndVelFile","wb");
	fseek(ContinueRunPosAndVelFile,0,SEEK_SET);
	
	returnValue = fwrite(&time, sizeof(double), 1, ContinueRunPosAndVelFile);
	if(returnValue != TotalNumberOfElements)
	{
		printf("\nTSU error: Error writing time to ContinueRunPosAndVelFile|n");
		exit(0);
	}
	returnValue = fwrite(Pos, sizeof(float4), TotalNumberOfElements, ContinueRunPosAndVelFile);
	if(returnValue != TotalNumberOfElements)
	{
		printf("\nTSU error: Error writing positions to ContinueRunPosAndVelFile|n");
		exit(0);
	}
	returnValue = fwrite(Vel, sizeof(float4), TotalNumberOfElements, ContinueRunPosAndVelFile);
	if(returnValue != TotalNumberOfElements)
	{
		printf("\nTSU error: Error writing velocities to ContinueRunPosAndVelFile|n");
		exit(0);
	}
	fclose(ContinueRunPosAndVelFile);
}

void cleanUpImpact()
{
	free(Pos);
	free(Vel);
	free(Force);
	
	cudaFree(Pos_DEV0);
	cudaFree(Vel_DEV0);
	cudaFree(Force_DEV0);
}
