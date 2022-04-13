// Prototypes of all the functions in this file
double4 getCenterOfMass(int);
double getRadiusOfBody(int);
double4 getAverageLinearVelocity(int);
double4 getAngularMomentum(int);
void createRawTargetImpactor();
void spinBody(int, float4);
void zeroOutTargetImpactorDrift();
void drawPictureSeperate();
void recordTargetImpactorStats();
void recordCarryForwardParameters();
void recordTargetImpactorInitialPosVel();
void cleanUpGenerateTargetImpactor();

// These first two functions are so that the callback functions have a common function to call
// (drawpicture and runSimulation) that is redirected to the correct funtion for this code.
// They are prototyped in the commonFunctions.h file.
void doStuff()
{
	generateTargetImpactor();
}

void drawStuff()
{
	drawPictureSeperate();
}

// The following are functions that I pulled out of the main code so the controll of the run is not so clutered.
double4 getCenterOfMass(int scope)
{
	double4 centerOfMass;
	double totalMass;
	int startFe, stopFe, startSi, stopSi;
	
	centerOfMass.x = 0.0;
	centerOfMass.y = 0.0;
	centerOfMass.z = 0.0;
	centerOfMass.w = 0.0;
	
	if(scope == 1)
	{
		totalMass = MassOfBody1;
		startFe = 0;
		stopFe = NFe1;
		startSi = NFe1;
		stopSi = NFe1 + NSi1;
	}
	else if(scope == 2)
	{
		totalMass = MassOfBody2;
		startFe = NFe1 + NSi1;
		stopFe = NFe1 + NSi1 + NFe2;
		startSi = NFe1 + NSi1 + NFe2;
		stopSi = NFe1 + NSi1 + NFe2 + NSi2;
	}
	else 
	{
		printf("\n\n\n Bad scope getCenterOfMass");
		exit(0);
	}
	if(totalMass < ASSUMEZERODOUBLE) return(centerOfMass);
	
	for(int i = startFe; i < stopFe; i++)
	{
    	centerOfMass.x += Pos[i].x*MassFe;
		centerOfMass.y += Pos[i].y*MassFe;
		centerOfMass.z += Pos[i].z*MassFe;
	}
	for(int i = startSi; i < stopSi; i++)
	{
    	centerOfMass.x += Pos[i].x*MassSi;
		centerOfMass.y += Pos[i].y*MassSi;
		centerOfMass.z += Pos[i].z*MassSi;
	}
	centerOfMass.x /= totalMass;
	centerOfMass.y /= totalMass;
	centerOfMass.z /= totalMass;
	centerOfMass.w = sqrt(centerOfMass.x*centerOfMass.x + centerOfMass.y*centerOfMass.y + centerOfMass.z*centerOfMass.z);
	return(centerOfMass);
}

double getRadiusOfBody(int scope)
{
	int start, stop;
	double3 r;
	double4 centerOfMass;
	double d, radius;
	
	if(scope == 1)
	{
		start = 0;
		stop = NFe1 + NSi1;
	}
	else if(scope == 2)
	{
		start = NFe1 + NSi1;
		stop = NFe1 + NSi1 + NFe2 + NSi2;
	}
	else 
	{
		printf("\n\n\n Bad scope getRadiusOfBody");
		exit(0);
	}
	
	centerOfMass = getCenterOfMass(scope);
	for(int i = start; i < stop; i++)
	{
		r.x = (double)Pos[i].x - centerOfMass.x;
		r.y = (double)Pos[i].y - centerOfMass.y;
		r.z = (double)Pos[i].z - centerOfMass.z;
		
		d = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
		
		if(d > radius) radius = d;
	}
	return(radius);
}

double4 getAverageLinearVelocity(int scope)
{
	double totalMass;
	double4 linearVelocity;
	int startFe, stopFe, startSi, stopSi;
	
	linearVelocity.x = 0.0;
	linearVelocity.y = 0.0;
	linearVelocity.z = 0.0;
	linearVelocity.w = 0.0;
	
	if(scope == 1)
	{
		totalMass = MassOfBody1;
		startFe = 0;
		stopFe = NFe1;
		startSi = NFe1;
		stopSi = NFe1 + NSi1;
	}
	else if(scope == 2)
	{
		totalMass = MassOfBody2;
		startFe = NFe1 + NSi1;
		stopFe = NFe1 + NSi1 + NFe2;
		startSi = NFe1 + NSi1 + NFe2;
		stopSi = NFe1 + NSi1 + NFe2 + NSi2;
	}
	else 
	{
		printf("\n\n\n Bad scope getCenterOfMass");
		exit(0);
	}
	if(totalMass < ASSUMEZERODOUBLE) return(linearVelocity);
	
	for(int i = startFe; i < stopFe; i++)
	{
    	linearVelocity.x += Vel[i].x*MassFe;
		linearVelocity.y += Vel[i].y*MassFe;
		linearVelocity.z += Vel[i].z*MassFe;
	}
	for(int i = startSi; i < stopSi; i++)
	{
    	linearVelocity.x += Vel[i].x*MassSi;
		linearVelocity.y += Vel[i].y*MassSi;
		linearVelocity.z += Vel[i].z*MassSi;
	}
	
	linearVelocity.x /= totalMass;
	linearVelocity.y /= totalMass;
	linearVelocity.z /= totalMass;
	linearVelocity.w = sqrt(linearVelocity.x*linearVelocity.x + linearVelocity.y*linearVelocity.y + linearVelocity.z*linearVelocity.z);
	return(linearVelocity);
}

double4 getAngularMomentum(int scope)
{
	double4 angularMomentum;
	double3 r, v;
	int startFe, stopFe, startSi, stopSi;
	
	double4 centerOfMass = getCenterOfMass(scope);
	double4 averageLinearVelocity = getAverageLinearVelocity(scope);
	
	angularMomentum.x = 0.0;
	angularMomentum.y = 0.0;
	angularMomentum.z = 0.0;
	angularMomentum.w = 0.0;
	
	if(scope == 1)
	{
		startFe = 0;
		stopFe = NFe1;
		startSi = NFe1;
		stopSi = NFe1 + NSi1;
	}
	else if(scope == 2)
	{
		startFe = NFe1 + NSi1;
		stopFe = NFe1 + NSi1 + NFe2;
		startSi = NFe1 + NSi1 + NFe2;
		stopSi = NFe1 + NSi1 + NFe2 + NSi2;
	}
	else 
	{
		printf("\n\n\n Bad scope getCenterOfMass");
		exit(0);
	}
	
	for(int i = startFe; i < stopFe; i++)
	{
		r.x = Pos[i].x - centerOfMass.x;
		r.y = Pos[i].y - centerOfMass.y;
		r.z = Pos[i].z - centerOfMass.z;
	
		v.x = Vel[i].x - averageLinearVelocity.x;
		v.y = Vel[i].y - averageLinearVelocity.y;
		v.z = Vel[i].z - averageLinearVelocity.z;
	
		angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassFe;
		angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassFe;
		angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassFe;
	}
	for(int i = startSi; i < stopSi; i++)
	{
		r.x = Pos[i].x - centerOfMass.x;
		r.y = Pos[i].y - centerOfMass.y;
		r.z = Pos[i].z - centerOfMass.z;
	
		v.x = Vel[i].x - averageLinearVelocity.x;
		v.y = Vel[i].y - averageLinearVelocity.y;
		v.z = Vel[i].z - averageLinearVelocity.z;
	
		angularMomentum.x +=  (r.y*v.z - r.z*v.y)*MassSi;
		angularMomentum.y += -(r.x*v.z - r.z*v.x)*MassSi;
		angularMomentum.z +=  (r.x*v.y - r.y*v.x)*MassSi;
	}
	angularMomentum.w = sqrt(angularMomentum.x*angularMomentum.x + angularMomentum.y*angularMomentum.y + angularMomentum.z*angularMomentum.z);
	return(angularMomentum);
}

void createRawTargetImpactor()
{
	double radius1, radius2, stretch;
	double volume, mag, radius, seperation;
	int test, repeatCount;
	time_t t;
	
	printf("\n Creating the raw bodies");
	//Creating body one
	//This assumes a 68% packing ratio of a shpere with shperes and then stretches it by strecth 
	//to safely fit all the elements in.
	stretch = 2.0;
	volume = ((4.0/3.0)*Pi*pow(Diameter,3)*(double)NFe1/0.68)*stretch;
	radius1 = pow(volume/((4.0/3.0)*Pi),(1.0/3.0));
	volume = ((4.0/3.0)*Pi*pow(Diameter,3)*(double)(NFe1 + NSi1)/0.68)*stretch;
	radius2 = pow(volume/((4.0/3.0)*Pi),(1.0/3.0));
	srand((unsigned) time(&t));
	
	repeatCount = 0;
	for(int i=0; i<NFe1; i++)
	{
		test = 0;
		while(test == 0)
		{
			Pos[i].x = ((double)rand()/(double)RAND_MAX)*2.0 - 1.0;
			Pos[i].y = ((double)rand()/(double)RAND_MAX)*2.0 - 1.0;
			Pos[i].z = ((double)rand()/(double)RAND_MAX)*2.0 - 1.0;
			mag = sqrt(Pos[i].x*Pos[i].x + Pos[i].y*Pos[i].y + Pos[i].z*Pos[i].z);
			radius = ((double)rand()/(double)RAND_MAX)*radius1;
			Pos[i].x *= radius/mag;
			Pos[i].y *= radius/mag;
			Pos[i].z *= radius/mag;
			test = 1;
			for(int j = 0; j < i; j++)
			{
				seperation = mag = sqrt((Pos[i].x-Pos[j].x)*(Pos[i].x-Pos[j].x) + (Pos[i].y-Pos[j].y)*(Pos[i].y-Pos[j].y) + (Pos[i].z-Pos[j].z)*(Pos[i].z-Pos[j].z));
				if(seperation < Diameter)
				{
					test = 0;
					repeatCount++;
					break;
				}
			}
		}
		Pos[i].w = 0.0;
		
		Vel[i].x = 0.0;
		Vel[i].y = 0.0;
		Vel[i].z = 0.0;
		Vel[i].w = MassFe;
	}
	
	for(int i = NFe1; i < (NFe1 + NSi1); i++)
	{
		test = 0;
		while(test == 0)
		{
			Pos[i].x = ((double)rand()/(double)RAND_MAX)*2.0 - 1.0;
			Pos[i].y = ((double)rand()/(double)RAND_MAX)*2.0 - 1.0;
			Pos[i].z = ((double)rand()/(double)RAND_MAX)*2.0 - 1.0;
			mag = sqrt(Pos[i].x*Pos[i].x + Pos[i].y*Pos[i].y + Pos[i].z*Pos[i].z);
			radius = ((double)rand()/(double)RAND_MAX)*(radius2-radius1) + radius1 + Diameter;
			Pos[i].x *= radius/mag;
			Pos[i].y *= radius/mag;
			Pos[i].z *= radius/mag;
			test = 1;
			for(int j = NFe1; j < i; j++)
			{
				seperation = mag = sqrt((Pos[i].x-Pos[j].x)*(Pos[i].x-Pos[j].x) + (Pos[i].y-Pos[j].y)*(Pos[i].y-Pos[j].y) + (Pos[i].z-Pos[j].z)*(Pos[i].z-Pos[j].z));
				if(seperation < Diameter)
				{
					test = 0;
					repeatCount++;
					break;
				}
			}
		}
		Pos[i].w = 1.0;
		
		Vel[i].x = 0.0;
		Vel[i].y = 0.0;
		Vel[i].z = 0.0;
		Vel[i].w = MassSi;
	}
	printf("\n repeat count body one= %d", repeatCount);
	
	//Creating body two
	//This assumes a 68% packing ratio of a shpere with shperes and then stretches it by strecth 
 	//to safely fit all the balls in.
	stretch = 2.0;
	volume = ((4.0/3.0)*Pi*pow(Diameter,3)*(double)NFe2/0.68)*stretch;
	radius1 = pow(volume/((4.0/3.0)*Pi),(1.0/3.0));
	volume = ((4.0/3.0)*Pi*pow(Diameter,3)*(double)(NFe2 + NSi2)/0.68)*stretch;
	radius2 = pow(volume/((4.0/3.0)*Pi),(1.0/3.0));
	srand((unsigned) time(&t));
	
	repeatCount = 0;
	for(int i = (NFe1 + NSi1); i < (NFe1 + NSi1 + NFe2); i++)
	{
		test = 0;
		while(test == 0)
		{
			Pos[i].x = ((double)rand()/(double)RAND_MAX)*2.0 - 1.0;
			Pos[i].y = ((double)rand()/(double)RAND_MAX)*2.0 - 1.0;
			Pos[i].z = ((double)rand()/(double)RAND_MAX)*2.0 - 1.0;
			mag = sqrt(Pos[i].x*Pos[i].x + Pos[i].y*Pos[i].y + Pos[i].z*Pos[i].z);
			radius = ((double)rand()/(double)RAND_MAX)*radius1;
			Pos[i].x *= radius/mag;
			Pos[i].y *= radius/mag;
			Pos[i].z *= radius/mag;
			test = 1;
			for(int j = (NFe1 + NSi1); j < i; j++)
			{
				seperation = mag = sqrt((Pos[i].x-Pos[j].x)*(Pos[i].x-Pos[j].x) + (Pos[i].y-Pos[j].y)*(Pos[i].y-Pos[j].y) + (Pos[i].z-Pos[j].z)*(Pos[i].z-Pos[j].z));
				if(seperation < Diameter)
				{
					test = 0;
					repeatCount++;
					break;
				}
			}
		}
		Pos[i].w = 2.0;
		
		Vel[i].x = 0.0;
		Vel[i].y = 0.0;
		Vel[i].z = 0.0;
		Vel[i].w = MassFe;
	}
	for(int i = (NFe1 + NSi1 + NFe2); i < TotalNumberOfElements; i++)
	{
		test = 0;
		while(test == 0)
		{
			Pos[i].x = ((double)rand()/(double)RAND_MAX)*2.0 - 1.0;
			Pos[i].y = ((double)rand()/(double)RAND_MAX)*2.0 - 1.0;
			Pos[i].z = ((double)rand()/(double)RAND_MAX)*2.0 - 1.0;
			mag = sqrt(Pos[i].x*Pos[i].x + Pos[i].y*Pos[i].y + Pos[i].z*Pos[i].z);
			radius = ((double)rand()/(double)RAND_MAX)*(radius2-radius1) + radius1 + Diameter;
			Pos[i].x *= radius/mag;
			Pos[i].y *= radius/mag;
			Pos[i].z *= radius/mag;
			test = 1;
			for(int j = (NFe1 + NSi1 + NFe2); j < i; j++)
			{
				seperation = mag = sqrt((Pos[i].x-Pos[j].x)*(Pos[i].x-Pos[j].x) + (Pos[i].y-Pos[j].y)*(Pos[i].y-Pos[j].y) + (Pos[i].z-Pos[j].z)*(Pos[i].z-Pos[j].z));
				if(seperation < Diameter)
				{
					test = 0;
					repeatCount++;
					break;
				}
			}
		}
		Pos[i].w = 3.0;
		
		Vel[i].x = 0.0;
		Vel[i].y = 0.0;
		Vel[i].z = 0.0;
		Vel[i].w = MassSi;
	}
	printf("\n repeat count body two = %d", repeatCount);
	
	printf("\n ***********************************************************************");
	printf("\n Raw bodies have been formed");
	printf("\n ***********************************************************************");
}

void spinBody(int bodyId, float4 spinVector)
{
	double3 r; //vector from center of mass to the position vector
	double4 centerOfMass;
	double3	n; //Unit vector perpendicular to the plane of spin
	double 	mag;
	double 	assumeZero = 0.0000001;
	int	start, stop;
	
	if(bodyId == 1)
	{
		start = 0;
		stop = NFe1 + NSi1;
	}
	else
	{
		start = NFe1 + NSi1;
		stop = TotalNumberOfElements;
	}
	
	//Making sure the spin vector is a unit vector
	mag = sqrt(spinVector.x*spinVector.x + spinVector.y*spinVector.y + spinVector.z*spinVector.z);
	if(assumeZero < mag)
	{
		spinVector.x /= mag;
		spinVector.y /= mag;
		spinVector.z /= mag;
	}
	else 
	{
		printf("\nTSU Error: In spinBodySeperate. The spin direction vector is zero.\n");
		exit(0);
	}
	
	centerOfMass = getCenterOfMass(bodyId);
	for(int i = start; i < stop; i++)
	{
		//Creating a vector from the center of mass to the point
		r.x = Pos[i].x - centerOfMass.x;
		r.y = Pos[i].y - centerOfMass.y;
		r.z = Pos[i].z - centerOfMass.z;
		double magsquared = r.x*r.x + r.y*r.y + r.z*r.z;
		double spinDota = spinVector.x*r.x + spinVector.y*r.y + spinVector.z*r.z;
		double perpendicularDistance = sqrt(magsquared - spinDota*spinDota);
		double perpendicularVelocity = spinVector.w*2.0*Pi*perpendicularDistance;
		
		//finding unit vector perpendicular to both the position vector and the spin vector
		n.x =  (spinVector.y*r.z - spinVector.z*r.y);
		n.y = -(spinVector.x*r.z - spinVector.z*r.x);
		n.z =  (spinVector.x*r.y - spinVector.y*r.x);
		mag = sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
		if(mag != 0.0)
		{
			n.x /= mag;
			n.y /= mag;
			n.z /= mag;
				
			//Spining the element
			Vel[i].x += perpendicularVelocity*n.x;
			Vel[i].y += perpendicularVelocity*n.y;
			Vel[i].z += perpendicularVelocity*n.z;
		}
	}		
}

void zeroOutTargetImpactorDrift()
{
	int	start, stop;
	double4 centerOfMass, linearVelocity;
	
	centerOfMass = getCenterOfMass(1); 
	linearVelocity = getAverageLinearVelocity(1); 
	start = 0;
	stop = NFe1 + NSi1;
	for(int i = start; i < stop; i++)
	{
		Pos[i].x -= centerOfMass.x;
		Pos[i].y -= centerOfMass.y;
		Pos[i].z -= centerOfMass.z;
		
		Vel[i].x -= linearVelocity.x;
		Vel[i].y -= linearVelocity.y;
		Vel[i].z -= linearVelocity.z;
	}	
	
	centerOfMass = getCenterOfMass(2); 
	linearVelocity = getAverageLinearVelocity(2); 
	start = NFe1 + NSi1;
	stop = TotalNumberOfElements;
	for(int i = start; i < stop; i++)
	{
		Pos[i].x -= centerOfMass.x;
		Pos[i].y -= centerOfMass.y;
		Pos[i].z -= centerOfMass.z;
		
		Vel[i].x -= linearVelocity.x;
		Vel[i].y -= linearVelocity.y;
		Vel[i].z -= linearVelocity.z;
	}	
}

void drawPictureSeperate()
{
	double4 centerOfMass1 = getCenterOfMass(1);
	double4 centerOfMass2 = getCenterOfMass(2);
	double4 linearVelocity1 = getAverageLinearVelocity(1);
	double4 linearVelocity2 = getAverageLinearVelocity(2);
	double4 angularMomentum1 = getAngularMomentum(1);
	double4 angularMomentum2 = getAngularMomentum(2);
	double Stretch, shift, mag;
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	//Coloring all the elements 
	glPointSize(2.0);
	glBegin(GL_POINTS);
     	for(int i = 0; i < TotalNumberOfElements; i++)
		{
			if(i < NFe1) 
			{
		    	glColor3d(1.0,0.0,0.0);
		    	shift = -4.0*RadiusOfEarth;
			}
			else if(i < NFe1 + NSi1)
			{
				glColor3d(1.0,1.0,0.5);
				shift = -4.0*RadiusOfEarth;
			}
			else if(i < NFe1 + NSi1 + NFe2) 
			{
		    	glColor3d(1.0,0.0,1.0);
		    	shift = 4.0*RadiusOfEarth;
			}
			else
			{
				glColor3d(0.0,0.5,0.0);
				shift = 4.0*RadiusOfEarth;
			}
			
			glVertex3f(Pos[i].x + shift, Pos[i].y, Pos[i].z);
		}
	glEnd();

	glLineWidth(1.0);
	shift = 4.0*RadiusOfEarth;
	//Placing a blue vector in the direction of the disired angular momentum 
	glColor3f(0.0,0.0,1.0);	
	Stretch = 20.0;
	mag = sqrt(InitialSpin1.x*InitialSpin1.x + InitialSpin1.y*InitialSpin1.y + InitialSpin1.z*InitialSpin1.z);
	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMass1.x - shift, centerOfMass1.y, centerOfMass1.z);
		glVertex3f(centerOfMass1.x + (InitialSpin1.x/mag)*Stretch - shift, centerOfMass1.y + (InitialSpin1.y/mag)*Stretch, centerOfMass1.z + (InitialSpin1.z/mag)*Stretch);
	glEnd();
	
	mag = sqrt(InitialSpin2.x*InitialSpin2.x + InitialSpin2.y*InitialSpin2.y + InitialSpin2.z*InitialSpin2.z);
	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMass2.x + shift, centerOfMass2.y, centerOfMass2.z);
		glVertex3f(centerOfMass2.x + (InitialSpin2.x/mag)*Stretch + shift, centerOfMass2.y + (InitialSpin2.y/mag)*Stretch, centerOfMass2.z + (InitialSpin2.z/mag)*Stretch);
	glEnd();
	
	//Placing a red vector in the direction of the actual angular momentum 
	glColor3f(1.0,0.0,0.0);	
	Stretch = 20.0;
	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMass1.x - shift, centerOfMass1.y, centerOfMass1.z);
		glVertex3f(centerOfMass1.x + (angularMomentum1.x/angularMomentum1.w)*Stretch - shift, centerOfMass1.y + (angularMomentum1.y/angularMomentum1.w)*Stretch, centerOfMass1.z + (angularMomentum1.z/angularMomentum1.w)*Stretch);
	glEnd();
	
	glBegin(GL_LINE_LOOP);
		glVertex3f(centerOfMass2.x + shift, centerOfMass2.y, centerOfMass2.z);
		glVertex3f(centerOfMass2.x + (angularMomentum2.x/angularMomentum1.w)*Stretch + shift, centerOfMass2.y + (angularMomentum2.y/angularMomentum1.w)*Stretch, centerOfMass2.z + (angularMomentum2.z/angularMomentum1.w)*Stretch);
	glEnd();
	
	glutSwapBuffers();
	
	// Making a video of the run.
	if(MovieOn == 1)
	{
		glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, MovieBuffer);
		fwrite(MovieBuffer, sizeof(int)*XWindowSize*YWindowSize, 1, MovieFile);
	}
}

void recordTargetImpactorStats()
{
	FILE *file;
	double mag;
	double radiusOfBody;
	double massOfBody;
	
	double4 centerOfMass;
	double4 linearVelocity;
	double4 angularMomentum;
	
	double lengthConvertion = UnitLength;
	double massConvertion = UnitMass;
	double velocityConvertion = UnitLength/UnitTime;
	double AngularMomentumConvertion = (UnitMass*UnitLength*UnitLength)/(UnitTime);
	
	file = fopen("./TargetImpactorInformation/targetImpactorStats", "wb");
		fprintf(file, "The conversion parameters to take you to and from our units to the real world units follow\n");
		
		fprintf(file, "\n Simulation length unit is %f kilometers", UnitLength);
		fprintf(file, "\n Simulation mass unit is %e kilograms", UnitMass);
		fprintf(file, "\n Simulation time unit is %f seconds", UnitTime);
		
		fprintf(file, "\n\n The initail statistics for this run in Simulation units follow\n");
		fprintf(file, "\n Diameter of an element: 			Diameter = %f", Diameter);
		fprintf(file, "\n Gravitational constant: 			Gravity = %f", Gravity);
		fprintf(file, "\n The mass of a silicate element: 	MassSi = %f", MassSi);
		fprintf(file, "\n The mass of an iron element: 		MassFe = %f", MassFe);
		
		fprintf(file, "\n\n The push back strength of iron: 	KFe = %f", KFe);
		fprintf(file, "\n The push back strength of silicate: 	KSi = %f\n", KSi);
		
		fprintf(file, "\n The mass of body one: 	MassOfBody1 = %f", MassOfBody1);
		fprintf(file, "\n The mass of body two: 	MassOfBody2 = %f\n", MassOfBody2);
		
		mag = sqrt(InitialSpin1.x*InitialSpin1.x + InitialSpin1.y*InitialSpin1.y + InitialSpin1.z*InitialSpin1.z);
		fprintf(file, "\n The initial spin in revolutions per time unit of body one: (%f, %f, %f, %f)", InitialSpin1.x/mag, InitialSpin1.y/mag, InitialSpin1.z/mag, InitialSpin1.w);
		mag = sqrt(InitialSpin2.x*InitialSpin2.x + InitialSpin2.y*InitialSpin2.y + InitialSpin2.z*InitialSpin2.z);
		fprintf(file, "\n The initial spin in revolutions per time unit of body two: (%f, %f, %f, %f)\n", InitialSpin2.x/mag, InitialSpin2.y/mag, InitialSpin2.z/mag, InitialSpin2.w);
		
		fprintf(file, "\n Total number of elements: 						TotalNumberOfElements = %d", TotalNumberOfElements);
		fprintf(file, "\n Total number of iron elements: 					NFe = %d", NFe);
		fprintf(file, "\n Total number of silicate elements: 				NSi = %d", NSi);
		fprintf(file, "\n Total number of iron elements in body1: 			NFe1 = %d", NFe1);
		fprintf(file, "\n Total number of silicate elements in body1: 		NSi1 = %d", NSi1);
		fprintf(file, "\n Total number of iron elements in body2 			NFe2: = %d", NFe2);
		fprintf(file, "\n Total number of silicate elements in body2: 		NSi2 = %d\n", NSi2);
		
		fprintf(file, "\n Time step in Simulation units: 		Dt = %f", Dt);
		fprintf(file, "\n Time step in seconds:					Dt = %f", Dt*UnitTime);
		fprintf(file, "\n Damp time in Simulation units: 		DampTime = %f", DampTime);
		fprintf(file, "\n Damp rest time in Simulation units: 	DampRestTime = %f", DampRestTime);
		fprintf(file, "\n Spin rest time in Simulation units: 	SpinRestTime = %f", SpinRestTime);
		
		
		fprintf(file, "\n\n\n*****************************************************************************************************");
		fprintf(file, "\nThe follow are the statistics of the final created bodies in real world units");
		fprintf(file, "\n*****************************************************************************************************");
		
		centerOfMass = getCenterOfMass(1);
		radiusOfBody = getRadiusOfBody(1);
		massOfBody = (double)MassFe*(double)NFe1 + (double)MassSi*(double)NSi1;
		
		fprintf(file, "\n\n ***** Stats for Body1 *****");
		fprintf(file, "\n Mass =  %e Kilograms", massOfBody*massConvertion);
		fprintf(file, "\n Radius =  %f Kilometers", radiusOfBody*lengthConvertion);
		
		fprintf(file, "\n The center of mass = (%f, %f, %f) Kilometers from (0, 0, 0)", centerOfMass.x*lengthConvertion, centerOfMass.y*lengthConvertion, centerOfMass.z*lengthConvertion);
		
		linearVelocity = getAverageLinearVelocity(1);
		fprintf(file, "\n The average linear velocity = (%f, %f, %f) Kilometers/second", linearVelocity.x*velocityConvertion, linearVelocity.y*velocityConvertion, linearVelocity.z*velocityConvertion);
		fprintf(file, "\n The magitude of the avergae linear velocity = %f Kilometers/second", linearVelocity.w*velocityConvertion);
		
		angularMomentum = getAngularMomentum(1);
		fprintf(file, "\n The angular momentum = (%e, %e, %e) Kilograms*kilometers*kilometers/second", angularMomentum.x*AngularMomentumConvertion, angularMomentum.y*AngularMomentumConvertion, angularMomentum.z*AngularMomentumConvertion);
		fprintf(file, "\n The magitude of the angular momentum = %e Kilograms*kilometers*kilometers/second", angularMomentum.w*AngularMomentumConvertion);
		
		centerOfMass = getCenterOfMass(2);
		radiusOfBody = getRadiusOfBody(2);
		massOfBody = (double)MassFe*(double)NFe2 + (double)MassSi*(double)NSi2;
		
		fprintf(file, "\n\n ***** Stats for Body2 *****");
		fprintf(file, "\n Mass =  %e Kilograms", massOfBody*massConvertion);
		fprintf(file, "\n Radius =  %f Kilometers", radiusOfBody*lengthConvertion);
		
		fprintf(file, "\n The center of mass = (%f, %f, %f) Kilometers from (0, 0, 0)", centerOfMass.x*lengthConvertion, centerOfMass.y*lengthConvertion, centerOfMass.z*lengthConvertion);
		
		linearVelocity = getAverageLinearVelocity(2);
		fprintf(file, "\n The average linear velocity = (%f, %f, %f) Kilometers/second", linearVelocity.x*velocityConvertion, linearVelocity.y*velocityConvertion, linearVelocity.z*velocityConvertion);
		fprintf(file, "\n The magitude of the avergae linear velocity = %f Kilometers/second", linearVelocity.w*velocityConvertion);
		
		angularMomentum = getAngularMomentum(2);
		fprintf(file, "\n The angular momentum = (%e, %e, %e) Kilograms*kilometers*kilometers/second", angularMomentum.x*AngularMomentumConvertion, angularMomentum.y*AngularMomentumConvertion, angularMomentum.z*AngularMomentumConvertion);
		fprintf(file, "\n The magitude of the angular momentum = %e Kilograms*kilometers*kilometers/second", angularMomentum.w*AngularMomentumConvertion);
		fprintf(file, "\n\n");
	fclose(file);
}

void recordCarryForwardParameters()
{
	FILE *file;
	
	file = fopen("./TargetImpactorInformation/targetImpactorParameters", "wb");
		fprintf(file, "\n UnitTime = %f", UnitTime);
		fprintf(file, "\n UnitLength = %f", UnitLength);
		fprintf(file, "\n UnitMass = %f", UnitMass);
		fprintf(file, "\n MassFe = %f", MassFe);
		fprintf(file, "\n MassSi = %f", MassSi);
		fprintf(file, "\n Diameter = %f", Diameter);
		fprintf(file, "\n KFe = %f", KFe);
		fprintf(file, "\n KSi = %f", KSi);
		fprintf(file, "\n KRFe = %f", KRFe);
		fprintf(file, "\n KRSi = %f", KRSi);
		fprintf(file, "\n SDFe = %f", SDFe);
		fprintf(file, "\n SDSi = %f", SDSi);
		fprintf(file, "\n TotalNumberOfElements = %d", TotalNumberOfElements);
		fprintf(file, "\n NFe1 = %d", NFe1);
		fprintf(file, "\n NSi1 = %d", NSi1);
		fprintf(file, "\n NFe2 = %d", NFe2);
		fprintf(file, "\n NSi2 = %d", NSi2);
	fclose(file);
}

void recordTargetImpactorInitialPosVel()
{
	FILE *file;
	
	file = fopen("./TargetImpactorInformation/targetImpactorInitialPosVel", "wb");
		fwrite(&TotalNumberOfElements, sizeof(int), 1, file);
		fwrite(Pos, sizeof(float4), TotalNumberOfElements, file);
		fwrite(Vel, sizeof(float4), TotalNumberOfElements, file);
	fclose(file);
}

void cleanUpGenerateTargetImpactor()
{
	free(Pos);
	free(Vel);
	free(Force);
	
	cudaFree(Pos_DEV0);
	cudaFree(Vel_DEV0);
	cudaFree(Force_DEV0);
}
