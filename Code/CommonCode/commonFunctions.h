// Prototypes of all the functions in this file.
void cudaErrorCheck(const char);
void copyPosVelToGPU();
void copyPosVelFromGPU();
void Display();
void idle();
void reshape(int, int);
void terminalPrint();
string getTimeStamp();
void KeyPressed(unsigned char, int, int);

void cudaErrorCheck(const char *message)
{
  cudaError_t  error;
  error = cudaGetLastError();

  if(error != cudaSuccess)
  {
    printf("\n CUDA ERROR: %s = %s\n", message, cudaGetErrorString(error));
    exit(0);
  }
}

void copyPosVelToGPU()
{
	cudaMemcpy( Pos_DEV0, Pos, TotalNumberOfElements *sizeof(float4), cudaMemcpyHostToDevice );
	cudaErrorCheck("cudaMemcpy Pos3");
	cudaMemcpy( Vel_DEV0, Vel, TotalNumberOfElements *sizeof(float4), cudaMemcpyHostToDevice );
	cudaErrorCheck("cudaMemcpy Vel");
}

void copyPosVelFromGPU()
{
	cudaMemcpy( Pos, Pos_DEV0, TotalNumberOfElements *sizeof(float4), cudaMemcpyDeviceToHost );
	cudaErrorCheck("cudaMemcpy Pos1");
	cudaMemcpy( Vel, Vel_DEV0, TotalNumberOfElements *sizeof(float4), cudaMemcpyDeviceToHost );
	cudaErrorCheck("cudaMemcpy Vel");
}

void idle()
{
	doStuff();
}

void Display()
{
	drawStuff();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);
}

// If calling from generator, call with 0, if calling from viewer use 1
void terminalPrint()
{
	int result = system("clear");
	printf("\n Time = %f days\n\n", RunTime*UnitTime/(24.0*60.0*60.0));
	printf(" In the system +x is to the right, +y is up, and +z is towards you.\n\n");
	
	printf(" (q) Quit the program\n");

	printf(" (g) Run/Pause toggle............");
	if(Pause == 1) printf("The simulation is paused.\n\n");
	else printf("The simulation is running.\n\n");
	
	printf(" (t) Set the graphics to translate\n");
	printf(" (r) Set the graphics to rotate\n");
	printf(" Use x/y/z to rotate/translate about the x/y/z-axis\n");
	if (TranslateRotate == 1) printf(" Currently in translate mode\n\n");
	else printf(" Currently in rotation mode\n\n");
	
	printf(" (1) Draw elements as points\n");
	printf(" (2) Draw elements as spheres (less performance than points)\n");
	printf(" (@) Draws what the program thinks is the Earth and the Moon\n\n");
	
	if (MovieOn) printf(" (m) Stop recording movie\t\t(Movie recording in progress)\n");
	else printf(" (m) Start recording movie\n");	
	printf(" (s) Take a screenshot\n\n");

	if (ViewingImpact)
	{
		printf(" (f/b) View run forward/backward............");
		//printf(" (f) View run forward\n");
		if (ForwardBackward == 1) printf(" Currently running forward\n");
		else printf(" Currently running backward\n");
	}
	printf("\n");
}

string getTimeStamp()
{
	// Want to get a time stamp string representing current date/time, so we have a
	// unique name for each video/screenshot taken.
	time_t t = time(0); 
	struct tm * now = localtime( & t );
	int month = now->tm_mon + 1, day = now->tm_mday, year = now->tm_year, 
				curTimeHour = now->tm_hour, curTimeMin = now->tm_min, curTimeSec = now->tm_sec;
	stringstream smonth, sday, syear, stimeHour, stimeMin, stimeSec;
	smonth << month;
	sday << day;
	syear << (year + 1900); // The computer starts counting from the year 1900, so 1900 is year 0. So we fix that.
	stimeHour << curTimeHour;
	stimeMin << curTimeMin;
	stimeSec << curTimeSec;
	string timeStamp;

	if (curTimeMin <= 9)	
		timeStamp = smonth.str() + "-" + sday.str() + "-" + syear.str() + '_' + stimeHour.str() + ":0" + stimeMin.str() + 
					":" + stimeSec.str();
	else			
		timeStamp = smonth.str() + "-" + sday.str() + '-' + syear.str() + "_" + stimeHour.str() + ":" + stimeMin.str() +
					":" + stimeSec.str();
	return timeStamp;
}

void KeyPressed(unsigned char key, int x, int y)
{
	float dx = 200.0/UnitLength;
	float dy = 200.0/UnitLength;
	float dz = 200.0/UnitLength;
	float angle = 0.4;
	
	/*if(key == 'h')
	{
		printf("\n\n");
		printf("\n In the system positive x is to the right, positive y is up, and positive z is towards you.");
		printf("\n h -> help");
		printf("\n q -> quits the program");
		printf("\n g -> go or run the program");
		printf("\n p -> pause the program");
		printf("\n t -> sets the graphics to translate");
		printf("\n r -> sets the graphics to rotate");
		printf("\n x -> translate or rotate about the x-axis");
		printf("\n y -> translate or rotate about the y-axis");
		printf("\n z -> translate or rotate about the z-axis");
		printf("\n 1 -> draw elements as points");
		printf("\n 2 -> draw elements as spheres (not available while generating bodies)");
		printf("\n ! -> draw All elements");
		printf("\n @ -> draw the program thinks is the Earth and the Moon (not available while generating bodies)");
		printf("\n m -> start recording a movie");
		printf("\n M -> stop recording a movie");
		printf("\n s -> take a screenshot");
		printf("\n b -> view run backward (only available in view program)");
		printf("\n f -> view run forward (only available in view program)");
		printf("\n\n");
	}*/
	if(key == 'q')
	{
		if(MovieOn == 1) 
		{
			pclose(MovieFile);
			free(MovieBuffer);
			MovieOn = 0;
		}
		glutDestroyWindow(Window);
		printf("\n Exiting program...Good Bye\n");
		exit(0);
	}
	if(key == 'g')
	{
		if (Pause == 0)
			Pause = 1;
		else
			Pause = 0;
		terminalPrint();
	}
	/*if(key == 'g')
	{
		if(Done != 1) Pause = 0;
	}*/
	if(key == 't') // We are translating.
	{
		TranslateRotate = 1;
		terminalPrint();
	}
	if(key == 'r') // We are rotating.
	{
		TranslateRotate = 0;
		terminalPrint();
	}
	if(key == 'f')
	{
		ForwardBackward = 1;
		terminalPrint();
	}
	if(key == 'b')
	{
		ForwardBackward = 0;
		terminalPrint();
	}
	/*if(key == '!')
	{
		DrawType = 1;
		drawStuff();
	}*/
	if(key == '@')
	{
		DrawType = 2;
		drawStuff();
		Pause = 1;
		DrawType = 1;
		terminalPrint();
	}
	if(key == '1')
	{
		DrawQuality = 1;
		drawStuff();
	}
	if(key == '2')
	{
		DrawQuality = 2;
		drawStuff();
	}
	
	if(key == 'x')
	{
		if(TranslateRotate == 1) 
		{
			glTranslatef(dx, 0.0, 0.0);
		}
		else 
		{
			glRotatef(angle, 1.0, 0.0, 0.0);
		}
		drawStuff();
	}
	if(key == 'X')
	{
		if(TranslateRotate == 1) 
		{
			glTranslatef(-dx, 0.0, 0.0);
		}
		else 
		{
			glRotatef(-angle, 1.0, 0.0, 0.0);
		}
		drawStuff();
	}
	if(key == 'y')
	{
		if(TranslateRotate == 1) 
		{
			glTranslatef(0.0, dy, 0.0);
		}
		else 
		{
			glRotatef(angle, 0.0, 1.0, 0.0);
		}
		
		drawStuff();
	}
	if(key == 'Y')
	{
		if(TranslateRotate == 1) 
		{
			glTranslatef(0.0, -dy, 0.0);
		}
		else 
		{
			glRotatef(-angle, 0.0, 1.0, 0.0);
		}
		drawStuff();
	}
	if(key == 'z')
	{
		if(TranslateRotate == 1) 
		{
			glTranslatef(0.0, 0.0, dz);
		}
		else 
		{
			glRotatef(angle, 0.0, 0.0, 1.0);
		}
		drawStuff();
	}
	if(key == 'Z')
	{
		if(TranslateRotate == 1) 
		{
			glTranslatef(0.0, 0.0, -dz);
		}
		else 
		{
			glRotatef(-angle, 0.0, 0.0, 1.0);
		}
		drawStuff();
	}
	if(key == 'm')
	{
		if (MovieOn == 0)
		{
		
		
			/*time_t t = time(0); 
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
			string targetImpactorFolderName = "" + monthday;
			const char *ccx = .c_str();*/
			
			// const char* cmd = "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
				      // "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip video.mp4";
			
			string ts = getTimeStamp();
			ts.append(".mp4");
		
			// Setting up the movie buffer.
			string ffmpegCommand = "ffmpeg -loglevel quiet -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
				      "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip ";
				      
			string z = ffmpegCommand + ts;
			const char *cmd = z.c_str();
				      
			MovieFile = popen(cmd, "w");
			//Buffer = new int[XWindowSize*YWindowSize];
			MovieBuffer = (int*)malloc(XWindowSize*YWindowSize*sizeof(int));
			MovieOn = 1;
			terminalPrint();
		}
		else
		{
			pclose(MovieFile);
			free(MovieBuffer);
			MovieOn = 0;
			terminalPrint();
			printf("----Movie finished recording----\n");
		}
	}
	/*if(key == 'M')
	{
		if(MovieOn == 1) 
		{
			pclose(MovieFile);
			free(MovieBuffer);
			MovieOn = 0;
		}
	}*/
	
	if(key == 's')
	{	
		int returnStatus;
		int pauseFlag;
		FILE* ScreenShotFile;
		int* buffer;
		const char* cmd = "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
		              "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output1.mp4";
		ScreenShotFile = popen(cmd, "w");
		buffer = (int*)malloc(XWindowSize*YWindowSize*sizeof(int));
		
		if(Pause == 0) 
		{
			Pause = 1;
			pauseFlag = 0;
		}
		else
		{
			pauseFlag = 1;
		}
		
		for(int i =0; i < 1; i++)
		{
			drawStuff();
			glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
			fwrite(buffer, sizeof(int)*XWindowSize*YWindowSize, 1, ScreenShotFile);
		}
		
		pclose(ScreenShotFile);
		free(buffer);
		string ts = getTimeStamp();
		ts.append(".jpeg");
		// returnStatus = system("ffmpeg -i output1.mp4 screenShot.jpeg");
		string s = "ffmpeg -loglevel quiet -i output1.mp4 screenShot.jpeg" + ts;
		cmd = s.c_str();
		returnStatus = system(cmd);
		if(returnStatus != 0) 
		{
			printf("\n TSU Error: ffmpeg call \n");
			exit(0);
		}
		returnStatus = system("rm output1.mp4");
		if(returnStatus != 0) 
		{
			printf("\n TSU Error: removing old mp4 \n");
			exit(0);
		}
		
		Pause = pauseFlag;
		terminalPrint();
		printf("----Screenshot captured----\n");
		//ffmpeg -i output1.mp4 output_%03d.jpeg
	}
}
