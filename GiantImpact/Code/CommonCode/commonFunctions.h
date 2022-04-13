// Prototypes of all the functions in this file.
void cudaErrorCheck(const char);
void copyPosVelToGPU();
void copyPosVelFromGPU();
void Display();
void idle();
void reshape(int, int);
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

void KeyPressed(unsigned char key, int x, int y)
{
	float dx = 0.2;
	float dy = 0.2;
	float dz = 0.2;
	float angle = 0.4;
	
	if(key == 'h')
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
	}
	if(key == 'q')
	{
		if(MovieOn == 1) 
		{
			pclose(MovieFile);
			free(MovieBuffer);
			MovieOn = 0;
		}
		glutDestroyWindow(Window);
		printf("\nw Good Bye\n");
		exit(0);
	}
	if(key == 'p')
	{
		Pause = 1;
	}
	if(key == 'g')
	{
		if(Done != 1) Pause = 0;
	}
	if(key == 't')
	{
		TranslateRotate = 1;
	}
	if(key == 'r')
	{
		TranslateRotate = 0;
	}
	if(key == 'f')
	{
		ForwardBackward = 1;
	}
	if(key == 'b')
	{
		ForwardBackward = 0;
	}
	if(key == '!')
	{
		DrawType = 1;
		drawStuff();
	}
	if(key == '@')
	{
		DrawType = 2;
		drawStuff();
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
		// Setting up the movie buffer.
		const char* cmd = "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
		              "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip video.mp4";
		MovieFile = popen(cmd, "w");
		//Buffer = new int[XWindowSize*YWindowSize];
		MovieBuffer = (int*)malloc(XWindowSize*YWindowSize*sizeof(int));
		MovieOn = 1;
	}
	if(key == 'M')
	{
		if(MovieOn == 1) 
		{
			pclose(MovieFile);
			free(MovieBuffer);
			MovieOn = 0;
		}
	}
	
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
		returnStatus = system("ffmpeg -i output1.mp4 screenShot.jpeg");
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
		//ffmpeg -i output1.mp4 output_%03d.jpeg
	}
}
