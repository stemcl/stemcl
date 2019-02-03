// $BACKGROUND$

// FEM segmented detector analysis tool (2018)

// written by S. Hilke 
// last edit: 28.01.2019

// Furthermore the author recommend to use the ---display tools--- by D.Mitchell script additional to this script
// to easy handle many images.




// -----------------------------------------------------------------Analysis-----------------------------------------------------------
Result("\n\n|------------------------------------------------------------------------------------------------------------------------------|")
Result("\n"+"|Segmented Detector Analysis - Tool v0.01--------------------------------------------------------------------------------------|")
Result("\n|by S. Hilke 2018   -----------------------------------------------------------------------------------------------------------|")
Result("\n|This script NEEDS ---PASAD tools (Version 2.0)--- by C.Gammer for azimuthal integration.--------------------------------------|")
Result("\n|------------------------------------------------------------------------------------------------------------------------------|")
Result("\n\n")





//------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------
//-------------------------------class with button routines---------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------


class CreateButtonDialog : uiframe
{


//-----------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------
// --------------- FEM Simulation Analysis using segmented ring detector STEM images simulated by STEMcl ----------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------


void built_stack_real_space(object self)
{

	try
		{
		/*
		================================================================================

		reads images, like tiff-files, and saves them in dm-file
		Vitalij Hieronymus-Schmidt
		21.12.2016

		modified 18.08.2017, Sven Hilke
		Last Changes: 08.02.2018, Sven Hilke

		================================================================================
		Optimized script to convert real space STEM images like used for segmented ring
		detector STEM images [1] utilized with STEMcl [2] into a Real-Space-STEM image 
		stack.

		[1] Hilke et al.; Ultramicroscopy (2018), to be published.
		[2] Radek et al.; Ultramicroscopy 188 (2018) 24-30.
		================================================================================

		*/
		Result("\nScript by Sven Hilke:\n")
		/*							MAIN ROUTINE							*/
		// opendialog: pathname conatins all necessary information 
		String pathname, directory, filename
		number dimx, dimy, scanx, scany, calx, caly
		// Either open a dialog or abort
		if (!OpenDialog(pathname))	exit(0)
		Result("\nStack builder for segmented ring detector STEM images.\n")
		/*
		================================================================================
		Ask for the OUTPUT PIXEL Number of scan size area as used in STEMcl.
		================================================================================
		*/
		Get2DSize(OpenImage(pathname), dimx, dimy)
		GetNumber("Please verify the size in dimension x:", dimx, dimx)
		GetNumber("Please verify the size in dimension y:", dimy, dimy)

		/*
		================================================================================
		Ask for the SCAN SIZE AREA as used in STEMcl. Preset 10x10 nm scan area.
		================================================================================
		*/
		scanx = 10
		scany = 10
		GetNumber("Please verify the scan area [nm] in dimension x:", scanx, scanx)
		GetNumber("Please verify the scan area [nm] in dimension y:", scany, scany)

		// real space scale is calculated
		calx = scanx/dimx
		caly = scany/dimy

		directory = PathExtractDirectory(pathname, 0)
		filename = PathExtractBaseName(pathname, 0)

		/*
		================================================================================
		Reading folder information (TagNumber: 0=directories, 1=files , 2=both)
		================================================================================
		*/ 
		TagGroup FilesinDir = GetFilesInDirectory(directory ,1)
		number NoFiles = FilesinDir.TagGroupCountTags()
		GetNumber("Found " + NoFiles + " files in dir. How many files should be read?", NoFiles, NoFiles)

		// create an empty image to save the stack in
		image newStack = RealImage("Stack of images", 4, dimx, dimy, NoFiles)

		/*
		================================================================================
		input file format and stack building
		================================================================================
		*/ 
		string FileExtension = "tiff"
		GetString("Please enter the externsion for the files to load:", FileExtension, FileExtension)

		number prozent

		for (number i = 0; i < NoFiles; i++) {
			TagGroup currentFile
			// get tag from file
			FilesinDir.TagGroupGetIndexedTagAsTagGroup(i, currentFile)
			// Progress
			prozent = (i+1)/NoFiles * 100
			Result("Progress: " + prozent + " %\n")
			String currentFileName
			currentFile.TagGroupGetTagAsString("Name", currentFileName)		
			if (pathextractextension(currentFileName, 0) == FileExtension) {		
				ImageSetDimensionCalibration(newStack, 0, 0, calx, "nm", 1)
				ImageSetDimensionCalibration(newStack, 1, 0, caly, "nm", 1)
				newStack.slice2(0, 0, i, 0, dimx, 1, 1, dimy, 1) = OpenImage(directory + currentFileName)
			}
		}
		Result("Dimension of STEM images: " + dimx + "x" + dimy +" pixels.\n")
		Result("Scan area: " + scanx + "x" + scany +" nm.\n")
		Result("Calibration factors: " + calx + " in x and " + caly +" in y [nm/px].\n")

		/*
		================================================================================
		Gives the created stack a name and save.
		================================================================================
		*/ 
		string filename2 = "Real-space-stack-of-segmented-STEM"
		GetString("How should the new file be named?", filename2, filename2)

		String newdir = directory + "Results_segmented-detector-Analysis/"
		if (!DoesDirectoryExist(newdir)) CreateDirectory(newdir)
		SaveAsGatan(newStack, newdir + filename2)

		Result("Conversion ready!\n\n\n")

		}
	catch
		{
		OkDialog("Something went wrong!")
		}
	
	return
}


void built_stack_FFT(object self)
{

	try
		{

		/* 

		================================================================================

		reads images, like tiff-files, and saves them in dm-file
		Vitalij Hieronymus-Schmidt
		21.12.2016

		modified 18.08.2017, Sven Hilke
		Last Changes: 08.02.2018, Sven Hilke

		================================================================================
		Optimized script to convert single real space STEM images like used for 
		segmented ring detector STEM images [1] utilized with STEMcl [2] into 
		a FFT-STEM-image stack.

		[1] Hilke et al.; Ultramicroscopy (2018), to be published.
		[2] Radek et al.; Ultramicroscopy 188 (2018) 24-30.
		================================================================================

		*/
		Result("\nScript by Sven Hilke:\n")
		/*							MAIN ROUTINE							*/
		// opendialog: pathname conatins all necessary information 
		String pathname, directory, filename
		number dimx, dimy, scanx, scany, calx, caly
		// Either open a dialog or abort
		if (!OpenDialog(pathname))	exit(0)
		Result("\nStack builder for segmented ring detector FFT-STEM images.\n")

		/*
		================================================================================
		Ask for the OUTPUT PIXEL NUMBER of scan size area as used in STEMcl.
		================================================================================
		*/
		Get2DSize(OpenImage(pathname), dimx, dimy)
		GetNumber("Please verify the size in dimension x:", dimx, dimx)
		GetNumber("Please verify the size in dimension y:", dimy, dimy)

		/*
		================================================================================
		Ask for the SCAN SIZE AREA as used in STEMcl. Preset 10x10 nm scan area.
		================================================================================
		*/
		scanx = 10
		scany = 10
		GetNumber("Please verify the scan area [nm] in dimension x:", scanx, scanx)
		GetNumber("Please verify the scan area [nm] in dimension y:", scany, scany)

		// real space scale is calculated
		calx = (scanx/dimx)
		caly = (scany/dimy)


		// reciprocal scale is calculated
		number scalex, scaley 
		scalex = 1/(calx*dimx)
		scaley = 1/(caly*dimy)

		directory = PathExtractDirectory(pathname, 0)
		filename = PathExtractBaseName(pathname, 0)

		/*
		================================================================================
		Reading folder information (TagNumber: 0=directories, 1=files , 2=both)
		================================================================================
		*/ 
		TagGroup FilesinDir = GetFilesInDirectory(directory ,1)
		number NoFiles = FilesinDir.TagGroupCountTags()
		GetNumber("Found " + NoFiles + " files in dir. How many files should be read?", NoFiles, NoFiles)

		// creates an empty image to save the stack in
		image newStack = RealImage("Stack of images", 4, dimx, dimy, NoFiles)

		// create 2 empty complex images for the FFT calculation
		image temp = RealImage("temp", 4, dimx, dimy)
		image FFTtemp = RealImage("FFTtemp", 4, dimx, dimy)
		
		/*
		================================================================================
		asks for the FFT type you like to use
		================================================================================
		*/
		number p
		!getnumber("ComplexFFT = 0, ModulusFFT = 1, Modulus with Hanning Window = 2:", 1, p)

		
		
		/*
		================================================================================
		create Hanning window: [taken from: Diffractogram_v1.s 	version1.0b0 	October/20/97]
		================================================================================
		*/
		// local variables
		image front, hannX, hannY, hanning
		number top, left, bottom, right, sizeX, sizeY, sizeXx, sizeYy 
		number ii, const1

			GetSize(OpenImage(pathname), sizeX, sizeY) 
			GetSelection(OpenImage(pathname), top, left, bottom, right )

			sizeXx = right - left 	
			sizeYy = bottom - top	
			
			OpenAndSetProgressWindow( "Calculating", "Hanning window", " ..." )
			const1 = 2 * Pi() / sizeXx 
			hannX := CreateFloatImage("", sizeXx, sizeYy) 
			hannX = 0 
			hannX[0,0,1,sizeXx] = 1 - cos( const1 * icol )
			ii = 1 
			while( ii < sizeXx )
			{	
				hannX[ii, 0, 2*ii, sizeXx] = hannX[0, 0, ii, sizeXx] 
				ii = ii * 2 
			}	
			hannY = hannX + 0; 
		RotateLeft(hannY) 
		hanning = hannX * hannY


		/*
		================================================================================
		input file format and stack building
		================================================================================
		*/ 
		string FileExtension = "tiff"
		GetString("Please enter the externsion for the files to load:", FileExtension, FileExtension)

		// each segmented ring detector STEM image is stacked after FFT
		number prozent

		for (number i = 0; i < NoFiles; i++) {
			TagGroup currentFile
			// get tag from file
			FilesinDir.TagGroupGetIndexedTagAsTagGroup(i, currentFile)
			
			String currentFileName
			currentFile.TagGroupGetTagAsString("Name", currentFileName)		
			if (pathextractextension(currentFileName, 0) == FileExtension) {
				temp = OpenImage(directory + currentFileName)
				// FFT of actual segmented detector STEM image
				if(p == 0) //normal FFT
				{
				FFTtemp = FFT(temp)
				}
				if(p == 1) // modulus FFT
				{
				FFTtemp = modulus(RealFFT(temp))
				}
				if(p == 2) // modulus FFT with Hanning
				{
				temp := temp * hanning
				FFTtemp = modulus( RealFFT( temp ) )  
				}
				// Progress
				prozent = (i+1)/NoFiles * 100
				Result("Progress: " + prozent + " %\n")
				ImageSetDimensionCalibration(newStack, 0, (dimx+1)/2, scalex, "1/nm", 1)
				ImageSetDimensionCalibration(newStack, 1, (dimy+1)/2, scaley, "1/nm", 1)
				newStack.slice2(0, 0, i, 0, dimx, 1, 1, dimy, 1) = FFTtemp
			}
		}
		Result("Dimension of STEM images: " + dimx + "x" + dimy +" pixels.\n")
		Result("Scan area: " + scanx + "x" + scany +" nm.\n")
		Result("Calibration factors: " + calx + " in x and " + caly +" in y [nm/px].\n")
		Result("Calibration factors [reciprocal]: " + scalex + " in x and " + scaley +" in y [1/nm 1/px].\n")

		/*
		================================================================================
		Gives the created stack a name and save.
		================================================================================
		*/ 
		string filename2 = "FFT-stack-of-segmented-STEM"
		GetString("How should the new file be named (Hanning or Modulus used?)?", filename2, filename2)

		String newdir = directory + "Results_segmented-detector-Analysis/"
		if (!DoesDirectoryExist(newdir)) CreateDirectory(newdir)
		SaveAsGatan(newStack, newdir + filename2)

		Result("Conversion ready!\n")

		}
	catch
		{
		OkDialog("Something went wrong!")
		}
	
	return
}



void Ka_Xi_Plot(object self)
{

	try
		{

		/* 

		================================================================================
		Analysis of stacked FFT-STEM images from segmented ring detector STEM images.
		First Built: 18.08.2017, Sven Hilke
		Last Changes: 08.02.2018, Sven Hilke

		================================================================================
		For segmented ring detector STEM images [1] utilized with STEMcl [2]: Analysis
		of FFT-STEM image stack.

		[1] Hilke et al.; Ultramicroscopy (2018), to be published.
		[2] Radek et al.; Ultramicroscopy 188 (2018) 24-30.
		================================================================================

		*/
		Result("Developed by Sven Hilke - 2017\n")
		/*							MAIN ROUTINE							*/
		// main variables
		number dimx, dimy, dimz, value, k, sum, FFTscale
		String pathname, directory, filename
		string name

		// load image and check the dimensions
		if (!OpenDialog(pathname))	exit(0)
		try
		{
			Get3DSize(OpenImage(pathname), dimx, dimy, dimz)
			Result("Size of the stack (x, y, z): (" + dimx + ", " + dimy + ", " + dimz +")\n")
		}
		catch
		{
			OKDialog("This script only works with a stack of FFT-STEM images.")
			break
		}

		directory = PathExtractDirectory(pathname, 0)
		filename = PathExtractBaseName(pathname, 0)

		// Defining the final image: 2D plot delta k_SD vs xi_STEM
		image arrayimg := RealImage("Number Detector vs k-value [Ka-Xi-Plot]", 4, dimz, ((dimx/2)-1) )
		image int := RealImage("Int", 4, dimz+1, dimx)

		/*
		================================================================================
		For each FFT-STEM image the azimutal integral is performed and is written 
		into the respective position in the final 2D plot [Ka-Xi-Plot].
		================================================================================
		*/
		number prozent

		for (number i = 0; i < dimz; i++) {
			OpenResultsWindow()
			azimuthalintegral(OpenImage(pathname)[0,0,i,dimx,dimy,i+1], int, (dimx+1)/2, (dimx+1)/2, 1)
			name = GetName(OpenImage(pathname)[0,0,i,dimx,dimy,i+1])
				for (number j = 0; j < ((dimx/2)-1); j++) {
					value = GetPixel(int, j, 0)
					SetPixel(arrayimg, i, j, value) 
			}
			sum = 0
			// Progress
			prozent = (i+1)/dimz * 100
			Result("Progress: " + prozent + " %\n")
		}

		/*
		================================================================================
		Calibration (FFT-scale was set in FFT-Built script via scan area size [nm] and
		output pixel size of real space STEM image.
		================================================================================
		*/
		// xi-scale
		FFTscale = ImageGetDimensionScale(OpenImage(pathname), 1)
		ImageSetDimensionCalibration(arrayimg, 1, 0.0, FFTscale, "1/nm", 1)
		Result("Calibration factor Xi [reciprocal]: " + FFTscale + " in [1/nm 1/px]. Maximum Xi value = " + (dimx+1)/2 + " 1/nm.\n")

		// segmentation scale delta k_SD
		number segmentation = 0.1
		GetNumber("Please verify the segmentation step-size of the detectors delta k_SD [mrad]:", segmentation, segmentation)
		ImageSetDimensionCalibration(arrayimg, 0, 0.0, segmentation, "detector delta k_SD [mrad]", 1)
		Result("Calibration factor Delta k_SD: " + segmentation + " in [mrad] segmentation step-size of the detectors. Maximum detector: " + dimz + " pixel.\n")

		/*
		================================================================================
		Ask for a name and save.
		================================================================================
		*/ 
		string filename2 = "Number-detector-Delta-k_SD-vs-xi-value-(Ka-Xi-Plot)"
		GetString("How should the new file be named (HANNING window used?)?", filename2, filename2)

		SaveAsGatan(arrayimg, directory + filename2)

		result("Analysis done!\n\n\n")

		}
	catch
		{
		OkDialog("Something went wrong!")
		}
	
	return
}




void sum_mean_var_nvar(object self)
{

	try
		{

		/* 

		================================================================================
		Summation and normalization procedure for the Ka-Xi-Plot. Close to method 3 
		presented by Daulton, Kelton, Bondi; Ultramicroscopy 110 (2010).

		First Built: 18.08.2017, Sven Hilke
		Last Changes: 27.02.2018, Sven Hilke

		================================================================================
		For segmented ring detector STEM images [1] utilized with STEMcl [2]: Analysis
		of FFT-STEM image stack.

		[1] Hilke et al.; Ultramicroscopy (2018), to be published.
		[2] Radek et al.; Ultramicroscopy 188 (2018) 24-30.
		================================================================================

		*/
		Result("\nDeveloped by Sven Hilke - 2017\n")
		/*							MAIN ROUTINE							*/
		// main variables
		number dimx, dimy, dimz, value, k, sum, sum2, sum3, mean, FFTscale
		String pathname, directory, filename

		// load image and check the dimensions
		if (!OpenDialog(pathname))	exit(0)
		Get2DSize(OpenImage(pathname), dimx, dimy)
		Result("Size of the Ka-Xi-Plot (x, y): (" + dimx + ", " + dimy + ")\n")
		
		directory = PathExtractDirectory(pathname, 0)
		filename = PathExtractBaseName(pathname, 0)

		// Defining the final images: line plots I_aziint vs Xi_STEM and I_norm vs Xi_STEM
		image arrayimgSum := RealImage("variance-profile-I_aziint(Xi_STEM)", 4, dimy-1, 1)
		image arrayimgSumNORM := RealImage("normalized-variance-profile-I_norm(Xi_STEM)", 4, dimy-1, 1)
		// defining the image to calculate the mean along the respective constant Xi_STEM
		image MEANalongConstXiSTEM := RealImage("MEANalong-const-Xi-STEM", 4, dimx, 1)
		image MEANvsXiSTEM := RealImage("MEANvsXiSTEM", 4, dimy-1, 1)

		sum = 0
		sum2 = 0
		sum3 = 0
		image img = OpenImage(pathname)

		/*
		================================================================================
		2D [Ka-Xi-Plot] is summed at constant Xi_STEM along all segmented detectors.
		================================================================================
		*/
		for (number i = 0; i < dimy-1; i++) {
			for (number j = 0; j < dimx; j++) {
					value = GetPixel(img, j, i)
					sum = sum + value
					sum2 = sum2 + value
				}
			SetPixel(arrayimgSum, i, 0, sum)
			sum = 0
		}

		/*
		================================================================================
		Calibration (FFT-scale was set in FFT-Built script via scan area size [nm] and
		output pixel size of real space STEM image and saving procedure.
		================================================================================
		*/
		FFTscale = ImageGetDimensionScale(OpenImage(pathname), 1)
		ImageSetDimensionCalibration(arrayimgSum, 0, 0.0, FFTscale, "1/nm", 1)

		string filename2 = "variance-profile-I_aziint(Xi_STEM)"
		GetString("How should the new (NOT NORMED) file be named (HANNING window used?)?", filename2, filename2)

		SaveAsGatan(arrayimgSum, directory + filename2)

		/*
		================================================================================
		Normalization via mean of variance-profile-I_aziint(Xi_STEM).
		================================================================================
		*/


		for (number i = 0; i < dimy-1; i++) {
			for (number j = 0; j < dimx; j++) {
					value = GetPixel(img, j, i)
					sum = sum + value
					SetPixel(MEANalongConstXiSTEM, j, 0, value)
				}
			//showimage(MEANalongConstXiSTEM)
			mean = mean(MEANalongConstXiSTEM)
			sum2 = sum/(mean)
			sum3 = sum/(sum2*sum2)
			SetPixel(MEANvsXiSTEM, i, 0, mean)
			SetPixel(arrayimgSumNORM, i, 0, sum3)
			
			sum = 0
			sum2 = 0
			sum3 = 0
		}
		string filename4 = "MEANvsXiSTEM"
		GetString("How should the new file be named?", filename4, filename4)
		SaveAsGatan(MEANvsXiSTEM, directory + filename4)

		/*
		================================================================================
		Calibration (FFT-scale was set in FFT-Built script via scan area size [nm] and
		output pixel size of real space STEM image and saving procedure.
		================================================================================
		*/
		ImageSetDimensionCalibration(arrayimgSumNORM, 0, 0.0, FFTscale, "1/nm", 1)
		Result("Calibration factor Xi [reciprocal]: " + FFTscale + " in [1/nm 1/px].\n")

		string filename3 = "normalized-variance-profile-I_norm(Xi_STEM)"
		GetString("How should the new (NORMED) file be named (HANNING window used?)?", filename3, filename3)

		SaveAsGatan(arrayimgSumNORM, directory + filename3)

		result("Normalization done!\n\n\n")	

		}
	catch
		{
		OkDialog("Something went wrong!")
		break
		}
	
	return
}







}

// -------------------------------------------------make buttons and dialog ------------------------------------------------




// ---------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------


taggroup MakeFEMSimulationButtons()
{

	// $BACKGROUND$

	taggroup FEMSimulation_items
	taggroup FEMSimulationbox=dlgcreatebox("FEM (Simulation) using simulated segmented ring detector STEM images [by STEMcl]", FEMSimulation_items)
	FEMSimulationbox.dlgexternalpadding(1,1)
	FEMSimulationbox.dlginternalpadding(2,2)

	// Creates the buttons

	TagGroup built_stack_real_spaceButton = DLGCreatePushButton("Built real space stack", "built_stack_real_space")
	built_stack_real_spaceButton.dlgexternalpadding(1,1)
	FEMSimulation_items.dlgaddelement(built_stack_real_spaceButton)

	TagGroup built_stack_FFTButton = DLGCreatePushButton("Built FFT stack", "built_stack_FFT")
	built_stack_FFTButton.dlgexternalpadding(1,1)
	FEMSimulation_items.dlgaddelement(built_stack_FFTButton)

	TagGroup Ka_Xi_PlotButton = DLGCreatePushButton("Analysis of FFT-STEM image stack [Ka-Xi-Plot]", "Ka_Xi_Plot")
	Ka_Xi_PlotButton.dlgexternalpadding(1,1)
	FEMSimulation_items.dlgaddelement(Ka_Xi_PlotButton)

	TagGroup sum_mean_var_nvarButton = DLGCreatePushButton("Calculate the normalized variance from the Ka-Xi-Plot [Var-Plot, Mean-Plot, nVar-Plot]", "sum_mean_var_nvar")
	sum_mean_var_nvarButton.dlgexternalpadding(1,1)
	FEMSimulation_items.dlgaddelement(sum_mean_var_nvarButton)

	return FEMSimulationbox
}



// -------------------------------------------------------------------------------------------------

void CreateDialog()
{
	// $BACKGROUND$


	TagGroup position;
	position = DLGBuildPositionFromApplication()
	position.TagGroupSetTagAsTagGroup( "Width", DLGBuildAutoSize() )
	position.TagGroupSetTagAsTagGroup( "Height", DLGBuildAutoSize() )
	position.TagGroupSetTagAsTagGroup( "X", DLGBuildRelativePosition( "Inside", 1 ) )
	position.TagGroupSetTagAsTagGroup( "Y", DLGBuildRelativePosition( "Inside", -1 ) )
	TagGroup dialog_items;
	TagGroup dialog = DLGCreateDialog("FEM Segmented Detector Analysis Tools v0.1", dialog_items).dlgposition(position);

	dialog_items.dlgaddelement( MakeFEMSimulationButtons() );

	
	
	object dialog_frame = alloc(CreateButtonDialog).init(dialog)
	dialog_frame.display("FEM Segmented Detector Analysis Tools v0.1");
}

// $BACKGROUND$
createdialog()
