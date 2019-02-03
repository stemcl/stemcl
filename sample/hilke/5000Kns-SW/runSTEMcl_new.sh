#!/bin/bash


probe_size=4096
output_size=256

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64


		stemcl nvidia 0 gpu_Titan_X.bin assets/parameter.dat > Titan_X.log # & #> /dev/null 2>&1 &
		#stemcl nvidia 1 gpu_GTX_580.bin assets/parameter.dat > GTX_580.log 

		sleep 90s

		# convert binary to tiff
		# pixel:x, pixel:y, file extension, num. detectors, merge?
		stemcl2tiff $output_size $output_size .bin 150 merged
		
		# save results and folder management
		folder=results
		mkdir -p $folder/detectors/raw-data
		mkdir -p $folder/detectors/tiff
		#mkdir -p $folder/diffraction-pattern/raw-data
		#mkdir -p $folder/diffraction-pattern/tiff
		mkdir -p $folder/parameter

		# move and copy paste input parameter-files as well as raw and merged tiff
		mv *.bin $folder/detectors/raw-data
		mv *.tiff $folder/detectors/tiff
		mv *.pgm $folder/parameter
		mv *.log $folder/parameter
		cp assets/*.dat $folder/parameter
		cp assets/*.xyz $folder/parameter

		# diffraction pattern files
		#cp stemcl2tiff diffraction_pattern
		#(cd diffraction_pattern && stemcl2tiff $probe_size $probe_size .bin 1)
		#mv diffraction_pattern/*.bin $folder/diffraction-pattern/raw-data
		#mv diffraction_pattern/*.tiff $folder/diffraction-pattern/tiff
		#rm -rf diffraction_pattern

		# rename parameter.dat-file for next simulation
		# mv assets/$filename assets/parameter.dat

