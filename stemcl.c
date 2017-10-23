//
//  main.c
//  stemcl
//
//  Created by Manuel Radek on 21.08.16.
//  Copyright © 2016 Manuel Radek. All rights reserved.
//

#include <clFFT.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>

#include "stemcl.h"
#include "helpers.h"
#include "kernels.h"

int num_parallel = 16;
int GPU_PROBE_CALCULATION = 1;
int SMOOTH_PROBE = 1;
int use_hdd_mem = 0;

const char version[] = "13-jun-2017";


/* global specimen and probe parameters to share */
const float bandwidth = (2.0F/3.0F);   // antialiasing bandwidth limit factor

// for the atomic scattering factor tables
const int NPMAX = 12;       // number of parameters for each Z
const int NZMIN = 1;        // min Z in featom.tab
const int NZMAX = 103;      // max Z in featom.tab

int feTableRead = 0;        // flag to remember if the param file has been read
double **fparams;           // to get fe(k) parameters
const int nl = 3, ng = 3;   // number of Lorenzians and Gaussians
cl_device_id *devices;
cl_context context;
cl_command_queue command_queue;
cl_command_queue copy_queue;
clfftPlanHandle plan, probe_plan;

cl_kernel kernel_propagate, kernel_transmit, kernel_sum, kernel_prepare_probe, kernel_scale_probe;
cl_kernel kernel_gauss_multiply, kernel_prepare_probe_espread;
cl_mem kxp2_buffer, kyp2_buffer, k2min_buffer, k2max_buffer, temp_buffer;

int n_slices, n_detect, probe_nx, probe_ny, nx, ny;
float vol_dim_x, vol_dim_y, z_total, scale, k2maxp;
float *det_min, *det_max, *p_spat_freqx, *p_spat_freqy, *spat_freqx, *spat_freqy;
float *p_spat_freqx2, *p_spat_freqy2, *spat_freqx2, *spat_freqy2;
cl_float *k2min, *k2max;
float **propxr, **propxi, **propyr, **propyi;
FILE *output_file;

cl_float2 **probes;
cl_float2 *transmissions;

bool verbose = true;

typedef struct {
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t num_detectors;
    int32_t *progress;
    cl_float *data;
} output_image_t;

typedef struct {
    uint32_t x;
    uint32_t y;
    uint32_t detector;
    float value;
} located_intensity;

#pragma mark - main
int main(int argc, const char * argv[]) {
    if (argc < 4) {
        if (argc > 1 && strstr(argv[1], "who")) {
            fprintf(stderr, "%s\nstemcl was originally built by Manuel Radek and Jan-Gerd Tenberge\n%s", KGRN, KNRM);
        } else {
            fprintf(stderr, "Usage: %s <OpenCL platform name> <OpenCL device id> <simulation directory>\n", argv[0]);
            return EXIT_SUCCESS;
        }

        return EXIT_FAILURE;
    }
	
    // print version info
    printf("openCL STEM simulation (stemCL 1.0)\n(%s)\n\n", version);
    
    int device_number = 0;
    const char *platform_name = "";
    platform_name = argv[1];
    device_number = atoi(argv[2]);
    chdir(argv[3]);
	
	char *output_file_name = malloc(1024);
	sprintf(output_file_name, "output_%s_%d.stemcl", platform_name, device_number);
    output_file = fopen(output_file_name, "ab+");
    free(output_file_name);
	
    // read parameter file
    SimDetails sim_details;
    int success = readParameterFile(&sim_details);
    if (success != 0) {
        print_verbose("%s[ERROR] Error in Parameter file. Exit!%s\n", KRED, KNRM);
        return EXIT_FAILURE;
    }
    
    probes = (cl_float2**)malloc(num_parallel*sizeof(cl_float2*));
    
    int fd = open("progress.pgm", O_RDWR|O_CREAT, 0777);
    size_t file_size = sim_details.nxout * sim_details.nyout * sizeof(int32_t) + 200;
    ftruncate(fd, file_size);
    void *progress_data = mmap(NULL, file_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_FILE, fd, 0);

    if (((char *)progress_data)[0] != 'P') {
        sprintf(progress_data, "P5\n%5d %5d\n255\n", sim_details.nxout, sim_details.nyout);
    }

    uint8_t *progress_image = progress_data + 19;

    // initialize openCL
    cl_device_id device_id = get_device(CL_DEVICE_TYPE_ALL, platform_name, device_number);
    initOpenCL(device_id);
    
    int *zNum;
    float *x_pos;
    float *y_pos;
    float *z_pos;
    
    int natoms = 0;
    
    fprintf(stdout, "\n");
    print_verbose("Specimen details\n");
    
    n_slices = readXYZfile(sim_details.inputFilename, &zNum, &x_pos, &y_pos, &z_pos, sim_details.sliceThickness, &natoms, &vol_dim_x, &vol_dim_y, &sim_details);
    
    if (n_slices < 0) {
        print_verbose("%s[ERROR] Error in xyz-file. Exit!%s\n", KRED, KNRM);
        return EXIT_FAILURE;
    }
    
    for(int islice = 0; islice<n_slices; islice++) {
        z_total +=  sim_details.sliceThickness;
    }

    print_verbose("├─ Input data file: %s\n", sim_details.inputFilename);    
    print_verbose("├─ Number of atoms: %d\n", natoms);
    print_verbose("├─ Dimensions: %.3fÅ x %.3fÅ\n", vol_dim_x, vol_dim_y);
    print_verbose("├─ Number of slices: %d\n", n_slices);
    print_verbose("├─ Slice thickness: %.3fÅ\n", sim_details.sliceThickness);
    print_verbose("└─ Thickness after slicing: %.3fÅ\n", z_total);
    
    fprintf(stdout, "\n");
    print_verbose("Simulation\n");
    print_verbose("├─ Progress file: %s\n", "progress.pgm");
    print_verbose("├─ Using GPU probe calculation: %s\n", (GPU_PROBE_CALCULATION > 0) ? "yes" : "no");
    print_verbose("├─ Using probe smoothing: %s\n", (SMOOTH_PROBE > 0) ? "yes" : "no");
    print_verbose("├─ Parallel probes: %d\n", num_parallel);
    print_verbose("├─ Save transmissions on HDD: %s\n", (use_hdd_mem > 0) ? "yes" : "no");
    print_verbose("│\n");
    print_verbose("├─ Electron wavelength: %.8fÅ\n", sim_details.wavlen);
    if (sim_details.dE_E > 0.0)	{
		print_verbose("├─ Using energy spread calculation\n", sim_details.wavlen);
	}   
    print_verbose("├─ Transmission size: %dpx x %dpx\n", nx, ny);
    print_verbose("├─ Probe size: %dpx x %dpx\n", probe_nx, probe_ny);
    print_verbose("├─ Output size: %dpx x %dpx\n", sim_details.nxout, sim_details.nyout);
    print_verbose("├─ Number of detectors: %d\n", n_detect-1);
    const char *color = (sim_details.neDiffPattern > 0) ? KGRN : KNRM;
    print_verbose("├─ Calculate diffraction pattern: %s%s%s\n", color, (sim_details.neDiffPattern > 0) ? "yes" : "no", KNRM);
    
    // No EDX support at the moment, will come in a future update
    //color = (sim_details.calculate_edx > 0) ? KGRN : KNRM;
    //print_verbose("├─ Calculate EDX signal: %s%s%s\n", color, (sim_details.calculate_edx > 0) ? "yes" : "no", KNRM);
    
    print_verbose("│\n");
    
    float *tmpx, *tmpy;
    
    double pi = 4.0 * atan( 1.0 );
    
    // precalculate spatial frequencies
    p_spat_freqx  = (float*) malloc(probe_nx * sizeof(float));
    p_spat_freqy  = (float*) malloc(probe_ny * sizeof(float));
    p_spat_freqx2  = (float*) malloc(probe_nx * sizeof(float));
    p_spat_freqy2  = (float*) malloc(probe_ny * sizeof(float));
    tmpx  = (float*) malloc(nx * sizeof(float));
    tmpy  = (float*) malloc(ny * sizeof(float));
    
    frequency(p_spat_freqx, p_spat_freqx2, tmpx, probe_nx, vol_dim_x*((double)probe_nx)/nx);
    frequency(p_spat_freqy, p_spat_freqy2, tmpy, probe_ny, vol_dim_y*((double)probe_ny)/ny);
    
    spat_freqx  = (float*) malloc(nx * sizeof(float));
    spat_freqy  = (float*) malloc(ny * sizeof(float));
    spat_freqx2 = (float*) malloc(nx * sizeof(float));
    spat_freqy2 = (float*) malloc(ny * sizeof(float));
    
    frequency(spat_freqx, spat_freqx2, tmpx, nx, vol_dim_x);
    frequency(spat_freqy, spat_freqy2, tmpy, ny, vol_dim_y);
    
    free(tmpx);
    free(tmpy);

    double sum = ((double)nx)/(2.0*vol_dim_x);
    k2maxp = ((double)ny)/(2.0*vol_dim_y);
    if( sum < k2maxp ) k2maxp = sum;
    k2maxp= bandwidth * k2maxp;
    k2maxp = k2maxp * k2maxp;
    
    cl_mem kxp_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * probe_nx, p_spat_freqx, &success);
    cl_mem kyp_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * probe_ny, p_spat_freqy, &success);
    cl_mem xp_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * nx, spat_freqx, &success);
    cl_mem yp_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * ny, spat_freqy, &success);
    
    scale = sim_details.wavlen * sim_details.mm0;
    size_t transmission_bytes = sizeof(cl_float2) * n_slices * nx * ny;
    char *transmission_size_string = format_size(transmission_bytes);
    print_verbose("├─ Precalculating transmission functions (%s)\n", transmission_size_string);
    free(transmission_size_string);
    
    if (use_hdd_mem == 0) {
		transmissions = (cl_float2 *)malloc(sizeof(cl_float2) * n_slices * nx * ny);
	}
	    
    cl_mem trans_buffer[2];
    trans_buffer[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float2) * nx * ny, NULL, &success);
    trans_buffer[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float2) * nx * ny, NULL, &success);
    
    int start = 0;
    for (int i = 0; i < n_slices; i++) {
        
        int na = 0;
        for (int a = start; a < natoms; a++) {
            if (z_pos[a] < (i + 1) * sim_details.sliceThickness) {
                na++;
            } else {
                break;
            }
        }
        calculateTransmissionLayer(&x_pos[start], &y_pos[start], &zNum[start], na, vol_dim_x, vol_dim_y, sim_details.v0, i, nx, ny, k2maxp, trans_buffer[0]);
        start += na;
		float percent = (float)i/n_slices * 100.0;
		print_verbose("├─ %3.0f%% Slice %d\r", percent, i);

        fflush(stdout);
    }
    print_verbose("├─ Precalculation complete\n");
    
    
    propxr = (float**) malloc(n_slices * probe_nx * sizeof(float*));
    for (int i = 0; i < n_slices; i++)
        propxr[i] = (float *)malloc(probe_nx * sizeof(float));
    
    propxi = (float**) malloc(n_slices * probe_nx * sizeof(float*));
    for (int i = 0; i < n_slices; i++)
        propxi[i] = (float *)malloc(probe_nx * sizeof(float));
    
    propyr = (float**) malloc(n_slices * probe_ny * sizeof(float*));
    for (int i = 0; i < n_slices; i++)
        propyr[i] = (float *)malloc(probe_ny * sizeof(float));
    
    propyi = (float**) malloc(n_slices * probe_ny * sizeof(float*));
    for (int i = 0; i < n_slices; i++)
        propyi[i] = (float *)malloc(probe_ny * sizeof(float));
    
    
    double tctx = 2.0 * tan(sim_details.ctiltx);
    double tcty = 2.0 * tan(sim_details.ctilty);
    
    
    for(int islice=0; islice<n_slices; islice++) {
        scale = sim_details.sliceThickness;
        for(int ix=0; ix<probe_nx; ix++) {
            double w = pi * scale * ( p_spat_freqx2[ix] * sim_details.wavlen - p_spat_freqx[ix]*tctx );
            propxr[islice][ix]= (float)  cos(w);
            propxi[islice][ix]= (float) -sin(w);
        }
        
        for(int iy=0; iy<probe_ny; iy++) {
            double w = pi * scale * ( p_spat_freqy2[iy] * sim_details.wavlen - p_spat_freqy[iy]*tcty );
            propyr[islice][iy]= (float)  cos(w);
            propyi[islice][iy]= (float) -sin(w);
        }
        
    }
    
    /*  convert aperture dimensions */
    
    k2min = (cl_float*) malloc( n_detect * sizeof(cl_float));
    k2max = (cl_float*) malloc( n_detect * sizeof(cl_float));
    
    for(int idetect=0; idetect<n_detect; idetect++) {
        k2max[idetect] = 0.001 * det_max[idetect]/sim_details.wavlen;
        k2max[idetect] = k2max[idetect] * k2max[idetect];
        k2min[idetect] = 0.001 * det_min[idetect]/sim_details.wavlen;
        k2min[idetect] = k2min[idetect] * k2min[idetect];
    }
    
    // Copy k2min, k2max to device
    cl_int cl_error = CL_SUCCESS;
    k2min_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * n_detect, k2min, &cl_error);
    k2max_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * n_detect, k2max, &cl_error);
    temp_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float) * n_detect * probe_nx, NULL, &cl_error);
        
    /*  init the min/max record of total integrated intensity */
    
    double totmin =  10.0;
    double totmax = -10.0;
    cl_float *detect = (cl_float*) malloc( n_detect * sizeof(cl_float));
    
    
    cl_float2 *prop_x_data = (cl_float2 *)malloc(sizeof(cl_float2) * probe_nx * n_slices);
    cl_float2 *prop_y_data = (cl_float2 *)malloc(sizeof(cl_float2) * probe_ny * n_slices);
    
    for (int l = 0; l < n_slices; l++) {
        
        for (int x = 0; x < probe_nx; x++) {
            int index = l * probe_nx + x;
            prop_x_data[index].s0 = propxr[l][x];
            prop_x_data[index].s1 = propxi[l][x];
        }
        
        for (int y = 0; y < probe_ny; y++) {
            int index = l * probe_ny + y;
            prop_y_data[index].s0 = propyr[l][y];
            prop_y_data[index].s1 = propyi[l][y];
        }
        
    }
    
    // set propagator data
    // use one propagator for all slices - only valid for constant sliceThickness!
    cl_mem prop_x_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float2) * nx * n_slices, prop_x_data, &cl_error);
    cl_mem prop_y_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float2) * ny * n_slices, prop_y_data, &cl_error);
    
    kxp2_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * probe_nx, p_spat_freqx2, &cl_error);
    kyp2_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * probe_ny, p_spat_freqy2, &cl_error);
    
    clSetKernelArg(kernel_propagate, 1, sizeof(cl_mem), &prop_x_buffer);
    clSetKernelArg(kernel_propagate, 2, sizeof(cl_mem), &prop_y_buffer);
    clSetKernelArg(kernel_propagate, 3, sizeof(cl_mem), &kxp2_buffer);
    clSetKernelArg(kernel_propagate, 4, sizeof(cl_mem), &kyp2_buffer);
    
    
    // set probe kernel arguments
    if (sim_details.dE_E > 0.0)	{
		// set smooth probe kernel arguments
		clSetKernelArg(kernel_prepare_probe_espread, 10, sizeof(cl_float), &sim_details.wavlen);
		clSetKernelArg(kernel_prepare_probe_espread, 11, sizeof(cl_float), &sim_details.dfa2);
		clSetKernelArg(kernel_prepare_probe_espread, 12, sizeof(cl_float), &sim_details.dfa2phi);
		clSetKernelArg(kernel_prepare_probe_espread, 13, sizeof(cl_float), &sim_details.dfa3);
		clSetKernelArg(kernel_prepare_probe_espread, 14, sizeof(cl_float), &sim_details.dfa3phi);
		clSetKernelArg(kernel_prepare_probe_espread, 15, sizeof(cl_float), &sim_details.df);
		clSetKernelArg(kernel_prepare_probe_espread, 16, sizeof(cl_mem), &kxp_buffer);
		clSetKernelArg(kernel_prepare_probe_espread, 17, sizeof(cl_mem), &kyp_buffer);
		clSetKernelArg(kernel_prepare_probe_espread, 18, sizeof(cl_mem), &kxp2_buffer);
		clSetKernelArg(kernel_prepare_probe_espread, 19, sizeof(cl_mem), &kyp2_buffer);
		clSetKernelArg(kernel_prepare_probe_espread, 20, sizeof(cl_mem), &xp_buffer);
		clSetKernelArg(kernel_prepare_probe_espread, 21, sizeof(cl_mem), &yp_buffer);
	} else {
		clSetKernelArg(kernel_prepare_probe, 10, sizeof(cl_float), &sim_details.wavlen);
		clSetKernelArg(kernel_prepare_probe, 11, sizeof(cl_float), &sim_details.dfa2);
		clSetKernelArg(kernel_prepare_probe, 12, sizeof(cl_float), &sim_details.dfa2phi);
		clSetKernelArg(kernel_prepare_probe, 13, sizeof(cl_float), &sim_details.dfa3);
		clSetKernelArg(kernel_prepare_probe, 14, sizeof(cl_float), &sim_details.dfa3phi);
		clSetKernelArg(kernel_prepare_probe, 15, sizeof(cl_float), &sim_details.df);
		clSetKernelArg(kernel_prepare_probe, 16, sizeof(cl_mem), &kxp_buffer);
		clSetKernelArg(kernel_prepare_probe, 17, sizeof(cl_mem), &kyp_buffer);
		clSetKernelArg(kernel_prepare_probe, 18, sizeof(cl_mem), &kxp2_buffer);
		clSetKernelArg(kernel_prepare_probe, 19, sizeof(cl_mem), &kyp2_buffer);
		clSetKernelArg(kernel_prepare_probe, 20, sizeof(cl_mem), &xp_buffer);
		clSetKernelArg(kernel_prepare_probe, 21, sizeof(cl_mem), &yp_buffer);
	}
	

    /* ----- probe wavefunction ----- */
    for (int i = 0; i < num_parallel; i++) {
        probes[i] = (cl_float2 *)malloc(probe_nx * probe_ny * sizeof(cl_float2));
    }
    
    cl_mem probes_buffer[num_parallel];
    for (int i = 0; i < num_parallel; i++) {
        probes_buffer[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float2) * probe_nx * probe_ny, NULL, &success);
    }

    float ***pixr = (float***) malloc( n_detect * sizeof(float**));
    for(int i=0; i<n_detect; i++){
        pixr[i] = (float**) malloc(sim_details.nxout * sim_details.nyout * sizeof(float*));
        for (int a = 0; a < sim_details.nxout; a++) {
            pixr[i][a] = (float *)malloc(sim_details.nyout * sizeof(float));
        }
    }

    for (int idetect = 0; idetect < n_detect; idetect++) {
        for (int ix = 0; ix < sim_details.nxout; ix++) {
            for (int iy = 0; iy < sim_details.nyout; iy++) {
                pixr[idetect][ix][iy] = 0.0;
            }
        }
    }
    
    // result array for edx caluclation. 3d dimensional array, that holds the intensity for every output pixel
    // after every slice.
    // dummy allocation for future use
    float *edx_signal = (float *)malloc(sizeof(float));//(float *)malloc(sizeof(float) * sim_details.nxout * sim_details.nyout * n_slices);

    /*  iterate the multislice algorithm proper for each position of
    the focused probe */

    double dx = (sim_details.xf-sim_details.xi)/((double)(sim_details.nxout-1));
    double dy = (sim_details.yf-sim_details.yi)/((double)(sim_details.nyout-1));
    int stepCount = 0;
    cl_int2 current_points[num_parallel];
    int location_count = num_parallel;
    cl_float2 *transmission_data = NULL;
    
    while (location_count > 0) {
        location_count = 0;
        
        if (use_hdd_mem > 0) {
			transmission_data = loadTransmissionData(0);
			clEnqueueWriteBuffer(copy_queue, trans_buffer[0], CL_FALSE, 0, sizeof(cl_float2)*nx*ny, transmission_data, 0, NULL, NULL);
		} else {
			clEnqueueWriteBuffer(copy_queue, trans_buffer[0], CL_FALSE, 0, sizeof(cl_float2)*nx*ny, &transmissions[transmission_index(0, 0, 0)], 0, NULL, NULL);
		}
		
        clFinish(copy_queue);

        for (int pid = 0; pid < num_parallel; pid++) {
            
            for (int iy = 0; iy < sim_details.nyout; iy++) {
                for (int ix = 0; ix < sim_details.nxout; ix++) {
                    int index = ix * sim_details.nyout + iy;
                    
                    if ((location_count > pid) || progress_image[index]) {
                        continue;
                    }

                    location_count++;
                    current_points[pid].s0 = ix;
                    current_points[pid].s1 = iy;
                    
                    progress_image[index] = (device_number+1)*100;
                    msync(progress_image + index, sizeof(uint8_t), MS_SYNC);
                    double y = sim_details.yi + dy * ((double) iy);
                    double x = sim_details.xi + dx * ((double) ix);
                    
                    
					if (sim_details.dE_E > 0.0) {
						prepare_probe_espread(x, y, probes[pid], sim_details);
						clEnqueueWriteBuffer(command_queue, probes_buffer[pid], CL_FALSE, 0, sizeof(cl_float2)*probe_nx*probe_ny, probes[pid], 0, NULL, NULL);
					} else {
						if (GPU_PROBE_CALCULATION == 0) {
							prepare_probe_kirkland(x, y, probes[pid], sim_details);
							clEnqueueWriteBuffer(command_queue, probes_buffer[pid], CL_FALSE, 0, sizeof(cl_float2)*probe_nx*probe_ny, probes[pid], 0, NULL, NULL);
						} else {
							prepare_probe_device(x, y, probes_buffer[pid], sim_details);
						}	
					}
					
					if (SMOOTH_PROBE == 1) {
						probe_ifft(probes_buffer[pid]);
						gauss_multiply_device(probes_buffer[pid], 0.05);
						probe_fft(probes_buffer[pid]);
					}
					
					float probe_sum = 0.0;
                    sum_intensity(detect, probes_buffer[pid], &probe_sum);
                    scale_probe_device(probes_buffer[pid], probe_sum);
                    
                    stepCount++;
                    break;
                }
            }

        }
        
        if (location_count == 0) {
            break;
        }
        
        for (int slice_id = 0; slice_id < n_slices; slice_id++) {

            cl_mem use = trans_buffer[1];
            cl_mem copy = trans_buffer[0];

            if (slice_id != n_slices - 1) {
                if ((slice_id + 1) % 2 != 0) {
                    copy = trans_buffer[1];
                    use = trans_buffer[0];
                }
                
                if (use_hdd_mem > 0) {
					free(transmission_data); // Free old transmission data from previous iteration
					transmission_data = loadTransmissionData(slice_id + 1); // Load new transmission from disk (cache)
					clEnqueueWriteBuffer(copy_queue, copy, CL_FALSE, 0, sizeof(cl_float2)*nx*ny, transmission_data, 0, NULL, NULL);
				} else {
					clEnqueueWriteBuffer(copy_queue, copy, CL_FALSE, 0, sizeof(cl_float2)*nx*ny, &transmissions[transmission_index(0, 0, slice_id + 1)], 0, NULL, NULL);
				}
				
            }
        
            for (int pid = 0; pid < location_count; pid++) {
                int ix = current_points[pid].s0;
                int iy = current_points[pid].s1;
                double y = sim_details.yi + dy * ((double)iy);
                double x = sim_details.xi + dx * ((double) ix);
                int pixel_num = ix*sim_details.nyout + iy;
                
                STEMsignal(x, y, detect, &sum, 0, slice_id, probes_buffer[pid], use, pid, sim_details, pixel_num, edx_signal, ix, iy);
            }
        
            clFinish(copy_queue);
            clFinish(command_queue);

        }
        
        double tmpMin = 1000.0;
        double tmpMax = 0.0;

        float total_sum = 0.0;
        
        for (int pid = 0; pid < location_count; pid++) {
            int ix = current_points[pid].s0;
            int iy = current_points[pid].s1;
            
            total_sum = 0.0;
            sum_intensity(detect, probes_buffer[pid], &total_sum);
            
            if (total_sum < tmpMin) tmpMin = total_sum;
            if (total_sum > tmpMax) tmpMax = total_sum;
            if( total_sum < totmin ) totmin = total_sum;
            if( total_sum > totmax ) totmax = total_sum;
            
            for(int i=1; i<n_detect; i++) {
                pixr[i][ix][iy] = (float) detect[i];
            }
        }
        
        int index = current_points[location_count - 1].s1 * sim_details.nxout + current_points[location_count - 1].s0;
        float percent = (float)index/(float)(sim_details.nxout * sim_details.nyout) * 100.0;
        
        int intensities_ok = 0;
        for (int i = 1; i < n_detect; i++) {
            if (detect[i] == 0.0) {
                intensities_ok = i+1;
            }
        }
        
        if (intensities_ok == 0) {
            const char *min_color = (tmpMin < 0.85) ? KRED : KGRN;
            const char *max_color = (tmpMax < 0.85) ? KRED : KGRN;
            print_verbose("└─ %3.0f%%: %smin = %.5f%s, %smax = %.5f%s\r", percent, min_color, tmpMin, KNRM, max_color, tmpMax, KNRM);
        } else {
            print_verbose("└─ %s[WARNING] %3.0f%%: detector %i has low signal!%s\r", KRED, percent, intensities_ok, KNRM);
        }
        
        fflush(stdout);
        
        for(int idetect=1; idetect<n_detect; idetect++) {
            
            for (int pid = 0; pid < location_count; pid++) {
                located_intensity intensity;
                intensity.x = current_points[pid].s0;
                intensity.y = current_points[pid].s1;
                intensity.value = pixr[idetect][intensity.x][intensity.y];
                intensity.detector = idetect-1;
                
                fwrite(&intensity, sizeof(located_intensity), 1, output_file);
                fflush(output_file);
            }
            
        }
        
    }
    
    if (sim_details.calculate_edx > 0) {
        FILE *edx_output = fopen("edx_signal.bin", "ab+");
        for (int i = 0; i < n_slices; i++) {
            for (int x = 0; x < sim_details.nxout; x++) {
                for (int y = 0; y < sim_details.nyout; y++) {
                    int index = i * sim_details.nxout * sim_details.nyout + x * sim_details.nyout + y;
                    
                    fwrite(&edx_signal[index], sizeof(float), 1, edx_output);
                    fflush(edx_output);
                }
            }
        }
        
        fclose(edx_output);
    }
    
    
    print_verbose("\n");
    print_verbose("%s Simulation complete!%s\n", KGRN, KNRM);
    fclose(output_file);
    // free resources
    free(x_pos);
    free(y_pos);
    free(z_pos);
    free(zNum);
    
    freeGlobalResources();
    clfftTeardown();
    return 0;
}

#pragma mark - Potential calculation

double sigma(double kev) {
    double s, pi, wavl, x;
    const double emass=510.99906; /* electron rest mass in keV */
    double wavelength( double kev );  /*  get electron wavelength */
    
    x = ( emass + kev ) / ( 2.0*emass + kev);
    wavl = wavelength( kev );
    pi = 4.0 * atan( 1.0 );
    
    s = 2.0 * pi * x / (wavl*kev);
    
    return s;
}

char *transmissionFilename(int slice_num) {
    char *transmission_file_name = malloc(1024);
    
    struct stat st = {0};
    if (stat("transmissions/", &st) == -1) {
        mkdir("transmissions/", 0777);
    }
    
    sprintf(transmission_file_name, "transmissions/transmission_%05d.bin", slice_num);
    return transmission_file_name;
}

cl_float2 *loadTransmissionData(int slice_num) {
    char *transmission_file_name = transmissionFilename(slice_num);
    FILE *transmission_file = fopen(transmission_file_name, "rb");
    fseek(transmission_file, 0, SEEK_END);
    long transmission_data_size = ftell(transmission_file);
    fseek(transmission_file, 0, SEEK_SET);
    cl_float2 *transmission_data = malloc(transmission_data_size);
    fread(transmission_data, sizeof(cl_float2), transmission_data_size/sizeof(cl_float2), transmission_file);
    fclose(transmission_file);
    free(transmission_file_name);
    return transmission_data;
}

void calculateTransmissionLayer(const float x[], const float y[],
                                const int Znum[], const int natom,
                                const float ax, const float by, const float kev,
                                int slice_num, const long nx, const long ny,
                                const float k2max, cl_mem trans_buffer) {
    char *transmission_file_name = transmissionFilename(slice_num);
    
    if ((use_hdd_mem == 1) && (access(transmission_file_name, F_OK) != -1)) {
        free(transmission_file_name);
        return; // File already exists.
    }

    FILE *transmission_file;
    
    if (use_hdd_mem == 1) {
		transmission_file = fopen(transmission_file_name, "wb");
	}
	
    int idx, idy, i, ixo, iyo, ix, iy, ixw, iyw, nx1, nx2, ny1, ny2;
    float k2;
    double r, rx2, rsq, vz, rmin, rmin2, sum, scale, scalex, scaley;
    size_t transmission_data_size = sizeof(cl_float2)*nx*ny;
    cl_float2 *transmission_data = malloc(transmission_data_size);
    const double rmax=3.0, rmax2=rmax*rmax;
    
    scale = sigma(kev) / 1000.0;
    
    scalex = ax/nx;
    scaley = by/ny;
    
    
    rmin = ax/((double)nx);
    r = by/((double)ny);
    rmin =  0.25 * sqrt( 0.5*(rmin*rmin + r*r) );
    rmin2 = rmin*rmin;
    
    idx = (int) ( nx*rmax/ax ) + 1;
    idy = (int) ( ny*rmax/by ) + 1;
    
    for( ix=0; ix<nx; ix++) {
        for( iy=0; iy<ny; iy++) {
            int trans_index = iy * nx + ix;
            transmission_data[trans_index].s0 = 0.0;
        }
    }
    
    for( i=0; i<natom; i++) {
        ixo = (int) ( x[i]/scalex );
        iyo = (int) ( y[i]/scaley );
        nx1 = ixo - idx;
        nx2 = ixo + idx;
        ny1 = iyo - idy;
        ny2 = iyo + idy;
        
        for( ix=nx1; ix<=nx2; ix++) {
            rx2 = x[i] - ((double)ix)*scalex;
            rx2 = rx2 * rx2;
            ixw = ix;
            while( ixw < 0 ) ixw = (int)(ixw + nx);
            ixw = ixw % nx;
            for( iy=ny1; iy<=ny2; iy++) {
                rsq = y[i] - ((double)iy)*scaley;
                rsq = rx2 + rsq*rsq;
                if( rsq <= rmax2 ) {
                    iyw = iy;
                    while( iyw < 0 ) iyw = (int)(iyw + ny);
                    iyw = iyw % ny;
                    if( rsq < rmin2 ) rsq = rmin2;
                    vz =  vzatomLUT( Znum[i], rsq );
                    int trans_index = iyw * nx + ixw;
                    transmission_data[trans_index].s0 += (float) vz;
                }
            }
        }
        
    }
    
    sum = 0;
    for( ix=0; ix<nx; ix++) {
        for( iy=0; iy<ny; iy++) {
            int trans_index = iy * nx + ix;
            vz = scale * transmission_data[trans_index].s0;
            sum += vz;
            transmission_data[trans_index].s0 = (float) cos( vz );
            transmission_data[trans_index].s1 = (float) sin( vz );
        }
    }
    
    /* bandwidth limit the transmission function */
    
    clEnqueueWriteBuffer(command_queue, trans_buffer, CL_TRUE, 0, sizeof(cl_float2)*nx*ny, transmission_data, 0, NULL, NULL);
    clfftEnqueueTransform(plan, CLFFT_FORWARD, 1, &command_queue, 0, NULL, NULL, &trans_buffer, NULL, NULL);
    clEnqueueReadBuffer(command_queue, trans_buffer, CL_TRUE, 0, sizeof(cl_float2)*nx*ny, transmission_data, 0, NULL, NULL);
    
    for( ix=0; ix<nx; ix++) {
        for( iy=0; iy<ny; iy++) {
            k2 = spat_freqy2[iy] + spat_freqx2[ix];
            if (k2 > k2max) {
                int trans_index = iy * nx + ix;
                transmission_data[trans_index].s0 = transmission_data[trans_index].s1 = 0.0;
            }
        }
    }
    
    clEnqueueWriteBuffer(command_queue, trans_buffer, CL_TRUE, 0, sizeof(cl_float2)*nx*ny, transmission_data, 0, NULL, NULL);
    clfftEnqueueTransform(plan, CLFFT_BACKWARD, 1, &command_queue, 0, NULL, NULL, &trans_buffer, NULL, NULL);
    clEnqueueReadBuffer(command_queue, trans_buffer, CL_TRUE, 0, sizeof(cl_float2)*nx*ny, transmission_data, 0, NULL, NULL);
    
    if (use_hdd_mem == 0) {
		for( ix=0; ix<nx; ix++) {
			for( iy=0; iy<ny; iy++) {
				int trans_index = iy * nx + ix;
				transmissions[transmission_index(ix, iy, slice_num)] = transmission_data[trans_index];
			}
		}
		
	} else {
		fwrite(transmission_data, sizeof(cl_float2), nx * ny, transmission_file);
		fclose(transmission_file);
		free(transmission_file_name);
		free(transmission_data);
	}
}

int transmission_index(int x, int y, int slice) {
    return (slice * ny * nx + x * ny + y);
}

double wavelength(double kev) {
    double w;
    const double emass=510.99906;
    const double hc=12.3984244;
    
    w = hc/sqrt(kev * (2*emass + kev));
    
    return w;
    
}

void frequency(float *ko, float *ko2, float *xo, int nk, double ak) {
    int i, imid;
    
    imid = (int) ( nk/2.0 + 0.5);
    
    for( i=0; i<nk; i++) {
        xo[i] = ((float) (i * ak) ) / ((float)(nk-1));
        if ( i > imid ) {
            ko[i]  = ((float)(i-nk)) / ((float)ak);
        } else {
            ko[i]  = ((float)i) / ((float)ak);
        }
        ko2[i] = ko[i] * ko[i];
    }
    
}

int ReadfeTable( )
{
    /*  hard code the scattering factor parameters so you
     don't have to keep track of the extra file and also
     to guarantee that the data won't be changed*/
    
    /* if the file has been initialized already then just return */
    if( feTableRead == 1 ) return(0);
    
    fparams = (double **)malloc((NZMAX + 1) * NPMAX * sizeof(double*));
    for (int i = 0; i < NZMAX + 1; i++)
        fparams[i] = (double *)malloc(NPMAX * sizeof(double));
    
    /*  scattering factor parameter data from fparams  */
    
    fparams[1][0] =  4.20298324e-003 ;
    fparams[1][1] =  2.25350888e-001 ;
    fparams[1][2] =  6.27762505e-002 ;
    fparams[1][3] =  2.25366950e-001 ;
    fparams[1][4] =  3.00907347e-002 ;
    fparams[1][5] =  2.25331756e-001 ;
    fparams[1][6] =  6.77756695e-002 ;
    fparams[1][7] =  4.38854001e+000 ;
    fparams[1][8] =  3.56609237e-003 ;
    fparams[1][9] =  4.03884823e-001 ;
    fparams[1][10] =  2.76135815e-002 ;
    fparams[1][11] =  1.44490166e+000 ;
    fparams[2][0] =  1.87543704e-005 ;
    fparams[2][1] =  2.12427997e-001 ;
    fparams[2][2] =  4.10595800e-004 ;
    fparams[2][3] =  3.32212279e-001 ;
    fparams[2][4] =  1.96300059e-001 ;
    fparams[2][5] =  5.17325152e-001 ;
    fparams[2][6] =  8.36015738e-003 ;
    fparams[2][7] =  3.66668239e-001 ;
    fparams[2][8] =  2.95102022e-002 ;
    fparams[2][9] =  1.37171827e+000 ;
    fparams[2][10] =  4.65928982e-007 ;
    fparams[2][11] =  3.75768025e+004 ;
    fparams[3][0] =  7.45843816e-002 ;
    fparams[3][1] =  8.81151424e-001 ;
    fparams[3][2] =  7.15382250e-002 ;
    fparams[3][3] =  4.59142904e-002 ;
    fparams[3][4] =  1.45315229e-001 ;
    fparams[3][5] =  8.81301714e-001 ;
    fparams[3][6] =  1.12125769e+000 ;
    fparams[3][7] =  1.88483665e+001 ;
    fparams[3][8] =  2.51736525e-003 ;
    fparams[3][9] =  1.59189995e-001 ;
    fparams[3][10] =  3.58434971e-001 ;
    fparams[3][11] =  6.12371000e+000 ;
    fparams[4][0] =  6.11642897e-002 ;
    fparams[4][1] =  9.90182132e-002 ;
    fparams[4][2] =  1.25755034e-001 ;
    fparams[4][3] =  9.90272412e-002 ;
    fparams[4][4] =  2.00831548e-001 ;
    fparams[4][5] =  1.87392509e+000 ;
    fparams[4][6] =  7.87242876e-001 ;
    fparams[4][7] =  9.32794929e+000 ;
    fparams[4][8] =  1.58847850e-003 ;
    fparams[4][9] =  8.91900236e-002 ;
    fparams[4][10] =  2.73962031e-001 ;
    fparams[4][11] =  3.20687658e+000 ;
    fparams[5][0] =  1.25716066e-001 ;
    fparams[5][1] =  1.48258830e-001 ;
    fparams[5][2] =  1.73314452e-001 ;
    fparams[5][3] =  1.48257216e-001 ;
    fparams[5][4] =  1.84774811e-001 ;
    fparams[5][5] =  3.34227311e+000 ;
    fparams[5][6] =  1.95250221e-001 ;
    fparams[5][7] =  1.97339463e+000 ;
    fparams[5][8] =  5.29642075e-001 ;
    fparams[5][9] =  5.70035553e+000 ;
    fparams[5][10] =  1.08230500e-003 ;
    fparams[5][11] =  5.64857237e-002 ;
    fparams[6][0] =  2.12080767e-001 ;
    fparams[6][1] =  2.08605417e-001 ;
    fparams[6][2] =  1.99811865e-001 ;
    fparams[6][3] =  2.08610186e-001 ;
    fparams[6][4] =  1.68254385e-001 ;
    fparams[6][5] =  5.57870773e+000 ;
    fparams[6][6] =  1.42048360e-001 ;
    fparams[6][7] =  1.33311887e+000 ;
    fparams[6][8] =  3.63830672e-001 ;
    fparams[6][9] =  3.80800263e+000 ;
    fparams[6][10] =  8.35012044e-004 ;
    fparams[6][11] =  4.03982620e-002 ;
    fparams[7][0] =  5.33015554e-001 ;
    fparams[7][1] =  2.90952515e-001 ;
    fparams[7][2] =  5.29008883e-002 ;
    fparams[7][3] =  1.03547896e+001 ;
    fparams[7][4] =  9.24159648e-002 ;
    fparams[7][5] =  1.03540028e+001 ;
    fparams[7][6] =  2.61799101e-001 ;
    fparams[7][7] =  2.76252723e+000 ;
    fparams[7][8] =  8.80262108e-004 ;
    fparams[7][9] =  3.47681236e-002 ;
    fparams[7][10] =  1.10166555e-001 ;
    fparams[7][11] =  9.93421736e-001 ;
    fparams[8][0] =  3.39969204e-001 ;
    fparams[8][1] =  3.81570280e-001 ;
    fparams[8][2] =  3.07570172e-001 ;
    fparams[8][3] =  3.81571436e-001 ;
    fparams[8][4] =  1.30369072e-001 ;
    fparams[8][5] =  1.91919745e+001 ;
    fparams[8][6] =  8.83326058e-002 ;
    fparams[8][7] =  7.60635525e-001 ;
    fparams[8][8] =  1.96586700e-001 ;
    fparams[8][9] =  2.07401094e+000 ;
    fparams[8][10] =  9.96220028e-004 ;
    fparams[8][11] =  3.03266869e-002 ;
    fparams[9][0] =  2.30560593e-001 ;
    fparams[9][1] =  4.80754213e-001 ;
    fparams[9][2] =  5.26889648e-001 ;
    fparams[9][3] =  4.80763895e-001 ;
    fparams[9][4] =  1.24346755e-001 ;
    fparams[9][5] =  3.95306720e+001 ;
    fparams[9][6] =  1.24616894e-003 ;
    fparams[9][7] =  2.62181803e-002 ;
    fparams[9][8] =  7.20452555e-002 ;
    fparams[9][9] =  5.92495593e-001 ;
    fparams[9][10] =  1.53075777e-001 ;
    fparams[9][11] =  1.59127671e+000 ;
    fparams[10][0] =  4.08371771e-001 ;
    fparams[10][1] =  5.88228627e-001 ;
    fparams[10][2] =  4.54418858e-001 ;
    fparams[10][3] =  5.88288655e-001 ;
    fparams[10][4] =  1.44564923e-001 ;
    fparams[10][5] =  1.21246013e+002 ;
    fparams[10][6] =  5.91531395e-002 ;
    fparams[10][7] =  4.63963540e-001 ;
    fparams[10][8] =  1.24003718e-001 ;
    fparams[10][9] =  1.23413025e+000 ;
    fparams[10][10] =  1.64986037e-003 ;
    fparams[10][11] =  2.05869217e-002 ;
    fparams[11][0] =  1.36471662e-001 ;
    fparams[11][1] =  4.99965301e-002 ;
    fparams[11][2] =  7.70677865e-001 ;
    fparams[11][3] =  8.81899664e-001 ;
    fparams[11][4] =  1.56862014e-001 ;
    fparams[11][5] =  1.61768579e+001 ;
    fparams[11][6] =  9.96821513e-001 ;
    fparams[11][7] =  2.00132610e+001 ;
    fparams[11][8] =  3.80304670e-002 ;
    fparams[11][9] =  2.60516254e-001 ;
    fparams[11][10] =  1.27685089e-001 ;
    fparams[11][11] =  6.99559329e-001 ;
    fparams[12][0] =  3.04384121e-001 ;
    fparams[12][1] =  8.42014377e-002 ;
    fparams[12][2] =  7.56270563e-001 ;
    fparams[12][3] =  1.64065598e+000 ;
    fparams[12][4] =  1.01164809e-001 ;
    fparams[12][5] =  2.97142975e+001 ;
    fparams[12][6] =  3.45203403e-002 ;
    fparams[12][7] =  2.16596094e-001 ;
    fparams[12][8] =  9.71751327e-001 ;
    fparams[12][9] =  1.21236852e+001 ;
    fparams[12][10] =  1.20593012e-001 ;
    fparams[12][11] =  5.60865838e-001 ;
    fparams[13][0] =  7.77419424e-001 ;
    fparams[13][1] =  2.71058227e+000 ;
    fparams[13][2] =  5.78312036e-002 ;
    fparams[13][3] =  7.17532098e+001 ;
    fparams[13][4] =  4.26386499e-001 ;
    fparams[13][5] =  9.13331555e-002 ;
    fparams[13][6] =  1.13407220e-001 ;
    fparams[13][7] =  4.48867451e-001 ;
    fparams[13][8] =  7.90114035e-001 ;
    fparams[13][9] =  8.66366718e+000 ;
    fparams[13][10] =  3.23293496e-002 ;
    fparams[13][11] =  1.78503463e-001 ;
    fparams[14][0] =  1.06543892e+000 ;
    fparams[14][1] =  1.04118455e+000 ;
    fparams[14][2] =  1.20143691e-001 ;
    fparams[14][3] =  6.87113368e+001 ;
    fparams[14][4] =  1.80915263e-001 ;
    fparams[14][5] =  8.87533926e-002 ;
    fparams[14][6] =  1.12065620e+000 ;
    fparams[14][7] =  3.70062619e+000 ;
    fparams[14][8] =  3.05452816e-002 ;
    fparams[14][9] =  2.14097897e-001 ;
    fparams[14][10] =  1.59963502e+000 ;
    fparams[14][11] =  9.99096638e+000 ;
    fparams[15][0] =  1.05284447e+000 ;
    fparams[15][1] =  1.31962590e+000 ;
    fparams[15][2] =  2.99440284e-001 ;
    fparams[15][3] =  1.28460520e-001 ;
    fparams[15][4] =  1.17460748e-001 ;
    fparams[15][5] =  1.02190163e+002 ;
    fparams[15][6] =  9.60643452e-001 ;
    fparams[15][7] =  2.87477555e+000 ;
    fparams[15][8] =  2.63555748e-002 ;
    fparams[15][9] =  1.82076844e-001 ;
    fparams[15][10] =  1.38059330e+000 ;
    fparams[15][11] =  7.49165526e+000 ;
    fparams[16][0] =  1.01646916e+000 ;
    fparams[16][1] =  1.69181965e+000 ;
    fparams[16][2] =  4.41766748e-001 ;
    fparams[16][3] =  1.74180288e-001 ;
    fparams[16][4] =  1.21503863e-001 ;
    fparams[16][5] =  1.67011091e+002 ;
    fparams[16][6] =  8.27966670e-001 ;
    fparams[16][7] =  2.30342810e+000 ;
    fparams[16][8] =  2.33022533e-002 ;
    fparams[16][9] =  1.56954150e-001 ;
    fparams[16][10] =  1.18302846e+000 ;
    fparams[16][11] =  5.85782891e+000 ;
    fparams[17][0] =  9.44221116e-001 ;
    fparams[17][1] =  2.40052374e-001 ;
    fparams[17][2] =  4.37322049e-001 ;
    fparams[17][3] =  9.30510439e+000 ;
    fparams[17][4] =  2.54547926e-001 ;
    fparams[17][5] =  9.30486346e+000 ;
    fparams[17][6] =  5.47763323e-002 ;
    fparams[17][7] =  1.68655688e-001 ;
    fparams[17][8] =  8.00087488e-001 ;
    fparams[17][9] =  2.97849774e+000 ;
    fparams[17][10] =  1.07488641e-002 ;
    fparams[17][11] =  6.84240646e-002 ;
    fparams[18][0] =  1.06983288e+000 ;
    fparams[18][1] =  2.87791022e-001 ;
    fparams[18][2] =  4.24631786e-001 ;
    fparams[18][3] =  1.24156957e+001 ;
    fparams[18][4] =  2.43897949e-001 ;
    fparams[18][5] =  1.24158868e+001 ;
    fparams[18][6] =  4.79446296e-002 ;
    fparams[18][7] =  1.36979796e-001 ;
    fparams[18][8] =  7.64958952e-001 ;
    fparams[18][9] =  2.43940729e+000 ;
    fparams[18][10] =  8.23128431e-003 ;
    fparams[18][11] =  5.27258749e-002 ;
    fparams[19][0] =  6.92717865e-001 ;
    fparams[19][1] =  7.10849990e+000 ;
    fparams[19][2] =  9.65161085e-001 ;
    fparams[19][3] =  3.57532901e-001 ;
    fparams[19][4] =  1.48466588e-001 ;
    fparams[19][5] =  3.93763275e-002 ;
    fparams[19][6] =  2.64645027e-002 ;
    fparams[19][7] =  1.03591321e-001 ;
    fparams[19][8] =  1.80883768e+000 ;
    fparams[19][9] =  3.22845199e+001 ;
    fparams[19][10] =  5.43900018e-001 ;
    fparams[19][11] =  1.67791374e+000 ;
    fparams[20][0] =  3.66902871e-001 ;
    fparams[20][1] =  6.14274129e-002 ;
    fparams[20][2] =  8.66378999e-001 ;
    fparams[20][3] =  5.70881727e-001 ;
    fparams[20][4] =  6.67203300e-001 ;
    fparams[20][5] =  7.82965639e+000 ;
    fparams[20][6] =  4.87743636e-001 ;
    fparams[20][7] =  1.32531318e+000 ;
    fparams[20][8] =  1.82406314e+000 ;
    fparams[20][9] =  2.10056032e+001 ;
    fparams[20][10] =  2.20248453e-002 ;
    fparams[20][11] =  9.11853450e-002 ;
    fparams[21][0] =  3.78871777e-001 ;
    fparams[21][1] =  6.98910162e-002 ;
    fparams[21][2] =  9.00022505e-001 ;
    fparams[21][3] =  5.21061541e-001 ;
    fparams[21][4] =  7.15288914e-001 ;
    fparams[21][5] =  7.87707920e+000 ;
    fparams[21][6] =  1.88640973e-002 ;
    fparams[21][7] =  8.17512708e-002 ;
    fparams[21][8] =  4.07945949e-001 ;
    fparams[21][9] =  1.11141388e+000 ;
    fparams[21][10] =  1.61786540e+000 ;
    fparams[21][11] =  1.80840759e+001 ;
    fparams[22][0] =  3.62383267e-001 ;
    fparams[22][1] =  7.54707114e-002 ;
    fparams[22][2] =  9.84232966e-001 ;
    fparams[22][3] =  4.97757309e-001 ;
    fparams[22][4] =  7.41715642e-001 ;
    fparams[22][5] =  8.17659391e+000 ;
    fparams[22][6] =  3.62555269e-001 ;
    fparams[22][7] =  9.55524906e-001 ;
    fparams[22][8] =  1.49159390e+000 ;
    fparams[22][9] =  1.62221677e+001 ;
    fparams[22][10] =  1.61659509e-002 ;
    fparams[22][11] =  7.33140839e-002 ;
    fparams[23][0] =  3.52961378e-001 ;
    fparams[23][1] =  8.19204103e-002 ;
    fparams[23][2] =  7.46791014e-001 ;
    fparams[23][3] =  8.81189511e+000 ;
    fparams[23][4] =  1.08364068e+000 ;
    fparams[23][5] =  5.10646075e-001 ;
    fparams[23][6] =  1.39013610e+000 ;
    fparams[23][7] =  1.48901841e+001 ;
    fparams[23][8] =  3.31273356e-001 ;
    fparams[23][9] =  8.38543079e-001 ;
    fparams[23][10] =  1.40422612e-002 ;
    fparams[23][11] =  6.57432678e-002 ;
    fparams[24][0] =  1.34348379e+000 ;
    fparams[24][1] =  1.25814353e+000 ;
    fparams[24][2] =  5.07040328e-001 ;
    fparams[24][3] =  1.15042811e+001 ;
    fparams[24][4] =  4.26358955e-001 ;
    fparams[24][5] =  8.53660389e-002 ;
    fparams[24][6] =  1.17241826e-002 ;
    fparams[24][7] =  6.00177061e-002 ;
    fparams[24][8] =  5.11966516e-001 ;
    fparams[24][9] =  1.53772451e+000 ;
    fparams[24][10] =  3.38285828e-001 ;
    fparams[24][11] =  6.62418319e-001 ;
    fparams[25][0] =  3.26697613e-001 ;
    fparams[25][1] =  8.88813083e-002 ;
    fparams[25][2] =  7.17297000e-001 ;
    fparams[25][3] =  1.11300198e+001 ;
    fparams[25][4] =  1.33212464e+000 ;
    fparams[25][5] =  5.82141104e-001 ;
    fparams[25][6] =  2.80801702e-001 ;
    fparams[25][7] =  6.71583145e-001 ;
    fparams[25][8] =  1.15499241e+000 ;
    fparams[25][9] =  1.26825395e+001 ;
    fparams[25][10] =  1.11984488e-002 ;
    fparams[25][11] =  5.32334467e-002 ;
    fparams[26][0] =  3.13454847e-001 ;
    fparams[26][1] =  8.99325756e-002 ;
    fparams[26][2] =  6.89290016e-001 ;
    fparams[26][3] =  1.30366038e+001 ;
    fparams[26][4] =  1.47141531e+000 ;
    fparams[26][5] =  6.33345291e-001 ;
    fparams[26][6] =  1.03298688e+000 ;
    fparams[26][7] =  1.16783425e+001 ;
    fparams[26][8] =  2.58280285e-001 ;
    fparams[26][9] =  6.09116446e-001 ;
    fparams[26][10] =  1.03460690e-002 ;
    fparams[26][11] =  4.81610627e-002 ;
    fparams[27][0] =  3.15878278e-001 ;
    fparams[27][1] =  9.46683246e-002 ;
    fparams[27][2] =  1.60139005e+000 ;
    fparams[27][3] =  6.99436449e-001 ;
    fparams[27][4] =  6.56394338e-001 ;
    fparams[27][5] =  1.56954403e+001 ;
    fparams[27][6] =  9.36746624e-001 ;
    fparams[27][7] =  1.09392410e+001 ;
    fparams[27][8] =  9.77562646e-003 ;
    fparams[27][9] =  4.37446816e-002 ;
    fparams[27][10] =  2.38378578e-001 ;
    fparams[27][11] =  5.56286483e-001 ;
    fparams[28][0] =  1.72254630e+000 ;
    fparams[28][1] =  7.76606908e-001 ;
    fparams[28][2] =  3.29543044e-001 ;
    fparams[28][3] =  1.02262360e-001 ;
    fparams[28][4] =  6.23007200e-001 ;
    fparams[28][5] =  1.94156207e+001 ;
    fparams[28][6] =  9.43496513e-003 ;
    fparams[28][7] =  3.98684596e-002 ;
    fparams[28][8] =  8.54063515e-001 ;
    fparams[28][9] =  1.04078166e+001 ;
    fparams[28][10] =  2.21073515e-001 ;
    fparams[28][11] =  5.10869330e-001 ;
    fparams[29][0] =  3.58774531e-001 ;
    fparams[29][1] =  1.06153463e-001 ;
    fparams[29][2] =  1.76181348e+000 ;
    fparams[29][3] =  1.01640995e+000 ;
    fparams[29][4] =  6.36905053e-001 ;
    fparams[29][5] =  1.53659093e+001 ;
    fparams[29][6] =  7.44930667e-003 ;
    fparams[29][7] =  3.85345989e-002 ;
    fparams[29][8] =  1.89002347e-001 ;
    fparams[29][9] =  3.98427790e-001 ;
    fparams[29][10] =  2.29619589e-001 ;
    fparams[29][11] =  9.01419843e-001 ;
    fparams[30][0] =  5.70893973e-001 ;
    fparams[30][1] =  1.26534614e-001 ;
    fparams[30][2] =  1.98908856e+000 ;
    fparams[30][3] =  2.17781965e+000 ;
    fparams[30][4] =  3.06060585e-001 ;
    fparams[30][5] =  3.78619003e+001 ;
    fparams[30][6] =  2.35600223e-001 ;
    fparams[30][7] =  3.67019041e-001 ;
    fparams[30][8] =  3.97061102e-001 ;
    fparams[30][9] =  8.66419596e-001 ;
    fparams[30][10] =  6.85657228e-003 ;
    fparams[30][11] =  3.35778823e-002 ;
    fparams[31][0] =  6.25528464e-001 ;
    fparams[31][1] =  1.10005650e-001 ;
    fparams[31][2] =  2.05302901e+000 ;
    fparams[31][3] =  2.41095786e+000 ;
    fparams[31][4] =  2.89608120e-001 ;
    fparams[31][5] =  4.78685736e+001 ;
    fparams[31][6] =  2.07910594e-001 ;
    fparams[31][7] =  3.27807224e-001 ;
    fparams[31][8] =  3.45079617e-001 ;
    fparams[31][9] =  7.43139061e-001 ;
    fparams[31][10] =  6.55634298e-003 ;
    fparams[31][11] =  3.09411369e-002 ;
    fparams[32][0] =  5.90952690e-001 ;
    fparams[32][1] =  1.18375976e-001 ;
    fparams[32][2] =  5.39980660e-001 ;
    fparams[32][3] =  7.18937433e+001 ;
    fparams[32][4] =  2.00626188e+000 ;
    fparams[32][5] =  1.39304889e+000 ;
    fparams[32][6] =  7.49705041e-001 ;
    fparams[32][7] =  6.89943350e+000 ;
    fparams[32][8] =  1.83581347e-001 ;
    fparams[32][9] =  3.64667232e-001 ;
    fparams[32][10] =  9.52190743e-003 ;
    fparams[32][11] =  2.69888650e-002 ;
    fparams[33][0] =  7.77875218e-001 ;
    fparams[33][1] =  1.50733157e-001 ;
    fparams[33][2] =  5.93848150e-001 ;
    fparams[33][3] =  1.42882209e+002 ;
    fparams[33][4] =  1.95918751e+000 ;
    fparams[33][5] =  1.74750339e+000 ;
    fparams[33][6] =  1.79880226e-001 ;
    fparams[33][7] =  3.31800852e-001 ;
    fparams[33][8] =  8.63267222e-001 ;
    fparams[33][9] =  5.85490274e+000 ;
    fparams[33][10] =  9.59053427e-003 ;
    fparams[33][11] =  2.33777569e-002 ;
    fparams[34][0] =  9.58390681e-001 ;
    fparams[34][1] =  1.83775557e-001 ;
    fparams[34][2] =  6.03851342e-001 ;
    fparams[34][3] =  1.96819224e+002 ;
    fparams[34][4] =  1.90828931e+000 ;
    fparams[34][5] =  2.15082053e+000 ;
    fparams[34][6] =  1.73885956e-001 ;
    fparams[34][7] =  3.00006024e-001 ;
    fparams[34][8] =  9.35265145e-001 ;
    fparams[34][9] =  4.92471215e+000 ;
    fparams[34][10] =  8.62254658e-003 ;
    fparams[34][11] =  2.12308108e-002 ;
    fparams[35][0] =  1.14136170e+000 ;
    fparams[35][1] =  2.18708710e-001 ;
    fparams[35][2] =  5.18118737e-001 ;
    fparams[35][3] =  1.93916682e+002 ;
    fparams[35][4] =  1.85731975e+000 ;
    fparams[35][5] =  2.65755396e+000 ;
    fparams[35][6] =  1.68217399e-001 ;
    fparams[35][7] =  2.71719918e-001 ;
    fparams[35][8] =  9.75705606e-001 ;
    fparams[35][9] =  4.19482500e+000 ;
    fparams[35][10] =  7.24187871e-003 ;
    fparams[35][11] =  1.99325718e-002 ;
    fparams[36][0] =  3.24386970e-001 ;
    fparams[36][1] =  6.31317973e+001 ;
    fparams[36][2] =  1.31732163e+000 ;
    fparams[36][3] =  2.54706036e-001 ;
    fparams[36][4] =  1.79912614e+000 ;
    fparams[36][5] =  3.23668394e+000 ;
    fparams[36][6] =  4.29961425e-003 ;
    fparams[36][7] =  1.98965610e-002 ;
    fparams[36][8] =  1.00429433e+000 ;
    fparams[36][9] =  3.61094513e+000 ;
    fparams[36][10] =  1.62188197e-001 ;
    fparams[36][11] =  2.45583672e-001 ;
    fparams[37][0] =  2.90445351e-001 ;
    fparams[37][1] =  3.68420227e-002 ;
    fparams[37][2] =  2.44201329e+000 ;
    fparams[37][3] =  1.16013332e+000 ;
    fparams[37][4] =  7.69435449e-001 ;
    fparams[37][5] =  1.69591472e+001 ;
    fparams[37][6] =  1.58687000e+000 ;
    fparams[37][7] =  2.53082574e+000 ;
    fparams[37][8] =  2.81617593e-003 ;
    fparams[37][9] =  1.88577417e-002 ;
    fparams[37][10] =  1.28663830e-001 ;
    fparams[37][11] =  2.10753969e-001 ;
    fparams[38][0] =  1.37373086e-002 ;
    fparams[38][1] =  1.87469061e-002 ;
    fparams[38][2] =  1.97548672e+000 ;
    fparams[38][3] =  6.36079230e+000 ;
    fparams[38][4] =  1.59261029e+000 ;
    fparams[38][5] =  2.21992482e-001 ;
    fparams[38][6] =  1.73263882e-001 ;
    fparams[38][7] =  2.01624958e-001 ;
    fparams[38][8] =  4.66280378e+000 ;
    fparams[38][9] =  2.53027803e+001 ;
    fparams[38][10] =  1.61265063e-003 ;
    fparams[38][11] =  1.53610568e-002 ;
    fparams[39][0] =  6.75302747e-001 ;
    fparams[39][1] =  6.54331847e-002 ;
    fparams[39][2] =  4.70286720e-001 ;
    fparams[39][3] =  1.06108709e+002 ;
    fparams[39][4] =  2.63497677e+000 ;
    fparams[39][5] =  2.06643540e+000 ;
    fparams[39][6] =  1.09621746e-001 ;
    fparams[39][7] =  1.93131925e-001 ;
    fparams[39][8] =  9.60348773e-001 ;
    fparams[39][9] =  1.63310938e+000 ;
    fparams[39][10] =  5.28921555e-003 ;
    fparams[39][11] =  1.66083821e-002 ;
    fparams[40][0] =  2.64365505e+000 ;
    fparams[40][1] =  2.20202699e+000 ;
    fparams[40][2] =  5.54225147e-001 ;
    fparams[40][3] =  1.78260107e+002 ;
    fparams[40][4] =  7.61376625e-001 ;
    fparams[40][5] =  7.67218745e-002 ;
    fparams[40][6] =  6.02946891e-003 ;
    fparams[40][7] =  1.55143296e-002 ;
    fparams[40][8] =  9.91630530e-002 ;
    fparams[40][9] =  1.76175995e-001 ;
    fparams[40][10] =  9.56782020e-001 ;
    fparams[40][11] =  1.54330682e+000 ;
    fparams[41][0] =  6.59532875e-001 ;
    fparams[41][1] =  8.66145490e-002 ;
    fparams[41][2] =  1.84545854e+000 ;
    fparams[41][3] =  5.94774398e+000 ;
    fparams[41][4] =  1.25584405e+000 ;
    fparams[41][5] =  6.40851475e-001 ;
    fparams[41][6] =  1.22253422e-001 ;
    fparams[41][7] =  1.66646050e-001 ;
    fparams[41][8] =  7.06638328e-001 ;
    fparams[41][9] =  1.62853268e+000 ;
    fparams[41][10] =  2.62381591e-003 ;
    fparams[41][11] =  8.26257859e-003 ;
    fparams[42][0] =  6.10160120e-001 ;
    fparams[42][1] =  9.11628054e-002 ;
    fparams[42][2] =  1.26544000e+000 ;
    fparams[42][3] =  5.06776025e-001 ;
    fparams[42][4] =  1.97428762e+000 ;
    fparams[42][5] =  5.89590381e+000 ;
    fparams[42][6] =  6.48028962e-001 ;
    fparams[42][7] =  1.46634108e+000 ;
    fparams[42][8] =  2.60380817e-003 ;
    fparams[42][9] =  7.84336311e-003 ;
    fparams[42][10] =  1.13887493e-001 ;
    fparams[42][11] =  1.55114340e-001 ;
    fparams[43][0] =  8.55189183e-001 ;
    fparams[43][1] =  1.02962151e-001 ;
    fparams[43][2] =  1.66219641e+000 ;
    fparams[43][3] =  7.64907000e+000 ;
    fparams[43][4] =  1.45575475e+000 ;
    fparams[43][5] =  1.01639987e+000 ;
    fparams[43][6] =  1.05445664e-001 ;
    fparams[43][7] =  1.42303338e-001 ;
    fparams[43][8] =  7.71657112e-001 ;
    fparams[43][9] =  1.34659349e+000 ;
    fparams[43][10] =  2.20992635e-003 ;
    fparams[43][11] =  7.90358976e-003 ;
    fparams[44][0] =  4.70847093e-001 ;
    fparams[44][1] =  9.33029874e-002 ;
    fparams[44][2] =  1.58180781e+000 ;
    fparams[44][3] =  4.52831347e-001 ;
    fparams[44][4] =  2.02419818e+000 ;
    fparams[44][5] =  7.11489023e+000 ;
    fparams[44][6] =  1.97036257e-003 ;
    fparams[44][7] =  7.56181595e-003 ;
    fparams[44][8] =  6.26912639e-001 ;
    fparams[44][9] =  1.25399858e+000 ;
    fparams[44][10] =  1.02641320e-001 ;
    fparams[44][11] =  1.33786087e-001 ;
    fparams[45][0] =  4.20051553e-001 ;
    fparams[45][1] =  9.38882628e-002 ;
    fparams[45][2] =  1.76266507e+000 ;
    fparams[45][3] =  4.64441687e-001 ;
    fparams[45][4] =  2.02735641e+000 ;
    fparams[45][5] =  8.19346046e+000 ;
    fparams[45][6] =  1.45487176e-003 ;
    fparams[45][7] =  7.82704517e-003 ;
    fparams[45][8] =  6.22809600e-001 ;
    fparams[45][9] =  1.17194153e+000 ;
    fparams[45][10] =  9.91529915e-002 ;
    fparams[45][11] =  1.24532839e-001 ;
    fparams[46][0] =  2.10475155e+000 ;
    fparams[46][1] =  8.68606470e+000 ;
    fparams[46][2] =  2.03884487e+000 ;
    fparams[46][3] =  3.78924449e-001 ;
    fparams[46][4] =  1.82067264e-001 ;
    fparams[46][5] =  1.42921634e-001 ;
    fparams[46][6] =  9.52040948e-002 ;
    fparams[46][7] =  1.17125900e-001 ;
    fparams[46][8] =  5.91445248e-001 ;
    fparams[46][9] =  1.07843808e+000 ;
    fparams[46][10] =  1.13328676e-003 ;
    fparams[46][11] =  7.80252092e-003 ;
    fparams[47][0] =  2.07981390e+000 ;
    fparams[47][1] =  9.92540297e+000 ;
    fparams[47][2] =  4.43170726e-001 ;
    fparams[47][3] =  1.04920104e-001 ;
    fparams[47][4] =  1.96515215e+000 ;
    fparams[47][5] =  6.40103839e-001 ;
    fparams[47][6] =  5.96130591e-001 ;
    fparams[47][7] =  8.89594790e-001 ;
    fparams[47][8] =  4.78016333e-001 ;
    fparams[47][9] =  1.98509407e+000 ;
    fparams[47][10] =  9.46458470e-002 ;
    fparams[47][11] =  1.12744464e-001 ;
    fparams[48][0] =  1.63657549e+000 ;
    fparams[48][1] =  1.24540381e+001 ;
    fparams[48][2] =  2.17927989e+000 ;
    fparams[48][3] =  1.45134660e+000 ;
    fparams[48][4] =  7.71300690e-001 ;
    fparams[48][5] =  1.26695757e-001 ;
    fparams[48][6] =  6.64193880e-001 ;
    fparams[48][7] =  7.77659202e-001 ;
    fparams[48][8] =  7.64563285e-001 ;
    fparams[48][9] =  1.66075210e+000 ;
    fparams[48][10] =  8.61126689e-002 ;
    fparams[48][11] =  1.05728357e-001 ;
    fparams[49][0] =  2.24820632e+000 ;
    fparams[49][1] =  1.51913507e+000 ;
    fparams[49][2] =  1.64706864e+000 ;
    fparams[49][3] =  1.30113424e+001 ;
    fparams[49][4] =  7.88679265e-001 ;
    fparams[49][5] =  1.06128184e-001 ;
    fparams[49][6] =  8.12579069e-002 ;
    fparams[49][7] =  9.94045620e-002 ;
    fparams[49][8] =  6.68280346e-001 ;
    fparams[49][9] =  1.49742063e+000 ;
    fparams[49][10] =  6.38467475e-001 ;
    fparams[49][11] =  7.18422635e-001 ;
    fparams[50][0] =  2.16644620e+000 ;
    fparams[50][1] =  1.13174909e+001 ;
    fparams[50][2] =  6.88691021e-001 ;
    fparams[50][3] =  1.10131285e-001 ;
    fparams[50][4] =  1.92431751e+000 ;
    fparams[50][5] =  6.74464853e-001 ;
    fparams[50][6] =  5.65359888e-001 ;
    fparams[50][7] =  7.33564610e-001 ;
    fparams[50][8] =  9.18683861e-001 ;
    fparams[50][9] =  1.02310312e+001 ;
    fparams[50][10] =  7.80542213e-002 ;
    fparams[50][11] =  9.31104308e-002 ;
    fparams[51][0] =  1.73662114e+000 ;
    fparams[51][1] =  8.84334719e-001 ;
    fparams[51][2] =  9.99871380e-001 ;
    fparams[51][3] =  1.38462121e-001 ;
    fparams[51][4] =  2.13972409e+000 ;
    fparams[51][5] =  1.19666432e+001 ;
    fparams[51][6] =  5.60566526e-001 ;
    fparams[51][7] =  6.72672880e-001 ;
    fparams[51][8] =  9.93772747e-001 ;
    fparams[51][9] =  8.72330411e+000 ;
    fparams[51][10] =  7.37374982e-002 ;
    fparams[51][11] =  8.78577715e-002 ;
    fparams[52][0] =  2.09383882e+000 ;
    fparams[52][1] =  1.26856869e+001 ;
    fparams[52][2] =  1.56940519e+000 ;
    fparams[52][3] =  1.21236537e+000 ;
    fparams[52][4] =  1.30941993e+000 ;
    fparams[52][5] =  1.66633292e-001 ;
    fparams[52][6] =  6.98067804e-002 ;
    fparams[52][7] =  8.30817576e-002 ;
    fparams[52][8] =  1.04969537e+000 ;
    fparams[52][9] =  7.43147857e+000 ;
    fparams[52][10] =  5.55594354e-001 ;
    fparams[52][11] =  6.17487676e-001 ;
    fparams[53][0] =  1.60186925e+000 ;
    fparams[53][1] =  1.95031538e-001 ;
    fparams[53][2] =  1.98510264e+000 ;
    fparams[53][3] =  1.36976183e+001 ;
    fparams[53][4] =  1.48226200e+000 ;
    fparams[53][5] =  1.80304795e+000 ;
    fparams[53][6] =  5.53807199e-001 ;
    fparams[53][7] =  5.67912340e-001 ;
    fparams[53][8] =  1.11728722e+000 ;
    fparams[53][9] =  6.40879878e+000 ;
    fparams[53][10] =  6.60720847e-002 ;
    fparams[53][11] =  7.86615429e-002 ;
    fparams[54][0] =  1.60015487e+000 ;
    fparams[54][1] =  2.92913354e+000 ;
    fparams[54][2] =  1.71644581e+000 ;
    fparams[54][3] =  1.55882990e+001 ;
    fparams[54][4] =  1.84968351e+000 ;
    fparams[54][5] =  2.22525983e-001 ;
    fparams[54][6] =  6.23813648e-002 ;
    fparams[54][7] =  7.45581223e-002 ;
    fparams[54][8] =  1.21387555e+000 ;
    fparams[54][9] =  5.56013271e+000 ;
    fparams[54][10] =  5.54051946e-001 ;
    fparams[54][11] =  5.21994521e-001 ;
    fparams[55][0] =  2.95236854e+000 ;
    fparams[55][1] =  6.01461952e+000 ;
    fparams[55][2] =  4.28105721e-001 ;
    fparams[55][3] =  4.64151246e+001 ;
    fparams[55][4] =  1.89599233e+000 ;
    fparams[55][5] =  1.80109756e-001 ;
    fparams[55][6] =  5.48012938e-002 ;
    fparams[55][7] =  7.12799633e-002 ;
    fparams[55][8] =  4.70838600e+000 ;
    fparams[55][9] =  4.56702799e+001 ;
    fparams[55][10] =  5.90356719e-001 ;
    fparams[55][11] =  4.70236310e-001 ;
    fparams[56][0] =  3.19434243e+000 ;
    fparams[56][1] =  9.27352241e+000 ;
    fparams[56][2] =  1.98289586e+000 ;
    fparams[56][3] =  2.28741632e-001 ;
    fparams[56][4] =  1.55121052e-001 ;
    fparams[56][5] =  3.82000231e-002 ;
    fparams[56][6] =  6.73222354e-002 ;
    fparams[56][7] =  7.30961745e-002 ;
    fparams[56][8] =  4.48474211e+000 ;
    fparams[56][9] =  2.95703565e+001 ;
    fparams[56][10] =  5.42674414e-001 ;
    fparams[56][11] =  4.08647015e-001 ;
    fparams[57][0] =  2.05036425e+000 ;
    fparams[57][1] =  2.20348417e-001 ;
    fparams[57][2] =  1.42114311e-001 ;
    fparams[57][3] =  3.96438056e-002 ;
    fparams[57][4] =  3.23538151e+000 ;
    fparams[57][5] =  9.56979169e+000 ;
    fparams[57][6] =  6.34683429e-002 ;
    fparams[57][7] =  6.92443091e-002 ;
    fparams[57][8] =  3.97960586e+000 ;
    fparams[57][9] =  2.53178406e+001 ;
    fparams[57][10] =  5.20116711e-001 ;
    fparams[57][11] =  3.83614098e-001 ;
    fparams[58][0] =  3.22990759e+000 ;
    fparams[58][1] =  9.94660135e+000 ;
    fparams[58][2] =  1.57618307e-001 ;
    fparams[58][3] =  4.15378676e-002 ;
    fparams[58][4] =  2.13477838e+000 ;
    fparams[58][5] =  2.40480572e-001 ;
    fparams[58][6] =  5.01907609e-001 ;
    fparams[58][7] =  3.66252019e-001 ;
    fparams[58][8] =  3.80889010e+000 ;
    fparams[58][9] =  2.43275968e+001 ;
    fparams[58][10] =  5.96625028e-002 ;
    fparams[58][11] =  6.59653503e-002 ;
    fparams[59][0] =  1.58189324e-001 ;
    fparams[59][1] =  3.91309056e-002 ;
    fparams[59][2] =  3.18141995e+000 ;
    fparams[59][3] =  1.04139545e+001 ;
    fparams[59][4] =  2.27622140e+000 ;
    fparams[59][5] =  2.81671757e-001 ;
    fparams[59][6] =  3.97705472e+000 ;
    fparams[59][7] =  2.61872978e+001 ;
    fparams[59][8] =  5.58448277e-002 ;
    fparams[59][9] =  6.30921695e-002 ;
    fparams[59][10] =  4.85207954e-001 ;
    fparams[59][11] =  3.54234369e-001 ;
    fparams[60][0] =  1.81379417e-001 ;
    fparams[60][1] =  4.37324793e-002 ;
    fparams[60][2] =  3.17616396e+000 ;
    fparams[60][3] =  1.07842572e+001 ;
    fparams[60][4] =  2.35221519e+000 ;
    fparams[60][5] =  3.05571833e-001 ;
    fparams[60][6] =  3.83125763e+000 ;
    fparams[60][7] =  2.54745408e+001 ;
    fparams[60][8] =  5.25889976e-002 ;
    fparams[60][9] =  6.02676073e-002 ;
    fparams[60][10] =  4.70090742e-001 ;
    fparams[60][11] =  3.39017003e-001 ;
    fparams[61][0] =  1.92986811e-001 ;
    fparams[61][1] =  4.37785970e-002 ;
    fparams[61][2] =  2.43756023e+000 ;
    fparams[61][3] =  3.29336996e-001 ;
    fparams[61][4] =  3.17248504e+000 ;
    fparams[61][5] =  1.11259996e+001 ;
    fparams[61][6] =  3.58105414e+000 ;
    fparams[61][7] =  2.46709586e+001 ;
    fparams[61][8] =  4.56529394e-001 ;
    fparams[61][9] =  3.24990282e-001 ;
    fparams[61][10] =  4.94812177e-002 ;
    fparams[61][11] =  5.76553100e-002 ;
    fparams[62][0] =  2.12002595e-001 ;
    fparams[62][1] =  4.57703608e-002 ;
    fparams[62][2] =  3.16891754e+000 ;
    fparams[62][3] =  1.14536599e+001 ;
    fparams[62][4] =  2.51503494e+000 ;
    fparams[62][5] =  3.55561054e-001 ;
    fparams[62][6] =  4.44080845e-001 ;
    fparams[62][7] =  3.11953363e-001 ;
    fparams[62][8] =  3.36742101e+000 ;
    fparams[62][9] =  2.40291435e+001 ;
    fparams[62][10] =  4.65652543e-002 ;
    fparams[62][11] =  5.52266819e-002 ;
    fparams[63][0] =  2.59355002e+000 ;
    fparams[63][1] =  3.82452612e-001 ;
    fparams[63][2] =  3.16557522e+000 ;
    fparams[63][3] =  1.17675155e+001 ;
    fparams[63][4] =  2.29402652e-001 ;
    fparams[63][5] =  4.76642249e-002 ;
    fparams[63][6] =  4.32257780e-001 ;
    fparams[63][7] =  2.99719833e-001 ;
    fparams[63][8] =  3.17261920e+000 ;
    fparams[63][9] =  2.34462738e+001 ;
    fparams[63][10] =  4.37958317e-002 ;
    fparams[63][11] =  5.29440680e-002 ;
    fparams[64][0] =  3.19144939e+000 ;
    fparams[64][1] =  1.20224655e+001 ;
    fparams[64][2] =  2.55766431e+000 ;
    fparams[64][3] =  4.08338876e-001 ;
    fparams[64][4] =  3.32681934e-001 ;
    fparams[64][5] =  5.85819814e-002 ;
    fparams[64][6] =  4.14243130e-002 ;
    fparams[64][7] =  5.06771477e-002 ;
    fparams[64][8] =  2.61036728e+000 ;
    fparams[64][9] =  1.99344244e+001 ;
    fparams[64][10] =  4.20526863e-001 ;
    fparams[64][11] =  2.85686240e-001 ;
    fparams[65][0] =  2.59407462e-001 ;
    fparams[65][1] =  5.04689354e-002 ;
    fparams[65][2] =  3.16177855e+000 ;
    fparams[65][3] =  1.23140183e+001 ;
    fparams[65][4] =  2.75095751e+000 ;
    fparams[65][5] =  4.38337626e-001 ;
    fparams[65][6] =  2.79247686e+000 ;
    fparams[65][7] =  2.23797309e+001 ;
    fparams[65][8] =  3.85931001e-002 ;
    fparams[65][9] =  4.87920992e-002 ;
    fparams[65][10] =  4.10881708e-001 ;
    fparams[65][11] =  2.77622892e-001 ;
    fparams[66][0] =  3.16055396e+000 ;
    fparams[66][1] =  1.25470414e+001 ;
    fparams[66][2] =  2.82751709e+000 ;
    fparams[66][3] =  4.67899094e-001 ;
    fparams[66][4] =  2.75140255e-001 ;
    fparams[66][5] =  5.23226982e-002 ;
    fparams[66][6] =  4.00967160e-001 ;
    fparams[66][7] =  2.67614884e-001 ;
    fparams[66][8] =  2.63110834e+000 ;
    fparams[66][9] =  2.19498166e+001 ;
    fparams[66][10] =  3.61333817e-002 ;
    fparams[66][11] =  4.68871497e-002 ;
    fparams[67][0] =  2.88642467e-001 ;
    fparams[67][1] =  5.40507687e-002 ;
    fparams[67][2] =  2.90567296e+000 ;
    fparams[67][3] =  4.97581077e-001 ;
    fparams[67][4] =  3.15960159e+000 ;
    fparams[67][5] =  1.27599505e+001 ;
    fparams[67][6] =  3.91280259e-001 ;
    fparams[67][7] =  2.58151831e-001 ;
    fparams[67][8] =  2.48596038e+000 ;
    fparams[67][9] =  2.15400972e+001 ;
    fparams[67][10] =  3.37664478e-002 ;
    fparams[67][11] =  4.50664323e-002 ;
    fparams[68][0] =  3.15573213e+000 ;
    fparams[68][1] =  1.29729009e+001 ;
    fparams[68][2] =  3.11519560e-001 ;
    fparams[68][3] =  5.81399387e-002 ;
    fparams[68][4] =  2.97722406e+000 ;
    fparams[68][5] =  5.31213394e-001 ;
    fparams[68][6] =  3.81563854e-001 ;
    fparams[68][7] =  2.49195776e-001 ;
    fparams[68][8] =  2.40247532e+000 ;
    fparams[68][9] =  2.13627616e+001 ;
    fparams[68][10] =  3.15224214e-002 ;
    fparams[68][11] =  4.33253257e-002 ;
    fparams[69][0] =  3.15591970e+000 ;
    fparams[69][1] =  1.31232407e+001 ;
    fparams[69][2] =  3.22544710e-001 ;
    fparams[69][3] =  5.97223323e-002 ;
    fparams[69][4] =  3.05569053e+000 ;
    fparams[69][5] =  5.61876773e-001 ;
    fparams[69][6] =  2.92845100e-002 ;
    fparams[69][7] =  4.16534255e-002 ;
    fparams[69][8] =  3.72487205e-001 ;
    fparams[69][9] =  2.40821967e-001 ;
    fparams[69][10] =  2.27833695e+000 ;
    fparams[69][11] =  2.10034185e+001 ;
    fparams[70][0] =  3.10794704e+000 ;
    fparams[70][1] =  6.06347847e-001 ;
    fparams[70][2] =  3.14091221e+000 ;
    fparams[70][3] =  1.33705269e+001 ;
    fparams[70][4] =  3.75660454e-001 ;
    fparams[70][5] =  7.29814740e-002 ;
    fparams[70][6] =  3.61901097e-001 ;
    fparams[70][7] =  2.32652051e-001 ;
    fparams[70][8] =  2.45409082e+000 ;
    fparams[70][9] =  2.12695209e+001 ;
    fparams[70][10] =  2.72383990e-002 ;
    fparams[70][11] =  3.99969597e-002 ;
    fparams[71][0] =  3.11446863e+000 ;
    fparams[71][1] =  1.38968881e+001 ;
    fparams[71][2] =  5.39634353e-001 ;
    fparams[71][3] =  8.91708508e-002 ;
    fparams[71][4] =  3.06460915e+000 ;
    fparams[71][5] =  6.79919563e-001 ;
    fparams[71][6] =  2.58563745e-002 ;
    fparams[71][7] =  3.82808522e-002 ;
    fparams[71][8] =  2.13983556e+000 ;
    fparams[71][9] =  1.80078788e+001 ;
    fparams[71][10] =  3.47788231e-001 ;
    fparams[71][11] =  2.22706591e-001 ;
    fparams[72][0] =  3.01166899e+000 ;
    fparams[72][1] =  7.10401889e-001 ;
    fparams[72][2] =  3.16284788e+000 ;
    fparams[72][3] =  1.38262192e+001 ;
    fparams[72][4] =  6.33421771e-001 ;
    fparams[72][5] =  9.48486572e-002 ;
    fparams[72][6] =  3.41417198e-001 ;
    fparams[72][7] =  2.14129678e-001 ;
    fparams[72][8] =  1.53566013e+000 ;
    fparams[72][9] =  1.55298698e+001 ;
    fparams[72][10] =  2.40723773e-002 ;
    fparams[72][11] =  3.67833690e-002 ;
    fparams[73][0] =  3.20236821e+000 ;
    fparams[73][1] =  1.38446369e+001 ;
    fparams[73][2] =  8.30098413e-001 ;
    fparams[73][3] =  1.18381581e-001 ;
    fparams[73][4] =  2.86552297e+000 ;
    fparams[73][5] =  7.66369118e-001 ;
    fparams[73][6] =  2.24813887e-002 ;
    fparams[73][7] =  3.52934622e-002 ;
    fparams[73][8] =  1.40165263e+000 ;
    fparams[73][9] =  1.46148877e+001 ;
    fparams[73][10] =  3.33740596e-001 ;
    fparams[73][11] =  2.05704486e-001 ;
    fparams[74][0] =  9.24906855e-001 ;
    fparams[74][1] =  1.28663377e-001 ;
    fparams[74][2] =  2.75554557e+000 ;
    fparams[74][3] =  7.65826479e-001 ;
    fparams[74][4] =  3.30440060e+000 ;
    fparams[74][5] =  1.34471170e+001 ;
    fparams[74][6] =  3.29973862e-001 ;
    fparams[74][7] =  1.98218895e-001 ;
    fparams[74][8] =  1.09916444e+000 ;
    fparams[74][9] =  1.35087534e+001 ;
    fparams[74][10] =  2.06498883e-002 ;
    fparams[74][11] =  3.38918459e-002 ;
    fparams[75][0] =  1.96952105e+000 ;
    fparams[75][1] =  4.98830620e+001 ;
    fparams[75][2] =  1.21726619e+000 ;
    fparams[75][3] =  1.33243809e-001 ;
    fparams[75][4] =  4.10391685e+000 ;
    fparams[75][5] =  1.84396916e+000 ;
    fparams[75][6] =  2.90791978e-002 ;
    fparams[75][7] =  2.84192813e-002 ;
    fparams[75][8] =  2.30696669e-001 ;
    fparams[75][9] =  1.90968784e-001 ;
    fparams[75][10] =  6.08840299e-001 ;
    fparams[75][11] =  1.37090356e+000 ;
    fparams[76][0] =  2.06385867e+000 ;
    fparams[76][1] =  4.05671697e+001 ;
    fparams[76][2] =  1.29603406e+000 ;
    fparams[76][3] =  1.46559047e-001 ;
    fparams[76][4] =  3.96920673e+000 ;
    fparams[76][5] =  1.82561596e+000 ;
    fparams[76][6] =  2.69835487e-002 ;
    fparams[76][7] =  2.84172045e-002 ;
    fparams[76][8] =  2.31083999e-001 ;
    fparams[76][9] =  1.79765184e-001 ;
    fparams[76][10] =  6.30466774e-001 ;
    fparams[76][11] =  1.38911543e+000 ;
    fparams[77][0] =  2.21522726e+000 ;
    fparams[77][1] =  3.24464090e+001 ;
    fparams[77][2] =  1.37573155e+000 ;
    fparams[77][3] =  1.60920048e-001 ;
    fparams[77][4] =  3.78244405e+000 ;
    fparams[77][5] =  1.78756553e+000 ;
    fparams[77][6] =  2.44643240e-002 ;
    fparams[77][7] =  2.82909938e-002 ;
    fparams[77][8] =  2.36932016e-001 ;
    fparams[77][9] =  1.70692368e-001 ;
    fparams[77][10] =  6.48471412e-001 ;
    fparams[77][11] =  1.37928390e+000 ;
    fparams[78][0] =  9.84697940e-001 ;
    fparams[78][1] =  1.60910839e-001 ;
    fparams[78][2] =  2.73987079e+000 ;
    fparams[78][3] =  7.18971667e-001 ;
    fparams[78][4] =  3.61696715e+000 ;
    fparams[78][5] =  1.29281016e+001 ;
    fparams[78][6] =  3.02885602e-001 ;
    fparams[78][7] =  1.70134854e-001 ;
    fparams[78][8] =  2.78370726e-001 ;
    fparams[78][9] =  1.49862703e+000 ;
    fparams[78][10] =  1.52124129e-002 ;
    fparams[78][11] =  2.83510822e-002 ;
    fparams[79][0] =  9.61263398e-001 ;
    fparams[79][1] =  1.70932277e-001 ;
    fparams[79][2] =  3.69581030e+000 ;
    fparams[79][3] =  1.29335319e+001 ;
    fparams[79][4] =  2.77567491e+000 ;
    fparams[79][5] =  6.89997070e-001 ;
    fparams[79][6] =  2.95414176e-001 ;
    fparams[79][7] =  1.63525510e-001 ;
    fparams[79][8] =  3.11475743e-001 ;
    fparams[79][9] =  1.39200901e+000 ;
    fparams[79][10] =  1.43237267e-002 ;
    fparams[79][11] =  2.71265337e-002 ;
    fparams[80][0] =  1.29200491e+000 ;
    fparams[80][1] =  1.83432865e-001 ;
    fparams[80][2] =  2.75161478e+000 ;
    fparams[80][3] =  9.42368371e-001 ;
    fparams[80][4] =  3.49387949e+000 ;
    fparams[80][5] =  1.46235654e+001 ;
    fparams[80][6] =  2.77304636e-001 ;
    fparams[80][7] =  1.55110144e-001 ;
    fparams[80][8] =  4.30232810e-001 ;
    fparams[80][9] =  1.28871670e+000 ;
    fparams[80][10] =  1.48294351e-002 ;
    fparams[80][11] =  2.61903834e-002 ;
    fparams[81][0] =  3.75964730e+000 ;
    fparams[81][1] =  1.35041513e+001 ;
    fparams[81][2] =  3.21195904e+000 ;
    fparams[81][3] =  6.66330993e-001 ;
    fparams[81][4] =  6.47767825e-001 ;
    fparams[81][5] =  9.22518234e-002 ;
    fparams[81][6] =  2.76123274e-001 ;
    fparams[81][7] =  1.50312897e-001 ;
    fparams[81][8] =  3.18838810e-001 ;
    fparams[81][9] =  1.12565588e+000 ;
    fparams[81][10] =  1.31668419e-002 ;
    fparams[81][11] =  2.48879842e-002 ;
    fparams[82][0] =  1.00795975e+000 ;
    fparams[82][1] =  1.17268427e-001 ;
    fparams[82][2] =  3.09796153e+000 ;
    fparams[82][3] =  8.80453235e-001 ;
    fparams[82][4] =  3.61296864e+000 ;
    fparams[82][5] =  1.47325812e+001 ;
    fparams[82][6] =  2.62401476e-001 ;
    fparams[82][7] =  1.43491014e-001 ;
    fparams[82][8] =  4.05621995e-001 ;
    fparams[82][9] =  1.04103506e+000 ;
    fparams[82][10] =  1.31812509e-002 ;
    fparams[82][11] =  2.39575415e-002 ;
    fparams[83][0] =  1.59826875e+000 ;
    fparams[83][1] =  1.56897471e-001 ;
    fparams[83][2] =  4.38233925e+000 ;
    fparams[83][3] =  2.47094692e+000 ;
    fparams[83][4] =  2.06074719e+000 ;
    fparams[83][5] =  5.72438972e+001 ;
    fparams[83][6] =  1.94426023e-001 ;
    fparams[83][7] =  1.32979109e-001 ;
    fparams[83][8] =  8.22704978e-001 ;
    fparams[83][9] =  9.56532528e-001 ;
    fparams[83][10] =  2.33226953e-002 ;
    fparams[83][11] =  2.23038435e-002 ;
    fparams[84][0] =  1.71463223e+000 ;
    fparams[84][1] =  9.79262841e+001 ;
    fparams[84][2] =  2.14115960e+000 ;
    fparams[84][3] =  2.10193717e-001 ;
    fparams[84][4] =  4.37512413e+000 ;
    fparams[84][5] =  3.66948812e+000 ;
    fparams[84][6] =  2.16216680e-002 ;
    fparams[84][7] =  1.98456144e-002 ;
    fparams[84][8] =  1.97843837e-001 ;
    fparams[84][9] =  1.33758807e-001 ;
    fparams[84][10] =  6.52047920e-001 ;
    fparams[84][11] =  7.80432104e-001 ;
    fparams[85][0] =  1.48047794e+000 ;
    fparams[85][1] =  1.25943919e+002 ;
    fparams[85][2] =  2.09174630e+000 ;
    fparams[85][3] =  1.83803008e-001 ;
    fparams[85][4] =  4.75246033e+000 ;
    fparams[85][5] =  4.19890596e+000 ;
    fparams[85][6] =  1.85643958e-002 ;
    fparams[85][7] =  1.81383503e-002 ;
    fparams[85][8] =  2.05859375e-001 ;
    fparams[85][9] =  1.33035404e-001 ;
    fparams[85][10] =  7.13540948e-001 ;
    fparams[85][11] =  7.03031938e-001 ;
    fparams[86][0] =  6.30022295e-001 ;
    fparams[86][1] =  1.40909762e-001 ;
    fparams[86][2] =  3.80962881e+000 ;
    fparams[86][3] =  3.08515540e+001 ;
    fparams[86][4] =  3.89756067e+000 ;
    fparams[86][5] =  6.51559763e-001 ;
    fparams[86][6] =  2.40755100e-001 ;
    fparams[86][7] =  1.08899672e-001 ;
    fparams[86][8] =  2.62868577e+000 ;
    fparams[86][9] =  6.42383261e+000 ;
    fparams[86][10] =  3.14285931e-002 ;
    fparams[86][11] =  2.42346699e-002 ;
    fparams[87][0] =  5.23288135e+000 ;
    fparams[87][1] =  8.60599536e+000 ;
    fparams[87][2] =  2.48604205e+000 ;
    fparams[87][3] =  3.04543982e-001 ;
    fparams[87][4] =  3.23431354e-001 ;
    fparams[87][5] =  3.87759096e-002 ;
    fparams[87][6] =  2.55403596e-001 ;
    fparams[87][7] =  1.28717724e-001 ;
    fparams[87][8] =  5.53607228e-001 ;
    fparams[87][9] =  5.36977452e-001 ;
    fparams[87][10] =  5.75278889e-003 ;
    fparams[87][11] =  1.29417790e-002 ;
    fparams[88][0] =  1.44192685e+000 ;
    fparams[88][1] =  1.18740873e-001 ;
    fparams[88][2] =  3.55291725e+000 ;
    fparams[88][3] =  1.01739750e+000 ;
    fparams[88][4] =  3.91259586e+000 ;
    fparams[88][5] =  6.31814783e+001 ;
    fparams[88][6] =  2.16173519e-001 ;
    fparams[88][7] =  9.55806441e-002 ;
    fparams[88][8] =  3.94191605e+000 ;
    fparams[88][9] =  3.50602732e+001 ;
    fparams[88][10] =  4.60422605e-002 ;
    fparams[88][11] =  2.20850385e-002 ;
    fparams[89][0] =  1.45864127e+000 ;
    fparams[89][1] =  1.07760494e-001 ;
    fparams[89][2] =  4.18945405e+000 ;
    fparams[89][3] =  8.89090649e+001 ;
    fparams[89][4] =  3.65866182e+000 ;
    fparams[89][5] =  1.05088931e+000 ;
    fparams[89][6] =  2.08479229e-001 ;
    fparams[89][7] =  9.09335557e-002 ;
    fparams[89][8] =  3.16528117e+000 ;
    fparams[89][9] =  3.13297788e+001 ;
    fparams[89][10] =  5.23892556e-002 ;
    fparams[89][11] =  2.08807697e-002 ;
    fparams[90][0] =  1.19014064e+000 ;
    fparams[90][1] =  7.73468729e-002 ;
    fparams[90][2] =  2.55380607e+000 ;
    fparams[90][3] =  6.59693681e-001 ;
    fparams[90][4] =  4.68110181e+000 ;
    fparams[90][5] =  1.28013896e+001 ;
    fparams[90][6] =  2.26121303e-001 ;
    fparams[90][7] =  1.08632194e-001 ;
    fparams[90][8] =  3.58250545e-001 ;
    fparams[90][9] =  4.56765664e-001 ;
    fparams[90][10] =  7.82263950e-003 ;
    fparams[90][11] =  1.62623474e-002 ;
    fparams[91][0] =  4.68537504e+000 ;
    fparams[91][1] =  1.44503632e+001 ;
    fparams[91][2] =  2.98413708e+000 ;
    fparams[91][3] =  5.56438592e-001 ;
    fparams[91][4] =  8.91988061e-001 ;
    fparams[91][5] =  6.69512914e-002 ;
    fparams[91][6] =  2.24825384e-001 ;
    fparams[91][7] =  1.03235396e-001 ;
    fparams[91][8] =  3.04444846e-001 ;
    fparams[91][9] =  4.27255647e-001 ;
    fparams[91][10] =  9.48162708e-003 ;
    fparams[91][11] =  1.77730611e-002 ;
    fparams[92][0] =  4.63343606e+000 ;
    fparams[92][1] =  1.63377267e+001 ;
    fparams[92][2] =  3.18157056e+000 ;
    fparams[92][3] =  5.69517868e-001 ;
    fparams[92][4] =  8.76455075e-001 ;
    fparams[92][5] =  6.88860012e-002 ;
    fparams[92][6] =  2.21685477e-001 ;
    fparams[92][7] =  9.84254550e-002 ;
    fparams[92][8] =  2.72917100e-001 ;
    fparams[92][9] =  4.09470917e-001 ;
    fparams[92][10] =  1.11737298e-002 ;
    fparams[92][11] =  1.86215410e-002 ;
    fparams[93][0] =  4.56773888e+000 ;
    fparams[93][1] =  1.90992795e+001 ;
    fparams[93][2] =  3.40325179e+000 ;
    fparams[93][3] =  5.90099634e-001 ;
    fparams[93][4] =  8.61841923e-001 ;
    fparams[93][5] =  7.03204851e-002 ;
    fparams[93][6] =  2.19728870e-001 ;
    fparams[93][7] =  9.36334280e-002 ;
    fparams[93][8] =  2.38176903e-001 ;
    fparams[93][9] =  3.93554882e-001 ;
    fparams[93][10] =  1.38306499e-002 ;
    fparams[93][11] =  1.94437286e-002 ;
    fparams[94][0] =  5.45671123e+000 ;
    fparams[94][1] =  1.01892720e+001 ;
    fparams[94][2] =  1.11687906e-001 ;
    fparams[94][3] =  3.98131313e-002 ;
    fparams[94][4] =  3.30260343e+000 ;
    fparams[94][5] =  3.14622212e-001 ;
    fparams[94][6] =  1.84568319e-001 ;
    fparams[94][7] =  1.04220860e-001 ;
    fparams[94][8] =  4.93644263e-001 ;
    fparams[94][9] =  4.63080540e-001 ;
    fparams[94][10] =  3.57484743e+000 ;
    fparams[94][11] =  2.19369542e+001 ;
    fparams[95][0] =  5.38321999e+000 ;
    fparams[95][1] =  1.07289857e+001 ;
    fparams[95][2] =  1.23343236e-001 ;
    fparams[95][3] =  4.15137806e-002 ;
    fparams[95][4] =  3.46469090e+000 ;
    fparams[95][5] =  3.39326208e-001 ;
    fparams[95][6] =  1.75437132e-001 ;
    fparams[95][7] =  9.98932346e-002 ;
    fparams[95][8] =  3.39800073e+000 ;
    fparams[95][9] =  2.11601535e+001 ;
    fparams[95][10] =  4.69459519e-001 ;
    fparams[95][11] =  4.51996970e-001 ;
    fparams[96][0] =  5.38402377e+000 ;
    fparams[96][1] =  1.11211419e+001 ;
    fparams[96][2] =  3.49861264e+000 ;
    fparams[96][3] =  3.56750210e-001 ;
    fparams[96][4] =  1.88039547e-001 ;
    fparams[96][5] =  5.39853583e-002 ;
    fparams[96][6] =  1.69143137e-001 ;
    fparams[96][7] =  9.60082633e-002 ;
    fparams[96][8] =  3.19595016e+000 ;
    fparams[96][9] =  1.80694389e+001 ;
    fparams[96][10] =  4.64393059e-001 ;
    fparams[96][11] =  4.36318197e-001 ;
    fparams[97][0] =  3.66090688e+000 ;
    fparams[97][1] =  3.84420906e-001 ;
    fparams[97][2] =  2.03054678e-001 ;
    fparams[97][3] =  5.48547131e-002 ;
    fparams[97][4] =  5.30697515e+000 ;
    fparams[97][5] =  1.17150262e+001 ;
    fparams[97][6] =  1.60934046e-001 ;
    fparams[97][7] =  9.21020329e-002 ;
    fparams[97][8] =  3.04808401e+000 ;
    fparams[97][9] =  1.73525367e+001 ;
    fparams[97][10] =  4.43610295e-001 ;
    fparams[97][11] =  4.27132359e-001 ;
    fparams[98][0] =  3.94150390e+000 ;
    fparams[98][1] =  4.18246722e-001 ;
    fparams[98][2] =  5.16915345e+000 ;
    fparams[98][3] =  1.25201788e+001 ;
    fparams[98][4] =  1.61941074e-001 ;
    fparams[98][5] =  4.81540117e-002 ;
    fparams[98][6] =  4.15299561e-001 ;
    fparams[98][7] =  4.24913856e-001 ;
    fparams[98][8] =  2.91761325e+000 ;
    fparams[98][9] =  1.90899693e+001 ;
    fparams[98][10] =  1.51474927e-001 ;
    fparams[98][11] =  8.81568925e-002 ;
    fparams[99][0] =  4.09780623e+000 ;
    fparams[99][1] =  4.46021145e-001 ;
    fparams[99][2] =  5.10079393e+000 ;
    fparams[99][3] =  1.31768613e+001 ;
    fparams[99][4] =  1.74617289e-001 ;
    fparams[99][5] =  5.02742829e-002 ;
    fparams[99][6] =  2.76774658e+000 ;
    fparams[99][7] =  1.84815393e+001 ;
    fparams[99][8] =  1.44496639e-001 ;
    fparams[99][9] =  8.46232592e-002 ;
    fparams[99][10] =  4.02772109e-001 ;
    fparams[99][11] =  4.17640100e-001 ;
    fparams[100][0] =  4.24934820e+000 ;
    fparams[100][1] =  4.75263933e-001 ;
    fparams[100][2] =  5.03556594e+000 ;
    fparams[100][3] =  1.38570834e+001 ;
    fparams[100][4] =  1.88920613e-001 ;
    fparams[100][5] =  5.26975158e-002 ;
    fparams[100][6] =  3.94356058e-001 ;
    fparams[100][7] =  4.11193751e-001 ;
    fparams[100][8] =  2.61213100e+000 ;
    fparams[100][9] =  1.78537905e+001 ;
    fparams[100][10] =  1.38001927e-001 ;
    fparams[100][11] =  8.12774434e-002 ;
    fparams[101][0] =  2.00942931e-001 ;
    fparams[101][1] =  5.48366518e-002 ;
    fparams[101][2] =  4.40119869e+000 ;
    fparams[101][3] =  5.04248434e-001 ;
    fparams[101][4] =  4.97250102e+000 ;
    fparams[101][5] =  1.45721366e+001 ;
    fparams[101][6] =  2.47530599e+000 ;
    fparams[101][7] =  1.72978308e+001 ;
    fparams[101][8] =  3.86883197e-001 ;
    fparams[101][9] =  4.05043898e-001 ;
    fparams[101][10] =  1.31936095e-001 ;
    fparams[101][11] =  7.80821071e-002 ;
    fparams[102][0] =  2.16052899e-001 ;
    fparams[102][1] =  5.83584058e-002 ;
    fparams[102][2] =  4.91106799e+000 ;
    fparams[102][3] =  1.53264212e+001 ;
    fparams[102][4] =  4.54862870e+000 ;
    fparams[102][5] =  5.34434760e-001 ;
    fparams[102][6] =  2.36114249e+000 ;
    fparams[102][7] =  1.68164803e+001 ;
    fparams[102][8] =  1.26277292e-001 ;
    fparams[102][9] =  7.50304633e-002 ;
    fparams[102][10] =  3.81364501e-001 ;
    fparams[102][11] =  3.99305852e-001 ;
    fparams[103][0] =  4.86738014e+000 ;
    fparams[103][1] =  1.60320520e+001 ;
    fparams[103][2] =  3.19974401e-001 ;
    fparams[103][3] =  6.70871138e-002 ;
    fparams[103][4] =  4.58872425e+000 ;
    fparams[103][5] =  5.77039373e-001 ;
    fparams[103][6] =  1.21482448e-001 ;
    fparams[103][7] =  7.22275899e-002 ;
    fparams[103][8] =  2.31639872e+000 ;
    fparams[103][9] =  1.41279737e+001 ;
    fparams[103][10] =  3.79258137e-001 ;
    fparams[103][11] =  3.89973484e-001 ;
    
    feTableRead = 1; /* remember that table has been read */
    return ((NZMAX-NZMIN+1)*NPMAX);
    
}

double bessi0( double x ) {
    int i;
    double ax, sum, t;
    
    double i0a[] = { 1.0, 3.5156229, 3.0899424, 1.2067492,
        0.2659732, 0.0360768, 0.0045813 };
    
    double i0b[] = { 0.39894228, 0.01328592, 0.00225319,
        -0.00157565, 0.00916281, -0.02057706, 0.02635537,
        -0.01647633, 0.00392377};
    
    ax = fabs( x );
    if( ax <= 3.75 ) {
        t = x / 3.75;
        t = t * t;
        sum = i0a[6];
        for( i=5; i>=0; i--) sum = sum*t + i0a[i];
    } else {
        t = 3.75 / ax;
        sum = i0b[8];
        for( i=7; i>=0; i--) sum = sum*t + i0b[i];
        sum = exp( ax ) * sum / sqrt( ax );
    }
    return sum;
}

double bessk0( double x ) {
    double bessi0(double);
    
    int i;
    double ax, x2, sum;
    double k0a[] = { -0.57721566, 0.42278420, 0.23069756,
        0.03488590, 0.00262698, 0.00010750, 0.00000740};
    
    double k0b[] = { 1.25331414, -0.07832358, 0.02189568,
        -0.01062446, 0.00587872, -0.00251540, 0.00053208};
    
    ax = fabs( x );
    if( (ax > 0.0)  && ( ax <=  2.0 ) ){
        x2 = ax/2.0;
        x2 = x2 * x2;
        sum = k0a[6];
        for( i=5; i>=0; i--) sum = sum*x2 + k0a[i];
        sum = -log(ax/2.0) * bessi0(x) + sum;
    } else if( ax > 2.0 ) {
        x2 = 2.0/ax;
        sum = k0b[6];
        for( i=5; i>=0; i--) sum = sum*x2 + k0b[i];
        sum = exp( -ax ) * sum / sqrt( ax );
    } else sum = 1.0e20;
    return sum;
    
}

double vzatom( int Z, double radius ) {
    int i, nfe;
    double suml, sumg, x, r;
    
    /* Lorenzian, Gaussian constants */
    const double al=300.8242834, ag=150.4121417;
    const double pi=3.141592654;
    
    if( (Z<NZMIN) || (Z>NZMAX) ) return( 0.0 );
    
    /* read in the table from a file if this is the
     first time this is called */
    if( feTableRead == 0 ) nfe = ReadfeTable();
    
    r = fabs( radius );
    if( r < 1.0e-10 ) r = 1.0e-10;  /* avoid singularity at r=0 */
    suml = sumg = 0.0;
    
    /* Lorenztians */
    x = 2.0*pi*r;
    for( i=0; i<2*nl; i+=2 )
        suml += fparams[Z][i]* bessk0( x*sqrt(fparams[Z][i+1]) );
    
    /* Gaussians */
    x = pi*r;
    x = x*x;
    for( i=2*nl; i<2*(nl+ng); i+=2 )
        sumg += fparams[Z][i]*exp(-x/fparams[Z][i+1]) / fparams[Z][i+1];
    
    return( al*suml + ag*sumg );
    
}

void splinh(double x[], double y[],
            double b[], double c[], double d[], int n) {
#define SMALL 1.0e-25
    
    int i, nm1, nm4;
    double m1, m2, m3, m4, m5, t1, t2, m54, m43, m32, m21, x43;
    
    if( n < 4) return;
    
    /* Do the first end point (special case),
     and get starting values */
    
    m5 = ( y[3] - y[2] ) / ( x[3] - x[2] ); /* mx = slope at pt x */
    m4 = ( y[2] - y[1] ) / ( x[2] - x[1] );
    m3 = ( y[1] - y[0] ) / ( x[1] - x[0] );
    
    m2 = m3 + m3 - m4;  /* eq. (9) of reference [1] */
    m1 = m2 + m2 - m3;
    
    m54 = fabs( m5 - m4);
    m43 = fabs( m4 - m3);
    m32 = fabs( m3 - m2);
    m21 = fabs( m2 - m1);
    
    if ( (m43+m21) > SMALL )
        t1 = ( m43*m2 + m21*m3 ) / ( m43 + m21 );
    else
        t1 = 0.5 * ( m2 + m3 );
    
    /*  Do everything up to the last end points */
    
    nm1 = n-1;
    nm4 = n-4;
    
    for( i=0; i<nm1; i++) {
        
        if( (m54+m32) > SMALL )
            t2= (m54*m3 + m32*m4) / (m54 + m32);
        else
            t2 = 0.5* ( m3 + m4 );
        
        x43 = x[i+1] - x[i];
        b[i] = t1;
        c[i] = ( 3.0*m3 - t1 - t1 - t2 ) /x43;
        d[i] = ( t1 + t2 - m3 - m3 ) / ( x43*x43 );
        
        m1 = m2;
        m2 = m3;
        m3 = m4;
        m4 = m5;
        if( i < nm4 ) {
            m5 = ( y[i+4] - y[i+3] ) / ( x[i+4] - x[i+3] );
        } else {
            m5 = m4 + m4 - m3;
        }
        
        m21 = m32;
        m32 = m43;
        m43 = m54;
        m54 = fabs( m5 - m4 );
        t1 = t2;
    }
}

double seval(double *x, double *y, double *b, double *c,
             double *d, int n, double x0) {
    int i, j, k;
    double z, seval1;
    
    /*  exit if x0 is outside the spline range */
    if( x0 <= x[0] ) i = 0;
    else if( x0 >= x[n-2] ) i = n-2;
    else {
        i = 0;
        j = n;
        do{ k = ( i + j ) / 2 ;
            if( x0 < x[k] )  j = k;
            else if( x0 >= x[k] ) i = k;
        } while ( (j-i) > 1 );
    }
    
    z = x0 - x[i];
    seval1 = y[i] + ( b[i] + ( c[i] + d[i] *z ) *z) *z;
    
    return( seval1 );
    
}

double vzatomLUT(int Z, double rsq) {
    int i, iz;
    double dlnr, vz, r;
    const static double RMIN= 0.01;    /* r (in Ang) range of LUT for vzatomLUT() */
    const static double RMAX= 5.0;
    const static int NRMAX= 100;       /* number of in look-up-table in vzatomLUT */
    
    /* spline interpolation coeff. */
    static int splineInit=0, *nspline;
    static double  *splinx, **spliny, **splinb, **splinc, **splind;
    
    if( splineInit == 0 ) {
        splinx = (double*)malloc(NRMAX * sizeof(double));
        
        spliny = (double**) malloc(NZMAX * NRMAX * sizeof(double*));
        for (int i = 0; i < NZMAX; i++)
            spliny[i] = (double *)malloc(NRMAX * sizeof(double));
        
        splinb = (double**) malloc(NZMAX * NRMAX * sizeof(double*));
        for (int i = 0; i < NZMAX; i++)
            splinb[i] = (double *)malloc(NRMAX * sizeof(double));
        
        splinc = (double**) malloc(NZMAX * NRMAX * sizeof(double*));
        for (int i = 0; i < NZMAX; i++)
            splinc[i] = (double *)malloc(NRMAX * sizeof(double));
        
        splind = (double**) malloc(NZMAX * NRMAX * sizeof(double*));
        for (int i = 0; i < NZMAX; i++)
            splind[i] = (double *)malloc(NRMAX * sizeof(double));
        
        
        /*  generate a set of logarithmic r values */
        dlnr = log(RMAX/RMIN)/(NRMAX-1);
        for( i=0; i<NRMAX; i++)
            splinx[i] = RMIN * exp( i * dlnr );

        for( i=0; i<NRMAX; i++)
            splinx[i] = splinx[i] * splinx[i];   /* use r^2 not r for speed */
        
        nspline = (int*) malloc(NZMAX * sizeof(int));
        for( i=0; i<NZMAX; i++) nspline[i] = 0;
        splineInit = 1;     /* remember that this has been done */
    }
    
    iz = Z - 1;  /* convert atomic number to array index */
    if( (Z < 1) || ( Z > NZMAX) ) {
        exit( 0 );
    }
    
    /* if this atomic number has not been called before
     generate the spline coefficients */
    if( nspline[iz] == 0 ) {
        
        for( i=0; i<NRMAX; i++) {
            r = sqrt( splinx[i] );
            spliny[iz][i] = vzatom(Z, r);
        }
        nspline[iz] = NRMAX;
        splinh( splinx, spliny[iz], splinb[iz],
               splinc[iz], splind[iz], NRMAX);
    }
    
    /* now that everything is set up find the
     scattering factor by interpolation in the table
     */
    
    vz = seval( splinx, spliny[iz], splinb[iz],
               splinc[iz], splind[iz], nspline[iz], rsq );
    
    return vz;
}

#pragma mark - helper functions

void initOpenCL(cl_device_id device_id) {
    
    cl_int success;
    
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &success);
    command_queue = clCreateCommandQueue(context, device_id, 0, &success);
    copy_queue = clCreateCommandQueue(context, device_id, 0, &success);
        
    size_t dimensions[2], p_dimensions[2];
    dimensions[0] = nx;
    dimensions[1] = ny;
    
    p_dimensions[0] = probe_nx;
    p_dimensions[1] = probe_ny;
    
    clfftStatus status;
    status = clfftCreateDefaultPlan(&plan, context, CLFFT_2D, dimensions);
    status = clfftCreateDefaultPlan(&probe_plan, context, CLFFT_2D, p_dimensions);
    clfftSetPlanPrecision(plan, CLFFT_SINGLE);
    clfftSetPlanPrecision(probe_plan, CLFFT_SINGLE);
    clfftSetLayout(plan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetLayout(probe_plan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(plan, CLFFT_INPLACE);
    clfftSetResultLocation(probe_plan, CLFFT_INPLACE);
    clfftBakePlan(plan, 1, &command_queue, NULL, NULL);
    clfftBakePlan(probe_plan, 1, &command_queue, NULL, NULL);
   
    cl_program program = clCreateProgramWithSource(context, 1, (const char**) &kernels_ocl, NULL, &success);

    success = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    kernel_propagate = clCreateKernel(program, "propagate", &success);
    kernel_transmit = clCreateKernel(program, "transmit", &success);
    kernel_sum = clCreateKernel(program, "sum_intensity", &success);
    kernel_prepare_probe = clCreateKernel(program, "prepare_probe", &success);
    kernel_scale_probe = clCreateKernel(program, "scale_probe", &success);
    kernel_gauss_multiply = clCreateKernel(program, "gauss_multiply", &success);
    kernel_prepare_probe_espread = clCreateKernel(program, "prepare_probe_espread", &success);
    
    if (success != CL_SUCCESS) {
        
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        
        // Allocate memory for the log
        char *log = (char *) malloc(log_size);
        
        // Get the log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        
        // Print the log
        printf("%s\n", log);
        
        exit(-1);
    }
}

void freeGlobalResources() {
    free(det_min);
    free(det_max);
}

int readXYZfile(char *filename, int **zNum,float **x_pos,float **y_pos,float **z_pos, float sliceThickness, int *na, float *ax, float *by, SimDetails *sim_details) {
    FILE *file = fopen(filename, "r");
    
    int nslices = 0;
    
    if (file == NULL) {
        fprintf(stderr, "Error reading file %s\n", filename);
        return -1;
    }
    
    int natoms = 0;
    size_t line_size = 1024;
    char *line = (char *)malloc(line_size);
    int count = 0;
    
    int occ;
    float wobble;
    
    float xmin, ymin, zmin;
    float xmax, ymax, zmax;
       
    getline(&line, &line_size, file);
    sscanf(line, "%i\n", &natoms);
    getline(&line, &line_size, file);
    sscanf(line, "%f %f\n", &xmin, &xmax);
    getline(&line, &line_size, file);
    sscanf(line, "%f %f\n", &ymin, &ymax);
    getline(&line, &line_size, file);
    sscanf(line, "%f %f\n", &zmin, &zmax);
    
    *zNum = (int *)malloc(sizeof(int)*natoms);
    *x_pos = (float *)malloc(sizeof(float)*natoms);
    *y_pos = (float *)malloc(sizeof(float)*natoms);
    *z_pos = (float *)malloc(sizeof(float)*natoms);
    
    *na = natoms;
    while (getline(&line, &line_size, file) >= 0) {
        
        sscanf(line, "%i\t%f\t%f\%f\t%d\t%f\n", &(*zNum)[count], &(*x_pos)[count], &(*y_pos)[count], &(*z_pos)[count], &occ, &wobble);
        count++;
        
    }
    if (count != natoms) {
        // malformatted xyz file
        return -1;
    }
    // coordinate origin on the lower left corner
    for (int i = 0; i < natoms; i++) {
        (*x_pos)[i] -= xmin;
        (*y_pos)[i] -= ymin;
        (*z_pos)[i] -= zmin;
    }
    
     // move scan area to desired location to be consistent
     sim_details->xi -= xmin;
     sim_details->xf -= xmin;
     sim_details->yi -= ymin;
     sim_details->yf -= ymin;
        
     // check if scan area is within sample area
     if (sim_details->xi < 0.0 || sim_details->xf > xmax-xmin || sim_details->yi < 0.0 || sim_details->yf > ymax-ymin) {
		print_verbose("%s[ERROR] Scan area out of bounce.%s\n", KRED, KNRM);
		exit(1);
	}
    
    
    // sort atoms according to z position for easy slicing
    int i, j, h, Znum2;
    float x2, y2, z2;
    
    for( h=1; h<=(natoms-1)/9; h=3*h+1);
    
    for( ; h>0; h/=3 ) {
        for( i=h; i<natoms; i++) {
            j = i;
            x2 = (*x_pos)[i];
            y2 = (*y_pos)[i];
            z2 = (*z_pos)[i];
            Znum2 = (*zNum)[i];
            while( (j >= h) && ( z2 < (*z_pos)[j-h]) ) {
                (*x_pos)[j] = (*x_pos)[j-h];
                (*y_pos)[j] = (*y_pos)[j-h];
                (*z_pos)[j] = (*z_pos)[j-h];
                (*zNum)[j] = (*zNum)[j-h];
                j -= h;
            }
            (*x_pos)[j] = x2;
            (*y_pos)[j] = y2;
            (*z_pos)[j] = z2;
            (*zNum)[j] = Znum2;
        }
    }
    
    *ax = xmax-xmin;
    *by = ymax-ymin;
    
    nslices = ceil((zmax - zmin) / sliceThickness);
    
    return nslices;
}

int readParameterFile(SimDetails *sim_details) {
    
    FILE *parameterFile = fopen("parameter.dat", "r");
    if (parameterFile == NULL) {
        print_verbose("%s[ERROR] Parameter file (parameter.dat) not found.%s\n", KRED, KNRM);
        return EXIT_FAILURE;
    }
    float pi = 4.0 * atan( 1.0 );
    
    size_t line_size = 1024;
    char *line = (char *)malloc(line_size);
    
    getline(&line, &line_size, parameterFile);
    
    getline(&line, &line_size, parameterFile);
    sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf\n", &sim_details->v0, &sim_details->Cs3, &sim_details->Cs5, &sim_details->df, &sim_details->apert1, &sim_details->apert2, &sim_details->Cc, &sim_details->dE_E);
    sim_details->dE_E /= sim_details->v0*1e3;
    sim_details->Cc *= 1e7;
    
    getline(&line, &line_size, parameterFile);
    
    getline(&line, &line_size, parameterFile);
    sscanf(line, "%f %f\n", &sim_details->dfa2, &sim_details->dfa2phi);
    sim_details->dfa2phi = (float) (sim_details->dfa2phi * pi /180.0F);
    
    getline(&line, &line_size, parameterFile);
    
    getline(&line, &line_size, parameterFile);
    sscanf(line, "%f %f\n", &sim_details->dfa3, &sim_details->dfa3phi);
    sim_details->dfa3phi = (float) (sim_details->dfa3phi * pi /180.0F);
    
    sim_details->mm0 = 1.0F + sim_details->v0/511.0F;
    sim_details->wavlen = wavelength(sim_details->v0);
    
    if( sim_details->apert1 > sim_details->apert2 ) {
        printf("Bad probe aperture specification.\n");
        printf("aperture1 must be less than aperture2.\n");
        printf("apert1=%f, apert2 = %f\n", sim_details->apert1, sim_details->apert2);
        exit(-1);
    }
    
    getline(&line, &line_size, parameterFile);
    
    getline(&line, &line_size, parameterFile);
    sscanf(line, "%f %f\n", &sim_details->ctiltx, &sim_details->ctilty);
    sim_details->ctiltx = sim_details->ctiltx * 0.001;
    sim_details->ctilty = sim_details->ctilty * 0.001;
    
    getline(&line, &line_size, parameterFile);
    
    getline(&line, &line_size, parameterFile);
    sscanf(line, "%d\n", &n_detect);
    n_detect++;
    
    det_min  = (float*) malloc(n_detect * sizeof(float));
    det_max  = (float*) malloc(n_detect * sizeof(float));
    
    det_min[0] = 0.0;
    det_max[0] = 10000.0;
    
    getline(&line, &line_size, parameterFile);
    
    for (int idetect = 1; idetect < n_detect; idetect++) {
        getline(&line, &line_size, parameterFile);
        sscanf(line, "%f %f\n", &det_min[idetect], &det_max[idetect]);
    }
    
    getline(&line, &line_size, parameterFile);
    
    getline(&line, &line_size, parameterFile);
    sscanf(line, "%f %f %f %f\n", &sim_details->xi, &sim_details->xf, &sim_details->yi, &sim_details->yf);
    
    getline(&line, &line_size, parameterFile);
    
    getline(&line, &line_size, parameterFile);
    sscanf(line, "%f\n", &sim_details->sliceThickness);
    
    getline(&line, &line_size, parameterFile);
    
    getline(&line, &line_size, parameterFile);
    sscanf(line, "%d %d\n", &probe_nx, &probe_ny);
    
    getline(&line, &line_size, parameterFile);
   
    getline(&line, &line_size, parameterFile);
    sscanf(line, "%d %d\n", &nx, &ny);
    
    if (probe_nx > nx) {
        probe_nx = nx;
        print_verbose("Resizing probe wavefunction (nxprobe) to: %d\n", probe_nx);
    }
    
    if (probe_ny > ny) {
        probe_ny = ny;
        print_verbose("Resizing probe wavefunction (nyprobe) to: %d\n", probe_ny);
    }
    
    getline(&line, &line_size, parameterFile);
    
    getline(&line, &line_size, parameterFile);
    sscanf(line, "%d %d\n", &sim_details->nxout, &sim_details->nyout);
    
    getline(&line, &line_size, parameterFile);
    
    getline(&line, &line_size, parameterFile);
    sim_details->inputFilename = malloc(1024);
    sscanf(line, "%s\n", sim_details->inputFilename);
    
    getline(&line, &line_size, parameterFile);
    
    getline(&line, &line_size, parameterFile);
    sscanf(line, "%d\n", &sim_details->neDiffPattern);
    
    getline(&line, &line_size, parameterFile);
    
    getline(&line, &line_size, parameterFile);
    sscanf(line, "%d %d %d\n", &GPU_PROBE_CALCULATION, &SMOOTH_PROBE, &num_parallel, &use_hdd_mem);
    
    fclose(parameterFile);
    return 0;
}

void prepare_probe_kirkland(double x, double y, cl_float2 *probe, SimDetails sim_details) {
    /*  generate a probe at position x,y which must be inside the
     0->ax and 0->by region
     normalize to have unit integrated intensity
     probe position (x,y) = (0,0) = lower left corner
     
     NOTE: The probe wavefunction is nxprobe x nyprobe which may
     be smaller than the transmission function.
     Position the probe near the center of the nxprobe x nyprobe
     region because it does not wrap around properly
     (i.e. manually wrap around in the nx x ny region if
     the probe is near a boundary of the nx x ny region))
     */
    
    double ixmid = probe_nx/2;
    double ixoff = (int) floor( x*((double)nx) / vol_dim_x ) - ixmid;
    double xoff  = x - vol_dim_x*((double)ixoff)/((double)nx);
    
    double iymid = probe_ny/2;
    double iyoff = (int) floor( y*((double)ny) / vol_dim_y ) - iymid;
    double yoff  = y - vol_dim_y*((double)iyoff)/((double)ny);
    
    float pi = 4.0 * atan( 1.0 );
    
    double chi1 = pi * sim_details.wavlen;
    double chi2 = 0.5 * sim_details.Cs3 * 1.e7*sim_details.wavlen*sim_details.wavlen;
    double chi3 = sim_details.Cs5 * 1.0e7 * sim_details.wavlen*sim_details.wavlen*sim_details.wavlen*sim_details.wavlen /3.0;
    double k2maxa = sim_details.apert1 * 0.001/sim_details.wavlen;
    k2maxa = k2maxa *k2maxa;
    double k2maxb = sim_details.apert2 * 0.001/sim_details.wavlen;
    k2maxb = k2maxb * k2maxb;
    
    double sum0 = 0.0;
    for(int ix=0; ix<probe_nx; ix++) {
        for(int iy=0; iy<probe_ny; iy++) {
            double k2 = p_spat_freqx2[ix] + p_spat_freqy2[iy];
            if( (k2 >= k2maxa) && (k2 <= k2maxb) ) {
                double w = 2.*pi* ( xoff*p_spat_freqx[ix] + yoff*p_spat_freqy[iy] );
                double phi = atan2( spat_freqy[iy], spat_freqx[ix] );
                double chi = chi1*k2* ( (chi2 + chi3*k2)*k2 - sim_details.df
                                       + sim_details.dfa2*sin( 2.0*(phi-sim_details.dfa2phi) )
                                       + 2.0F*sim_details.dfa3*sim_details.wavlen*sqrt(k2)*
                                       sin( 3.0*(phi-sim_details.dfa3phi) )/3.0 );
                chi= - chi + w;
                double tr, ti;
                probe[iy*probe_nx + ix].s0 = tr = (float) cos( chi );  /* real */
                probe[iy*probe_nx + ix].s1 = ti = (float) sin( chi );  /* imag */
                sum0 += (double) (tr*tr + ti*ti);
            } else {
                probe[iy*probe_nx + ix].s0 = 0.0F;
                probe[iy*probe_nx + ix].s1 = 0.0F;
            }
        }
	}
}

void prepare_probe_espread(double x, double y, cl_float2 *probe, SimDetails sim_details) {
    /*  generate a probe at position x,y which must be inside the
     0->ax and 0->by region
     normalize to have unit integrated intensity
     probe position (x,y) = (0,0) = lower left corner
     
     NOTE: The probe wavefunction is nxprobe x nyprobe which may
     be smaller than the transmission function.
     Position the probe near the center of the nxprobe x nyprobe
     region because it does not wrap around properly
     (i.e. manually wrap around in the nx x ny region if
     the probe is near a boundary of the nx x ny region))
     */
    double ixmid = probe_nx/2;
    double ixoff = (int) floor( x*((double)nx) / vol_dim_x ) - ixmid;
    double xoff  = x - vol_dim_x*((double)ixoff)/((double)nx);
        
    double iymid = probe_ny/2;
    double iyoff = (int) floor( y*((double)ny) / vol_dim_y ) - iymid;
    double yoff  = y - vol_dim_y*((double)iyoff)/((double)ny);
    
    double gaussScale = 0.05;
    double delta = sim_details.Cc * sim_details.dE_E;
    
    double rx = 1.0/vol_dim_x;
	double rx2 = rx * rx;
	double ry = 1.0/vol_dim_y;
	double ry2 = ry * ry;
    
    float pi = 4.0 * atan( 1.0 );
    
   /*
    double chi1 = pi * sim_details.wavlen;
    double chi2 = 0.5 * sim_details.Cs3 * 1.e7 * sim_details.wavlen*sim_details.wavlen;
    double chi3 = sim_details.Cs5 * 1.0e7 * sim_details.wavlen*sim_details.wavlen*sim_details.wavlen*sim_details.wavlen /3.0;
    */
    
    double k2maxa = sim_details.apert1 * 0.001/sim_details.wavlen;
    k2maxa = k2maxa *k2maxa;
    double k2maxb = sim_details.apert2 * 0.001/sim_details.wavlen;
    k2maxb = k2maxb * k2maxb;
    
    double pixel = ( rx2 + ry2 );
	double scale = 1.0/sqrt((double)probe_nx*(double)probe_ny);
    for(int ix=0; ix<probe_nx; ix++) {
        for(int iy=0; iy<probe_ny; iy++) {
            double k2 = p_spat_freqx2[ix] + p_spat_freqy2[iy]; 
            
            double ktheta2 = k2*(sim_details.wavlen*sim_details.wavlen);
			double phi = atan2( spat_freqy[iy], spat_freqx[ix] );
			double w = 2.*pi* ( xoff*p_spat_freqx[ix] + yoff*p_spat_freqy[iy] );
			    
            double chi = ktheta2*(-sim_details.df+delta + sim_details.dfa2*sin(2.0*(phi-sim_details.dfa2phi)))/2.0;
            chi *= 2*pi/sim_details.wavlen;
            chi -= w;
            
            if (fabs(k2-k2maxa) <= pixel) {
					probe[iy*probe_nx + ix].s0 = (float)(0.5*scale * cos(chi));
					probe[iy*probe_nx + ix].s1 = (float)(-0.5*scale * sin(chi));
			} else if( (k2 >= k2maxa) && (k2 <= k2maxb) ) {
	           probe[iy*probe_nx + ix].s0 = (float) scale * cos( chi );  /* real */
                probe[iy*probe_nx + ix].s1 = (float) -scale * sin( chi );  /* imag */
            } else {
                probe[iy*probe_nx + ix].s0 = 0.0F;
                probe[iy*probe_nx + ix].s1 = 0.0F;
            }
        }
    }
}

void prepare_probe_espread_device(cl_float x, cl_float y, cl_mem probe_buffer, SimDetails sim_details) {
	double ixmid = probe_nx/2;
    double ixoff = (int) floor( x*((double)nx) / vol_dim_x ) - ixmid;
    double xoff  = x - vol_dim_x*((double)ixoff)/((double)nx);
    
    double iymid = probe_ny/2;
    double iyoff = (int) floor( y*((double)ny) / vol_dim_y ) - iymid;
    double yoff  = y - vol_dim_y*((double)iyoff)/((double)ny);
    
    float pi = 4.0f * atan( 1.0f );
    
    double rx = 1.0/vol_dim_x;
	double rx2 = rx * rx;
	double ry = 1.0/vol_dim_y;
	double ry2 = ry * ry;
	float pixel = ( rx2 + ry2 );
	float delta = sim_details.Cc * sim_details.dE_E;
    
    cl_float chi3 = sim_details.Cs5 * 1.0e7 * sim_details.wavlen * sim_details.wavlen * sim_details.wavlen * sim_details.wavlen /3.0;
    cl_float k2maxa = sim_details.apert1 * 0.001 / sim_details.wavlen;
    k2maxa = k2maxa * k2maxa;
    cl_float k2maxb = sim_details.apert2 * 0.001 / sim_details.wavlen;
    k2maxb = k2maxb * k2maxb;
    clSetKernelArg(kernel_prepare_probe_espread, 0, sizeof(cl_float), &x);
    clSetKernelArg(kernel_prepare_probe_espread, 1, sizeof(cl_float), &y);
    clSetKernelArg(kernel_prepare_probe_espread, 2, sizeof(cl_float), &k2maxa);
    clSetKernelArg(kernel_prepare_probe_espread, 3, sizeof(cl_float), &k2maxb);
    clSetKernelArg(kernel_prepare_probe_espread, 4, sizeof(cl_float), &pixel);
    clSetKernelArg(kernel_prepare_probe_espread, 5, sizeof(cl_float), &delta);
    clSetKernelArg(kernel_prepare_probe_espread, 6, sizeof(cl_float), &chi3);
    clSetKernelArg(kernel_prepare_probe_espread, 7, sizeof(cl_float), &xoff);
    clSetKernelArg(kernel_prepare_probe_espread, 8, sizeof(cl_float), &yoff);
    clSetKernelArg(kernel_prepare_probe_espread, 9, sizeof(cl_mem), &probe_buffer);
    
    size_t work_dim[2];
    work_dim[0] = probe_nx;
    work_dim[1] = probe_ny;
    clEnqueueNDRangeKernel(command_queue, kernel_prepare_probe_espread, 2, NULL, work_dim, NULL, 0, NULL, NULL);
}

void prepare_probe_device(cl_float x, cl_float y, cl_mem probe_buffer, SimDetails sim_details) {
    
    cl_float ixmid = probe_nx/2;
    cl_float ixoff = (int) floor( x*((cl_float)nx) / vol_dim_x ) - ixmid;
    cl_float xoff  = x - vol_dim_x*((cl_float)ixoff)/((cl_float)nx);
    
    cl_float iymid = probe_ny/2;
    cl_float iyoff = (int) floor( y*((cl_float)ny) / vol_dim_y ) - iymid;
    cl_float yoff  = y - vol_dim_y*((cl_float)iyoff)/((cl_float)ny);
    
    float pi = 4.0f * atan( 1.0f );
    
    cl_float chi1 = pi * sim_details.wavlen;
    cl_float chi2 = 0.5 * sim_details.Cs3 * 1.e7 * sim_details.wavlen * sim_details.wavlen;
    cl_float chi3 = sim_details.Cs5 * 1.0e7 * sim_details.wavlen * sim_details.wavlen * sim_details.wavlen * sim_details.wavlen /3.0;
    cl_float k2maxa = sim_details.apert1 * 0.001 / sim_details.wavlen;
    k2maxa = k2maxa * k2maxa;
    cl_float k2maxb = sim_details.apert2 * 0.001 / sim_details.wavlen;
    k2maxb = k2maxb * k2maxb;
    
    clSetKernelArg(kernel_prepare_probe, 0, sizeof(cl_float), &x);
    clSetKernelArg(kernel_prepare_probe, 1, sizeof(cl_float), &y);
    clSetKernelArg(kernel_prepare_probe, 2, sizeof(cl_float), &k2maxa);
    clSetKernelArg(kernel_prepare_probe, 3, sizeof(cl_float), &k2maxb);
    clSetKernelArg(kernel_prepare_probe, 4, sizeof(cl_float), &chi1);
    clSetKernelArg(kernel_prepare_probe, 5, sizeof(cl_float), &chi2);
    clSetKernelArg(kernel_prepare_probe, 6, sizeof(cl_float), &chi3);
    clSetKernelArg(kernel_prepare_probe, 7, sizeof(cl_float), &xoff);
    clSetKernelArg(kernel_prepare_probe, 8, sizeof(cl_float), &yoff);
    clSetKernelArg(kernel_prepare_probe, 9, sizeof(cl_mem), &probe_buffer);
    
    size_t work_dim[2];
    work_dim[0] = probe_nx;
    work_dim[1] = probe_ny;
    clEnqueueNDRangeKernel(command_queue, kernel_prepare_probe, 2, NULL, work_dim, NULL, 0, NULL, NULL);
}

void scale_probe_device(cl_mem probe_buffer, float probe_sum) {
    clSetKernelArg(kernel_scale_probe, 0, sizeof(cl_mem), &probe_buffer);
    clSetKernelArg(kernel_scale_probe, 1, sizeof(cl_float), &probe_sum);
    
    size_t scale_work_dim[2];
    scale_work_dim[0] = probe_nx;
    scale_work_dim[1] = probe_ny;
    clEnqueueNDRangeKernel(command_queue, kernel_scale_probe, 2, NULL, scale_work_dim, NULL, 0, NULL, NULL);
}

void sum_intensity(cl_float *detect, cl_mem probe_buffer, float *sum) {
    clSetKernelArg(kernel_sum, 0, sizeof(cl_mem), &probe_buffer);
    clSetKernelArg(kernel_sum, 1, sizeof(cl_mem), &kxp2_buffer);
    clSetKernelArg(kernel_sum, 2, sizeof(cl_mem), &kyp2_buffer);
    clSetKernelArg(kernel_sum, 3, sizeof(cl_mem), &k2min_buffer);
    clSetKernelArg(kernel_sum, 4, sizeof(cl_mem), &k2max_buffer);
    clSetKernelArg(kernel_sum, 5, sizeof(int), &n_detect);
    clSetKernelArg(kernel_sum, 6, sizeof(int), &probe_ny);
    clSetKernelArg(kernel_sum, 7, sizeof(cl_mem), &temp_buffer);
    
    size_t work_dim[1];
    work_dim[0] = probe_nx;
    clEnqueueNDRangeKernel(command_queue, kernel_sum, 1, NULL, work_dim, NULL, 0, NULL, NULL);
    cl_float *temp = (cl_float *)malloc(sizeof(cl_float)*n_detect*probe_ny);
    clEnqueueReadBuffer(command_queue, temp_buffer, CL_TRUE, 0, sizeof(cl_float)*n_detect*probe_ny, temp, 0, NULL, NULL);

    for (int idetect=0; idetect < n_detect; idetect++) {
        detect[idetect] = 0.0;
    }

    for (int ix = 0; ix < probe_nx; ix++) {
        
        for (int idetect=0; idetect < n_detect; idetect++) {
            detect[idetect] += temp[idetect * probe_nx + ix];
            if (idetect == 0) {
                *sum += temp[idetect * probe_nx + ix];
            }
        }
        
    }

    free(temp);
}

void probe_ifft(cl_mem buffer) {
    clfftEnqueueTransform(probe_plan, CLFFT_BACKWARD, 1, &command_queue, 0, NULL, NULL, &buffer, NULL, NULL);
}

void probe_fft(cl_mem buffer) {
    clfftEnqueueTransform(probe_plan, CLFFT_FORWARD, 1, &command_queue, 0, NULL, NULL, &buffer, NULL, NULL);
}

void transmit (cl_mem probe_buffer, cl_mem transmit_buffer, int nxoff, int nyoff) {
    clSetKernelArg(kernel_transmit, 0, sizeof(cl_mem), &probe_buffer);
    clSetKernelArg(kernel_transmit, 1, sizeof(cl_mem), &transmit_buffer);
    clSetKernelArg(kernel_transmit, 2, sizeof(cl_int), &nxoff);
    clSetKernelArg(kernel_transmit, 3, sizeof(cl_int), &nyoff);
    clSetKernelArg(kernel_transmit, 4, sizeof(cl_int), &nx);
    clSetKernelArg(kernel_transmit, 5, sizeof(cl_int), &ny);

    size_t work_dim[2];
    work_dim[0] = probe_nx;
    work_dim[1] = probe_ny;
    clEnqueueNDRangeKernel(command_queue, kernel_transmit, 2, NULL, work_dim, NULL, 0, NULL, NULL);
}

void propagate(cl_mem probe_buffer, int pid) {
    cl_int l = pid;
    clSetKernelArg(kernel_propagate, 0, sizeof(cl_mem), &probe_buffer);
    clSetKernelArg(kernel_propagate, 5, sizeof(cl_float), &k2maxp);
    clSetKernelArg(kernel_propagate, 6, sizeof(cl_int), &l);
    size_t work_dim[2];
    work_dim[0] = probe_nx;
    work_dim[1] = probe_ny;
    clEnqueueNDRangeKernel(command_queue, kernel_propagate, 2, NULL, work_dim, NULL, 0, NULL, NULL);
}


void STEMsignal( double x, double y, cl_float *detect, double *sum, int saveDP, int islice, cl_mem probe_buffer, cl_mem transmit_buffer, int pid, SimDetails details, int pixel_num, float *edx_signal, int ix, int iy) {

    long nxprobel, nyprobel;
    
    int ixmid = probe_nx/2;
    int ixoff = (int) floor( x*((double)nx) / vol_dim_x ) - ixmid;
    
    int iymid = probe_ny/2;
    int iyoff = (int) floor( y*((double)ny) / vol_dim_y ) - iymid;
    
    /*   make sure x,y are ok */
    
    if( (x < 0.0) || (x > vol_dim_x) ||
       (y < 0.0) || (y > vol_dim_y) ) {
        *sum = -1.2345;
        printf("bad x=%f,y=%f\n", x, y);
        return;
    }
    
    /*  transmit thru nslice layers
     beware ixoff,iyoff must be with one nx,ny
     of the array bounds
     */
    
    nxprobel = (long) probe_nx;
    nyprobel = (long) probe_ny;
    
    probe_ifft(probe_buffer);
    transmit(probe_buffer, transmit_buffer, ixoff, iyoff);
    
    if (details.calculate_edx > 0) {
        float probe_sum = 0.0;
        sum_intensity(detect, probe_buffer, &probe_sum);
        int index = islice * details.nxout * details.nyout + ix * details.nyout + iy;
        edx_signal[index] = probe_sum;
    }
    
    if (islice == n_slices - 1 && details.neDiffPattern != 0 && pixel_num % details.neDiffPattern == 0) {
        // copy probe and calculate diffraction pattern in linescan mode
        clEnqueueReadBuffer(command_queue, probe_buffer, CL_TRUE, 0, sizeof(cl_float2)*probe_nx*probe_ny, probes[pid], 0, NULL, NULL);
        save_diffraction_pattern(probes[pid], x, y);
    }
    
    probe_fft(probe_buffer);
    propagate(probe_buffer, islice);
}

void save_diffraction_pattern(cl_float2 *probe, double x, double y) {
    
    float *real = (float *)malloc(sizeof(float) * probe_nx * probe_ny);
    
    for (int x = 0; x < probe_nx; x++) {
        for (int y = 0; y < probe_ny; y++) {
            float prr = probe[y*probe_nx + x].s0;
            float pri = probe[y*probe_nx + x].s1;
            
            float value = fabs(prr*prr)+fabs(pri*pri);
            real[y * probe_nx + x] = value;
        }
    }
    
    char *filename = malloc(1024);
    
    struct stat st = {0};
    if (stat("diffraction_pattern/", &st) == -1) {
        mkdir("diffraction_pattern/", 0777);
    }
    
    sprintf(filename, "diffraction_pattern/pattern_%.3f_%.3f.dp", x, y);
    FILE *diff_output = fopen(filename, "ab+");
    for (int x = 0; x < probe_nx; x++) {
        for (int y = 0; y < probe_ny; y++) {
            located_intensity intensity;
            intensity.x = x;
            intensity.y = y;
            intensity.value = real[y * probe_nx + x];
            intensity.detector = 0;
            fwrite(&intensity, sizeof(located_intensity), 1, diff_output);
            fflush(diff_output);
        }
    }
    fclose(diff_output);
    free(real);
}

void gauss_multiply_device(cl_mem probe_buffer, float gausscale) {
	clSetKernelArg(kernel_gauss_multiply, 0, sizeof(cl_mem), &probe_buffer);
    clSetKernelArg(kernel_gauss_multiply, 1, sizeof(cl_float), &gausscale);
    
    size_t work_dim[2];
    work_dim[0] = probe_nx;
    work_dim[1] = probe_ny;
    clEnqueueNDRangeKernel(command_queue, kernel_gauss_multiply, 2, NULL, work_dim, NULL, 0, NULL, NULL);
}
