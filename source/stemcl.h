//
//  stemcl.h
//  Project
//
//  Created by Manuel Radek on 11.09.16.
//
//

#ifndef stemcl_h
#define stemcl_h

#include <fcntl.h>
#include <sys/stat.h>

typedef struct {
    double v0, Cs3, Cs5, df, apert1, apert2;
    double Cc, dE_E;
    float dfa2, dfa2phi, dfa3, dfa3phi;
    float mm0, ctiltx, ctilty;
    float xi, xf, yi, yf, sliceThickness;
    int nxout, nyout;
    float wavlen;
    int neDiffPattern;
    char *inputFilename;
    int calculate_edx;
} SimDetails;

# pragma mark - Function Prototypes
void initOpenCL(cl_device_id device_id);
void freeGlobalResources();
double wavelength(double kev);
void calculateTransmissionLayer(const float x[], const float y[],
                                const int Znum[], const int natom,
                                const float ax, const float by, const float kev,
                                int slice_num, const long nx, const long ny,
                                const float k2max, cl_mem trans_buffer);
int transmission_index(int x, int y, int slice);
cl_float2 *loadTransmissionData(int slice_num);
int readXYZfile(char *filename, int **zNum,float **x_pos,float **y_pos,float **z_pos, float sliceThickness, int *na, float *ax, float *by, SimDetails *sim_details);
int readParameterFile(SimDetails *sim_details);
void frequency(float *ko, float *ko2, float *xo, int nk, double ak);
double vzatomLUT(int Z, double rsq);
void writeTransmissions(int width, int height, int slice_num);
void prepare_probe_kirkland(double x, double y, cl_float2 *probe, SimDetails sim_details);
void prepare_probe_device(cl_float x, cl_float y, cl_mem probe_buffer, SimDetails sim_details);
void prepare_probe_espread_device(cl_float x, cl_float y, cl_mem probe_buffer, SimDetails sim_details);
void prepare_probe_espread(double x, double y, cl_float2 *probe, SimDetails sim_details);
void scale_probe_device(cl_mem probe_buffer, float probe_sum);
void sum_intensity(cl_float *detect, cl_mem probe_buffer, float *sum);
void STEMsignal( double x, double y, cl_float *detect, double *sum, int saveDP, int islice, cl_mem probe_buffer, cl_mem transmit_buffer, int pid, SimDetails details, int pixel_num, float *edx_signal, int current_px_x, int current_px_y);
void save_diffraction_pattern(cl_float2 *probe, double x, double y);
void gauss_multiply_device(cl_mem probe_buffer, float gausscale);
void probe_ifft(cl_mem buffer);
void probe_fft(cl_mem buffer);
#endif /* stemcl_h */
