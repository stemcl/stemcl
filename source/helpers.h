#ifdef __APPLE__
	#include <stdlib.h>
   	#include <OpenCL/opencl.h>
#else
	#define _GNU_SOURCE
   	#include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <sys/time.h>
#include <math.h>
#include <time.h>

#ifndef FLT_MIN
	#define FLT_MIN 0x1.0p-23f
#endif

extern const char *KNRM;
extern const char *KRED;
extern const char *KGRN;
extern const char *KYEL;
extern const char *KBLU;
extern const char *KMAG;
extern const char *KCYN;
extern const char *KWHT;
extern const char *KCLR;


// Global configuration
extern bool verbose;
extern FILE *file_timings;

cl_event enqueue_kernel(cl_command_queue queue, cl_kernel kernel, cl_int work_dim, const size_t *work_size);
cl_event enqueue_mem_read(cl_command_queue queue, cl_mem buffer, size_t size, void *target);
cl_event enqueue_mem_write(cl_command_queue queue, cl_mem buffer, size_t size, void *source);

void print_error(cl_int error, const char *description);
void print_verbose(const char *format, ...);
const char *command_name(cl_command_type command_type);
const char *error_name(cl_int error);
void cl_error (const char *errinfo, const void *private_info, size_t cb, void *user_data);
char *format_size(size_t size);
char* str_replace(char* string, const char* substr, const char* replacement);


/// Formats a duration given in seconds as hh:mm:ss
/// Returned array is statically allocated and subsequent calls change its
/// contents. Make a copy if you want to keep it.
char *format_time(double seconds);
cl_device_id get_device(cl_device_type requested_device_type, const char *requested_platform_name, int device_number);

// Event callback for OpenCL profiling
void CL_CALLBACK event_callback(cl_event event, cl_int cmd_exec_status, void *user_data);
	
// Creates a filename from a given (user-supplied) basename, addendum, and
// extensions. Example: extend_basename("subj", "fa", "nii") returns
// "subj_fa.nii". You must call free() on the returned string after usage.
char *extend_basename (const char *basename, const char *addendum, const char *extension);
