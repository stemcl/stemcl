#include "helpers.h"

const char *KNRM = "\x1B[0m";
const char *KRED = "\x1B[31m";
const char *KGRN = "\x1B[32m";
const char *KYEL = "\x1B[33m";
const char *KBLU = "\x1B[34m";
const char *KMAG = "\x1B[35m";
const char *KCYN = "\x1B[36m";
const char *KWHT = "\x1B[37m";
const char *KCLR = "\x1B[2K";
time_t time_start = 0;

void print_error(cl_int error, const char *description) {
	if (time_start == 0) {
		time_start = time(0);
	}

	if (verbose || error != CL_SUCCESS) {
		FILE *output = (error == CL_SUCCESS) ? stdout : stderr;
		const char *color = (error == CL_SUCCESS) ? KGRN : KRED;
		char *time_string = format_time(time(0) - time_start);
		fprintf(output, "[%s] %s%s%s %s\n", time_string, color, error_name(error), KNRM, description);
		free(time_string);
	}
	
	if (error != CL_SUCCESS) {
		exit(error);
	}
	
}

void print_verbose(const char *format, ...) {
	if (time_start == 0) {
		time_start = time(0);
	}

	if (verbose) {
	    va_list args;
	    va_start(args, format);
		char *format_new = malloc(strlen(format) + 60);
		char *time_string = format_time(time(0) - time_start);
		sprintf(format_new, "[%s] %s", time_string, format);
		vfprintf(stdout, format_new, args);
		fflush(stdout);
	    va_end(args);
		free(format_new);
		free(time_string);
	}
	
}

char *str_replace(char *string, const char *substr, const char *replacement) {
	char *tok = NULL;
	char *newstr = NULL;
	char *oldstr = NULL;
	int oldstr_len = 0;
	int substr_len = 0;
	int replacement_len = 0;

	newstr = strdup(string);
	substr_len = strlen(substr);
	replacement_len = strlen(replacement);

	if (substr == NULL || replacement == NULL) {
		return newstr;
	}

	while ((tok = strstr(newstr, substr))) {
		oldstr = newstr;
		oldstr_len = strlen(oldstr);
		newstr = (char*)malloc(sizeof(char) * (oldstr_len - substr_len + replacement_len + 1));

		if (newstr == NULL) {
			free(oldstr);
			return NULL;
		}

		memcpy(newstr, oldstr, tok - oldstr);
		memcpy(newstr + (tok - oldstr), replacement, replacement_len);
		memcpy(newstr + (tok - oldstr) + replacement_len, tok + substr_len, oldstr_len - substr_len - (tok - oldstr));
		memset(newstr + oldstr_len - substr_len + replacement_len, 0, 1);

		free(oldstr);
	}

	return newstr;
}

void cl_error (const char *errinfo, const void *private_info, size_t cb, void *user_data) {
	fprintf(stderr, "%s%s%s\n", KRED, errinfo, KNRM);
	exit(1);
}

char *format_size(size_t size) {
	const char *units[] = {"Bytes", "KB", "MB", "GB", "TB", "PB", "EB"};
	double index = floor(log2(size)/10.0);
	const char *unit = units[(int)index];
	char *return_value = malloc(strlen(unit) + 12);
	sprintf(return_value, "%.0f %s", (double)size/pow(2, index*10), unit);
	return return_value;
}

char *format_frequency(size_t size) {
	const char *units[] = {"Hz", "KHz", "MHz", "GHz", "Thz", "PHz", "EHz"};
	double index = floor(log10(size)/3);
	const char *unit = units[(int)index];
	char *return_value = malloc(strlen(unit) + 12);
	sprintf(return_value, "%.1f %s", (double)size/pow(10, index*3), unit);
	return return_value;
}

char *format_time(double seconds) {
	char *return_value = malloc(32);
	double hours = floor(seconds/3600.0);
	double minutes = floor(seconds/60.0) - hours * 60.0;
	seconds = seconds - minutes * 60.0 - hours * 60.0 * 60.0;
	sprintf(return_value, "%02.0f:%02.0f:%02.0f", hours, minutes, seconds);
	return return_value;
}


cl_device_id get_device(cl_device_type requested_device_type, const char *requested_platform_name, int device_number) {
    // OpenCL initialization begins here
    const cl_uint platforms_max = 10;
    cl_platform_id platforms[platforms_max];
    cl_uint platforms_count = 0;
    clGetPlatformIDs(platforms_max, platforms, &platforms_count);
	cl_device_id device_id = 0;

    if (!platforms_count) {
        fprintf(stderr, "%sCould not initialize OpenCL. No platforms found%s\n", KRED, KNRM);
        return device_id;
    }
    
    // Show information about all available platforms
    for (int idx = 0; idx < platforms_count; idx++) {
        char platform_version[255] = {0x0};
        char platform_name[255] = {0x0};
        clGetPlatformInfo(platforms[idx], CL_PLATFORM_VERSION, 255, platform_version, NULL);
        clGetPlatformInfo(platforms[idx], CL_PLATFORM_NAME, 255, platform_name, NULL);
		bool platform_matched = strcasestr(platform_name, requested_platform_name);
        print_verbose("%s%s (%s)%s\n", platform_matched ? KGRN : KNRM, platform_name, platform_version, KNRM);

        const cl_uint devices_max = 10;
        cl_device_id devices[devices_max];
		cl_device_type device_type;
        cl_uint devices_count = 0;
        cl_uint compute_units;
        cl_uint clock_frequency;
        cl_ulong mem_size;
        char device_name[255] = {0x0};
		bool device_type_matched = false;
		bool device_name_matched = false;
        clGetDeviceIDs(platforms[idx], CL_DEVICE_TYPE_ALL, devices_max, devices, &devices_count);
		
        for (cl_uint device_idx = 0; device_idx < devices_count; device_idx++) {
            clGetDeviceInfo(devices[device_idx], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
            clGetDeviceInfo(devices[device_idx], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
            clGetDeviceInfo(devices[device_idx], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
            clGetDeviceInfo(devices[device_idx], CL_DEVICE_NAME, 255, device_name, NULL);
			clGetDeviceInfo(devices[device_idx], CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
			device_type_matched = device_type & requested_device_type;
			device_name_matched = device_idx == device_number;
			const char *device_color = KNRM;

			if (device_type_matched && device_name_matched && platform_matched) {
				
				if (!device_id) {
					device_id = devices[device_idx];
					device_color = KGRN;
				}
				
			}

			char *device_type_string = "";
			switch (device_type) {
				case CL_DEVICE_TYPE_CPU:
					device_type_string = "CPU";
					break;
				case CL_DEVICE_TYPE_GPU:
					device_type_string = "GPU";
					break;
				case CL_DEVICE_TYPE_ACCELERATOR:
					device_type_string= "Accelerator";
					break;
				default:
					break;
			}

			bool last_device = (device_idx == devices_count - 1);
			const char *device_type_color = device_type_matched ? KGRN : KNRM;
			char *tree_char = last_device ? "└" : "├";
			print_verbose("%s─ %s%s%s\n",tree_char, device_color, device_name, KNRM);
			char *size_string = format_size(mem_size);
			char *clock_string = format_frequency(clock_frequency * 1000000);
			tree_char = last_device ? " " : "│";
			print_verbose("%s  ├─ Device type: %s%s%s\n", tree_char, device_type_color, device_type_string, KNRM);
			print_verbose("%s  ├─ Global memory: %s\n", tree_char, size_string);
			print_verbose("%s  ├─ Clock frequency: %s\n", tree_char, clock_string);
			print_verbose("%s  └─ Compute units: %d\n", tree_char, compute_units);
			free(clock_string);
			free(size_string);
        }
		
    }
	
    if (!device_id) {
        print_error(CL_INVALID_DEVICE, "Could not find suitable device");
        char *device_type_string = "";
        switch (requested_device_type) {
            case CL_DEVICE_TYPE_CPU:
                device_type_string = "CPU";
                break;
            case CL_DEVICE_TYPE_GPU:
                device_type_string = "GPU";
                break;
            case CL_DEVICE_TYPE_ACCELERATOR:
                device_type_string= "Accelerator";
                break;
            default:
                break;
        }
        print_verbose("Requested device type: %s\n", device_type_string);
        print_verbose("Requested platform: %s\n", requested_platform_name);
        print_verbose("Requested device: %d\n", device_number);
    }
    
	return device_id;
}

cl_event enqueue_kernel(cl_command_queue queue, cl_kernel kernel, cl_int work_dim, const size_t *work_size) {
	char *function_name = malloc(128);
	clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 128, function_name, NULL);
	cl_event event;
	clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, work_size, NULL, 0, NULL, &event);
	clSetEventCallback(event, CL_COMPLETE, event_callback, function_name);
	return event;
}

cl_event enqueue_mem_read(cl_command_queue queue, cl_mem buffer, size_t size, void *target) {
	cl_event event;
	clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, size, target, 0, NULL, &event);
	char *size_human = format_size(size);
	clSetEventCallback(event, CL_COMPLETE, event_callback, size_human);
	return event;
}

cl_event enqueue_mem_write(cl_command_queue queue, cl_mem buffer, size_t size, void *source) {
	cl_event event;
	clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, size, source, 0, NULL, &event);
	char *size_human = format_size(size);
	clSetEventCallback(event, CL_COMPLETE, event_callback, size_human);
	return event;
}

// Event callback for OpenCL profiling
void CL_CALLBACK event_callback(cl_event event, cl_int cmd_exec_status, void *user_data) {
	/*
	if (file_timings || (cmd_exec_status != CL_SUCCESS)) {
		cl_ulong timestamp_queued, timestamp_submitted, timestamp_started, timestamp_ended;
		cl_command_type command_type;
		clGetEventInfo(event, CL_EVENT_COMMAND_TYPE, sizeof(cl_command_type), &command_type, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &timestamp_queued, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &timestamp_submitted, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &timestamp_started, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &timestamp_ended, NULL);
		fprintf(file_timings, "%s\t%llu\t%llu\t%llu\t%llu\t%s\t%s\n", command_name(command_type), timestamp_queued, timestamp_submitted, timestamp_started, timestamp_ended, error_name(cmd_exec_status), user_data);
	}
	
	free(user_data);
	clReleaseEvent(event);
    */
}
	
// Creates a filename from a given (user-supplied) basename, addendum, and
// extensions. Example: extend_basename("subj", "fa", "nii") returns
// "subj_fa.nii". You must call free() on the returned string after usage.
char *extend_basename (const char *basename, const char *addendum, const char *extension) {
	// Add three bytes for trailing null byte, _ separator between basename and 
	// addendum, and . separator between addendum and extension.
	size_t output_size = strlen(basename) + strlen(addendum) + strlen(extension) + 3;
	char *filename = malloc(output_size);
	sprintf(filename, "%s_%s.%s", basename, addendum, extension);
	return filename;
}

const char *command_name(cl_command_type command_type) {
	switch (command_type) {
		case CL_COMMAND_NDRANGE_KERNEL:
			return "NDRANGE_KERNEL";
		case CL_COMMAND_TASK:
			return "TASK";
		case CL_COMMAND_NATIVE_KERNEL:
			return "NATIVE_KERNEL";
		case CL_COMMAND_READ_BUFFER:
			return "READ_BUFFER";
		case CL_COMMAND_WRITE_BUFFER:
			return "WRITE_BUFFER";
		case CL_COMMAND_COPY_BUFFER:
			return "COPY_BUFFER";
		case CL_COMMAND_READ_IMAGE:
			return "READ_IMAGE";
		case CL_COMMAND_WRITE_IMAGE:
			return "WRITE_IMAGE";
		case CL_COMMAND_COPY_IMAGE:
			return "COPY_IMAGE";
		case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
			return "COPY_BUFFER_TO_IMAGE";
		case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
			return "COPY_IMAGE_TO_BUFFER";
		case CL_COMMAND_MAP_BUFFER:
			return "MAP_BUFFER";
		case CL_COMMAND_MAP_IMAGE:
			return "MAP_IMAGE";
		case CL_COMMAND_UNMAP_MEM_OBJECT:
			return "UNMAP_MEM_OBJECT";
		case CL_COMMAND_MARKER:
			return "MARKER";
		case CL_COMMAND_ACQUIRE_GL_OBJECTS:
			return "ACQUIRE_GL_OBJECTS";
		case CL_COMMAND_RELEASE_GL_OBJECTS:
			return "RELEASE_GL_OBJECTS";
		case CL_COMMAND_READ_BUFFER_RECT:
			return "READ_BUFFER_RECT";
		case CL_COMMAND_WRITE_BUFFER_RECT:
			return "WRITE_BUFFER_RECT";
		case CL_COMMAND_COPY_BUFFER_RECT:
			return "COPY_BUFFER_RECT";
		case CL_COMMAND_USER:
			return "USER";
	}
	
	return "UNKNOWN";
}

const char *error_name(cl_int error) {
	switch (error) {
		case 0:
			return "SUCCESS";
		case -1:
			return "DEVICE_NOT_FOUND";
		case -2:
			return "DEVICE_NOT_AVAILABLE";
		case -3:
			return "COMPILER_NOT_AVAILABLE";
		case -4:
			return "MEM_OBJECT_ALLOCATION_FAILURE";
		case -5:
			return "OUT_OF_RESOURCES";
		case -6:
			return "OUT_OF_HOST_MEMORY";
		case -7:
			return "PROFILING_INFO_NOT_AVAILABLE";
		case -8:
			return "MEM_COPY_OVERLAP";
		case -9:
			return "IMAGE_FORMAT_MISMATCH";
		case -10:
			return "IMAGE_FORMAT_NOT_SUPPORTED";
		case -11:
			return "BUILD_PROGRAM_FAILURE";
		case -12:
			return "MAP_FAILURE";
		case -13:
			return "MISALIGNED_SUB_BUFFER_OFFSET";
		case -14:
			return "EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15:
			return "COMPILE_PROGRAM_FAILURE";
		case -16:
			return "LINKER_NOT_AVAILABLE";
		case -17:
			return "LINK_PROGRAM_FAILURE";
		case -18:
			return "DEVICE_PARTITION_FAILED";
		case -19:
			return "KERNEL_ARG_INFO_NOT_AVAILABLE";
		case -30:
			return "INVALID_VALUE";
		case -31:
			return "INVALID_DEVICE_TYPE";
		case -32:
			return "INVALID_PLATFORM";
		case -33:
			return "INVALID_DEVICE";
		case -34:
			return "INVALID_CONTEXT";
		case -35:
			return "INVALID_QUEUE_PROPERTIES";
		case -36:
			return "INVALID_COMMAND_QUEUE";
		case -37:
			return "INVALID_HOST_PTR";
		case -38:
			return "INVALID_MEM_OBJECT";
		case -39:
			return "INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40:
			return "INVALID_IMAGE_SIZE";
		case -41:
			return "INVALID_SAMPLER";
		case -42:
			return "INVALID_BINARY";
		case -43:
			return "INVALID_BUILD_OPTIONS";
		case -44:
			return "INVALID_PROGRAM";
		case -45:
			return "INVALID_PROGRAM_EXECUTABLE";
		case -46:
			return "INVALID_KERNEL_NAME";
		case -47:
			return "INVALID_KERNEL_DEFINITION";
		case -48:
			return "INVALID_KERNEL";
		case -49:
			return "INVALID_ARG_INDEX";
		case -50:
			return "INVALID_ARG_VALUE";
		case -51:
			return "INVALID_ARG_SIZE";
		case -52:
			return "INVALID_KERNEL_ARGS";
		case -53:
			return "INVALID_WORK_DIMENSION";
		case -54:
			return "INVALID_WORK_GROUP_SIZE";
		case -55:
			return "INVALID_WORK_ITEM_SIZE";
		case -56:
			return "INVALID_GLOBAL_OFFSET";
		case -57:
			return "INVALID_EVENT_WAIT_LIST";
		case -58:
			return "INVALID_EVENT";
		case -59:
			return "INVALID_OPERATION";
		case -60:
			return "INVALID_GL_OBJECT";
		case -61:
			return "INVALID_BUFFER_SIZE";
		case -62:
			return "INVALID_MIP_LEVEL";
		case -63:
			return "INVALID_GLOBAL_WORK_SIZE";
		case -64:
			return "INVALID_PROPERTY";
		case -65:
			return "INVALID_IMAGE_DESCRIPTOR";
		case -66:
			return "INVALID_COMPILER_OPTIONS";
		case -67:
			return "INVALID_LINKER_OPTIONS";
		case -68:
			return "INVALID_DEVICE_PARTITION_COUNT";
	}
	
	return "UNKNOWN";
}
