#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/cl.h>
#endif

const size_t block_size = 16384/4;
const size_t cl_local_size = 16;

struct matrix {
    unsigned int rows;
    float *elements;
};
struct matrix random_symmetric_matrix(unsigned int rows);
void floyd_warshall(struct matrix m);
void print_matrix(struct matrix m);
void floyd_warshall_blocked(struct matrix m);
void floyd_warshall_parallel(struct matrix m);
void floyd_warshall_block(struct matrix m, unsigned int x, unsigned int y, unsigned int w);

int main(int argc, const char *argv[]) {
    size_t matrix_size = ceil((double)atol(argv[1]) / block_size) * block_size;
    fprintf(stdout, "Matrix size: %zu × %zu\n", matrix_size, matrix_size);

    clock_t time_a, time_b;
    double seconds;

    struct matrix m = random_symmetric_matrix(matrix_size);
    print_matrix(m);
    time_a = clock();
    floyd_warshall_parallel(m);
    time_b = clock();
    seconds = (double)(time_b - time_a) / CLOCKS_PER_SEC;
    print_matrix(m);
    fprintf(stdout, "Blocked GPU runtime: %6.2f\n", seconds);
    fprintf(stdout, "\n");
}

// Callback function for OpenCL errors on the context
void CL_CALLBACK context_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
    fprintf(stderr, "\e[31m%s\e[0m\n", errinfo);
}

// Callback for errors during the compilation and building of the OpenCL program.
void CL_CALLBACK program_notify(cl_program program, void *user_data) {
    cl_uint device_count = 0;
    clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(device_count), &device_count, NULL);
    size_t devices_size = sizeof(cl_device_id) * device_count;
    cl_device_id *devices = malloc(devices_size);
    clGetProgramInfo(program, CL_PROGRAM_DEVICES, devices_size, devices, NULL);

    for (size_t i = 0; i < device_count; i++) {
        cl_build_status build_status;
        cl_device_id device = devices[i];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(build_status), &build_status, NULL);
        
        if (build_status == CL_BUILD_ERROR) {
            fprintf(stdout, "Device #%zu build status: %d\n", i, build_status);
            char *build_log = malloc(4096);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 4096, build_log, NULL);
            fprintf(stdout, "\e[31m%s\e[0m\n", build_log);
            free(build_log);
        }
        
    }

    free(devices);
}

cl_program program_from_file(const char *filename, cl_context context) {
    FILE *file = fopen(filename, "r");
    fseek (file, 0, SEEK_END);
    size_t source_length = ftell(file);
    char *source = malloc(source_length);
    fseek(file, 0, SEEK_SET);
    fread(source, 1, source_length, file);
    fclose(file);
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, &source_length, NULL);
    char *program_arguments = malloc(4096);
    sprintf(program_arguments, "-cl-std=1.2 -D BLOCK_SIZE=%zu -D LOCAL_SIZE=%zu", block_size, cl_local_size);
    clBuildProgram(program, 0, NULL, program_arguments, program_notify, NULL);
    free(source);
    return program;
}

// Copy the block at coordinates x, y to the compute device. Coordinates are in
// blocks, i.e. 0 < x, y < m.rows/block_size
void block_to_device(struct matrix m, size_t x, size_t y, cl_command_queue queue, cl_mem buffer) {
    size_t block_row_size_bytes = sizeof(float) * block_size;
    size_t matrix_row_size_bytes = sizeof(float) * m.rows;

    for (size_t block_row_idx = 0; block_row_idx < block_size; block_row_idx++) {
        size_t device_offset = block_row_idx * block_row_size_bytes;
        size_t host_offset = (y * block_size + block_row_idx) * m.rows + (x * block_size);
        float *block_row_start = m.elements + host_offset;
        clEnqueueWriteBuffer(queue, buffer, CL_BLOCKING, device_offset, block_row_size_bytes, block_row_start, 0, NULL, NULL);
    }
    
}

void block_to_host(struct matrix m, size_t x, size_t y, cl_command_queue queue, cl_mem buffer) {
    size_t block_row_size_bytes = sizeof(float) * block_size;
    size_t matrix_row_size_bytes = sizeof(float) * m.rows;

    for (size_t block_row_idx = 0; block_row_idx < block_size; block_row_idx++) {
        size_t device_offset = block_row_idx * block_row_size_bytes;
        size_t host_offset = (y * block_size + block_row_idx) * m.rows + (x * block_size);
        float *block_row_start = m.elements + host_offset;
        clEnqueueReadBuffer(queue, buffer, CL_BLOCKING, device_offset, block_row_size_bytes, block_row_start, 0, NULL, NULL);
    }
    
}

// Perform Floyd-Warshall on an OpenCL device.
void floyd_warshall_parallel(struct matrix m) {
    const size_t device_id = 0;
    const size_t block_size_bytes = block_size * block_size * sizeof(float);
    size_t blocks_num = m.rows / block_size;
    
    cl_context context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, context_notify, NULL, NULL);
    cl_uint num_devices = 0;
    clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES, sizeof(num_devices), &num_devices, NULL);
    size_t devices_size = sizeof(cl_device_id) * num_devices;
    cl_device_id *devices = malloc(devices_size);
    clGetContextInfo(context, CL_CONTEXT_DEVICES, devices_size, devices, NULL);
    cl_command_queue *queues = malloc(sizeof(cl_command_queue) * num_devices);

    for (size_t i = 0; i < num_devices; i++) {
        queues[i] = clCreateCommandQueue(context, devices[i], 0, NULL);
    }

    cl_program program = program_from_file("fwcl.cl", context);
    cl_mem buffer_i_j = clCreateBuffer(context, CL_MEM_READ_WRITE, block_size_bytes, NULL, NULL);
    cl_mem buffer_i_k = clCreateBuffer(context, CL_MEM_READ_WRITE, block_size_bytes, NULL, NULL);
    cl_mem buffer_k_j = clCreateBuffer(context, CL_MEM_READ_WRITE, block_size_bytes, NULL, NULL);
    cl_kernel apsp_kernel = clCreateKernel(program, "floyd_warshall_block", NULL);
    clSetKernelArg(apsp_kernel, 0, sizeof(cl_mem), &buffer_i_j);
    clSetKernelArg(apsp_kernel, 1, sizeof(cl_mem), &buffer_i_k);
    clSetKernelArg(apsp_kernel, 2, sizeof(cl_mem), &buffer_k_j);
    size_t work_offset[2] = {0, 0};
    size_t global_size[2] = {block_size, block_size};
    size_t local_size[2] = {cl_local_size, cl_local_size};

    for (size_t k = 0; k < blocks_num; k++) {
        printf("Diagonal block: (%zu, %zu)\n", k, k);
        block_to_device(m, k, k, queues[0], buffer_i_k);
        block_to_device(m, k, k, queues[0], buffer_k_j);
        block_to_device(m, k, k, queues[0], buffer_i_j);
        
        for (size_t i = 0; i < blocks_num; i++) {
            size_t i_idx = (i + k) % blocks_num;
            block_to_device(m, i_idx, k, queues[0], buffer_i_k);            

            for (size_t j = 0; j < blocks_num; j++) {
                size_t j_idx = (j + k) % blocks_num;
                block_to_device(m, k, j_idx, queues[0], buffer_k_j);            
                block_to_device(m, i_idx, j_idx, queues[0], buffer_i_j);
                clEnqueueNDRangeKernel(queues[0], apsp_kernel, 2, work_offset, global_size, local_size, 0, NULL, NULL);
                block_to_host(m, i_idx, j_idx, queues[0], buffer_i_j);
            }

        }
        
    }

}

// Create a random symmetric matrix. Creates the same matrix every time.
struct matrix random_symmetric_matrix(unsigned int rows) {
    struct matrix m;
    m.rows = rows;
    m.elements = malloc(sizeof(float) * rows * rows);
    srand48(3);
    
    for (unsigned int x = 0; x < rows; x++) {
        for (unsigned int y = x + 1; y < rows; y++) {
            m.elements[y*rows + x] = m.elements[x*rows + y] = drand48() * 99;
            if (x == y) {
                m.elements[y*rows + x] = 0;
            }
        }
    }
    
    return m;
}

// Serial Floyd-Warshall implementation
void floyd_warshall(struct matrix m) {
    for (unsigned int k = 0; k < m.rows; k++) {
        for (unsigned int i = 0; i < m.rows; i++){
            for (unsigned int j = 0; j < m.rows; j++){
                float old = m.elements[j * m.rows + i];
                float new = m.elements[k * m.rows + i] + 
                            m.elements[j * m.rows + k];
                m.elements[j * m.rows + i] = fminf(new, old);
            }
        }
    }
}

// Copy block (x, y) from matrix m to block_data
void copy_block(struct matrix m, unsigned int x, unsigned int y, float *block_data) {
    for (unsigned int i = 0; i < block_size; i++) {
        for (unsigned int j = 0; j < block_size; j++) {
            block_data[j * block_size + i] = m.elements[(y * block_size + j) * m.rows + x * block_size + i];
        }
    }
}

// Write given block_data back to the matrix m at (x, y)
void sync_block(struct matrix m, unsigned int x, unsigned int y, float *block_data) {
    for (unsigned int i = 0; i < block_size; i++) {
        for (unsigned int j = 0; j < block_size; j++) {
            m.elements[(y * block_size + j) * m.rows + x * block_size + i] = block_data[j * block_size + i];
        }
    }
}

// Runs the Floyd-Warshall algorithm on block (x, y) for intermediate nodes in block (w, w)
void floyd_warshall_block(struct matrix m, unsigned int x, unsigned int y, unsigned int w) {
//    fprintf(stdout, "Calculating APSP for (%d, %d, %d)\n", x, y, w);
    float *block_i_j = malloc(sizeof(float) * block_size * block_size);
    float *block_i_k = malloc(sizeof(float) * block_size * block_size);
    float *block_k_j = malloc(sizeof(float) * block_size * block_size);
    
    copy_block(m, x, y, block_i_j); // c(i, j)
    copy_block(m, x, w, block_i_k); // c(i, k)
    copy_block(m, w, y, block_k_j); // c(k, j)
    
    for (unsigned int k = 0; k < block_size; k++) {
        for (unsigned int i = 0; i < block_size; i++) {
            for (unsigned int j = 0; j < block_size; j++) {
                float old = block_i_j[j * block_size + i];
                float new = block_i_k[k * block_size + i]
                          + block_k_j[j * block_size + k];
//                fprintf(stdout, "%.2f + %.2f < %.2f?\n", block_i_k[k * block_size + i], block_k_j[j * block_size + k], old);
                block_i_j[j * block_size + i] = fminf(new, old);
            }
        }
    }

    sync_block(m, x, y, block_i_j);
    free(block_i_j);
    free(block_i_k);
    free(block_k_j);
}

// Blocked Floyd-Warshall implementation
void floyd_warshall_blocked(struct matrix m) {

    for (unsigned int w = 0; w < m.rows / block_size; w++) {
        floyd_warshall_block(m, w, w, w);

        for (unsigned int x = 0; x < m.rows / block_size; x++) {
            floyd_warshall_block(m, x, w, w);
        }

        for (unsigned int y = 0; y < m.rows / block_size; y++) {
            floyd_warshall_block(m, w, y, w);
        }

        for (unsigned int x = 0; x < m.rows / block_size; x++) {
            for (unsigned int y = 0; y < m.rows / block_size; y++) {
                if (x != y) {
                    floyd_warshall_block(m, x, y, w);
                }
            }
        }

    }

}

// Print matrix to stdout
void print_matrix(struct matrix m) {
    if (m.rows > 16) {
        // Too large to print.
        return;
    }
    
    for (unsigned int y = 0; y < m.rows; y++) {
        fprintf(stdout, "║ ");

        for (unsigned int x = 0; x < m.rows; x++) {
            fprintf(stdout, "%2.0f ", m.elements[y * m.rows + x]);
        }
        
        fprintf(stdout, "║\n");
    }

}