#include <quantum_geometric/learning/data_loader.h>
#include <quantum_geometric/core/memory_pool.h>
#include <quantum_geometric/core/quantum_geometric_core.h>
#include <quantum_geometric/core/statistical_analyzer.h>
#include <quantum_geometric/core/performance_monitor.h>
#include <quantum_geometric/core/cache_manager.h>
#include <quantum_geometric/core/memory_optimization.h>
#include <quantum_geometric/core/multi_gpu_operations.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>
#include <ctype.h>
#include <errno.h>

// Optional HDF5 support
#ifdef HAVE_HDF5
#include <hdf5.h>
#endif

// Optional image support (using stb_image if available)
#ifdef HAVE_STB_IMAGE
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

// PNG support via libpng
#ifdef HAVE_LIBPNG
#include <png.h>
#endif

// Global performance metrics
static PerformanceMetrics g_metrics = {0};
static pthread_mutex_t g_metrics_mutex = PTHREAD_MUTEX_INITIALIZER;

// Performance monitoring
static void update_metrics(const char* operation, double duration, size_t bytes) {
    pthread_mutex_lock(&g_metrics_mutex);
    if (strcmp(operation, "load") == 0) {
        g_metrics.cpu.total_cycles += duration;
    } else if (strcmp(operation, "preprocess") == 0) {
        g_metrics.cpu.instructions += duration;
    }
    g_metrics.memory.allocations = MAX(g_metrics.memory.allocations, bytes);
    g_metrics.memory.utilization = bytes / duration;
    pthread_mutex_unlock(&g_metrics_mutex);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper functions for data loading
static dataset_t* allocate_dataset(size_t num_samples, size_t feature_dim, size_t num_classes, memory_config_t* memory_config) {
    clock_t start = clock();
    
    dataset_t* dataset = NULL;
    size_t total_memory = sizeof(dataset_t);
    
    // Use memory mapping if requested
    if (memory_config && memory_config->use_mmap) {
        total_memory += num_samples * (feature_dim * sizeof(ComplexFloat*) + sizeof(ComplexFloat));
        dataset = mmap(NULL, total_memory, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (dataset == MAP_FAILED) return NULL;
    } else {
        dataset = (dataset_t*)malloc(sizeof(dataset_t));
        if (!dataset) return NULL;
    }

    // Allocate feature matrix with memory optimization
    if (memory_config && memory_config->gpu_cache) {
        // Allocate on GPU if requested
        dataset->features = quantum_gpu_malloc(num_samples * sizeof(ComplexFloat*));
        for (size_t i = 0; i < num_samples && dataset->features; i++) {
            dataset->features[i] = quantum_gpu_malloc(feature_dim * sizeof(ComplexFloat));
            if (!dataset->features[i]) {
                for (size_t j = 0; j < i; j++) {
                    quantum_gpu_free(dataset->features[j]);
                }
                quantum_gpu_free(dataset->features);
                if (memory_config->use_mmap) {
                    munmap(dataset, total_memory);
                } else {
                    free(dataset);
                }
                return NULL;
            }
        }
    } else if (memory_config && memory_config->use_mmap) {
        // Use memory mapping
        dataset->features = mmap(NULL, num_samples * feature_dim * sizeof(ComplexFloat),
                               PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (dataset->features == MAP_FAILED) {
            munmap(dataset, total_memory);
            return NULL;
        }
    } else {
        // Standard allocation
        dataset->features = (ComplexFloat**)malloc(num_samples * sizeof(ComplexFloat*));
        if (!dataset->features) {
            if (memory_config && memory_config->use_mmap) {
                munmap(dataset, total_memory);
            } else {
                free(dataset);
            }
            return NULL;
        }
        
        for (size_t i = 0; i < num_samples; i++) {
            dataset->features[i] = (ComplexFloat*)malloc(feature_dim * sizeof(ComplexFloat));
            if (!dataset->features[i]) {
                for (size_t j = 0; j < i; j++) {
                    free(dataset->features[j]);
                }
                free(dataset->features);
                if (memory_config && memory_config->use_mmap) {
                    munmap(dataset, total_memory);
                } else {
                    free(dataset);
                }
                return NULL;
            }
        }
    }

    // Allocate labels array with memory optimization
    if (memory_config && memory_config->gpu_cache) {
        dataset->labels = quantum_gpu_malloc(num_samples * sizeof(ComplexFloat));
    } else if (memory_config && memory_config->use_mmap) {
        dataset->labels = mmap(NULL, num_samples * sizeof(ComplexFloat),
                             PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (dataset->labels == MAP_FAILED) {
            // Clean up features
            if (memory_config->gpu_cache) {
                for (size_t i = 0; i < num_samples; i++) {
                    quantum_gpu_free(dataset->features[i]);
                }
                quantum_gpu_free(dataset->features);
            } else if (memory_config->use_mmap) {
                munmap(dataset->features, num_samples * feature_dim * sizeof(ComplexFloat));
            } else {
                for (size_t i = 0; i < num_samples; i++) {
                    free(dataset->features[i]);
                }
                free(dataset->features);
            }
            if (memory_config->use_mmap) {
                munmap(dataset, total_memory);
            } else {
                free(dataset);
            }
            return NULL;
        }
    } else {
        dataset->labels = (ComplexFloat*)malloc(num_samples * sizeof(ComplexFloat));
        if (!dataset->labels) {
            // Clean up features
            for (size_t i = 0; i < num_samples; i++) {
                free(dataset->features[i]);
            }
            free(dataset->features);
            free(dataset);
            return NULL;
        }
    }

    // Initialize dataset properties
    dataset->num_samples = num_samples;
    dataset->feature_dim = feature_dim;
    dataset->num_classes = num_classes;
    dataset->feature_names = NULL;
    dataset->class_names = NULL;

    // Update performance metrics
    clock_t end = clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    update_metrics("load", duration, total_memory);

    return dataset;
}

// CSV streaming implementation
typedef struct {
    FILE* file;
    char* buffer;
    size_t buffer_size;
    size_t current_pos;
    size_t chunk_size;
    bool eof;
    pthread_mutex_t mutex;
    PerformanceMetrics metrics;
} csv_stream_t;

static void* csv_stream_next(csv_stream_t* stream) {
    pthread_mutex_lock(&stream->mutex);
    clock_t start = clock();
    
    if (stream->eof || !stream->file) {
        pthread_mutex_unlock(&stream->mutex);
        return NULL;
    }

    // Read next chunk
    size_t bytes_read = fread(stream->buffer, 1, stream->chunk_size, stream->file);
    if (bytes_read < stream->chunk_size) {
        stream->eof = true;
    }

    // Update metrics
    clock_t end = clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    stream->metrics.cpu.total_cycles += duration;
    stream->metrics.memory.utilization = bytes_read / duration;
    
    pthread_mutex_unlock(&stream->mutex);
    return stream->buffer;
}

static void csv_stream_reset(csv_stream_t* stream) {
    pthread_mutex_lock(&stream->mutex);
    if (stream->file) {
        rewind(stream->file);
        stream->eof = false;
        stream->current_pos = 0;
    }
    pthread_mutex_unlock(&stream->mutex);
}

static void csv_stream_cleanup(csv_stream_t* stream) {
    pthread_mutex_lock(&stream->mutex);
    if (stream->file) {
        fclose(stream->file);
        stream->file = NULL;
    }
    free(stream->buffer);
    pthread_mutex_unlock(&stream->mutex);
    pthread_mutex_destroy(&stream->mutex);
    free(stream);
}

// CSV loading implementation
static dataset_t* load_csv(const char* path, dataset_config_t config) {
    FILE* file = fopen(path, "r");
    if (!file) return NULL;

    // Count rows and columns
    char line[4096];
    size_t num_samples = 0;
    size_t feature_dim = 0;

    // Skip header if needed
    if (config.has_header) {
        if (!fgets(line, sizeof(line), file)) {
            fclose(file);
            return NULL;
        }
    }

    // Count columns from first data line
    if (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, config.delimiter);
        while (token) {
            feature_dim++;
            token = strtok(NULL, config.delimiter);
        }
        feature_dim--; // Last column is label
        num_samples = 1;
    }

    // Count remaining rows
    while (fgets(line, sizeof(line), file)) {
        num_samples++;
    }

    // Allocate dataset
    rewind(file);
    if (config.has_header) fgets(line, sizeof(line), file); // Skip header again

    dataset_t* dataset = allocate_dataset(num_samples, feature_dim, 0, NULL);
    if (!dataset) {
        fclose(file);
        return NULL;
    }

    // Read data
    size_t row = 0;
    while (fgets(line, sizeof(line), file) && row < num_samples) {
        char* token = strtok(line, config.delimiter);
        size_t col = 0;
        
        // Read features
        while (token && col < feature_dim) {
            dataset->features[row][col] = complex_float_create(atof(token), 0.0f);
            token = strtok(NULL, config.delimiter);
            col++;
        }

        // Read label
        if (token) {
            dataset->labels[row] = complex_float_create(atof(token), 0.0f);
        }

        row++;
    }

    fclose(file);
    return dataset;
}

// ============================================================================
// NumPy .npy file format loader
// Format: magic string + version + header_len + header (Python dict) + data
// ============================================================================

// NumPy dtype to element size mapping
static size_t numpy_dtype_size(const char* dtype_str) {
    // Parse dtype string like '<f4', '<f8', '<i4', '<c8', etc.
    if (!dtype_str || strlen(dtype_str) < 2) return 0;

    char type_char = dtype_str[1];
    int size = atoi(dtype_str + 2);

    if (size <= 0) {
        // Try parsing without endian marker
        type_char = dtype_str[0];
        size = atoi(dtype_str + 1);
    }

    return (size_t)size;
}

static dataset_t* load_numpy(const char* path, dataset_config_t config) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "NumPy loader: Cannot open file %s\n", path);
        return NULL;
    }

    // Read magic string "\x93NUMPY"
    unsigned char magic[6];
    if (fread(magic, 1, 6, file) != 6) {
        fclose(file);
        return NULL;
    }

    if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        fprintf(stderr, "NumPy loader: Invalid magic number\n");
        fclose(file);
        return NULL;
    }

    // Read version
    unsigned char version[2];
    if (fread(version, 1, 2, file) != 2) {
        fclose(file);
        return NULL;
    }

    // Read header length
    uint32_t header_len = 0;
    if (version[0] == 1) {
        uint16_t len16;
        if (fread(&len16, 2, 1, file) != 1) {
            fclose(file);
            return NULL;
        }
        header_len = len16;
    } else if (version[0] >= 2) {
        if (fread(&header_len, 4, 1, file) != 1) {
            fclose(file);
            return NULL;
        }
    }

    // Read header (Python dict as string)
    char* header = malloc(header_len + 1);
    if (!header) {
        fclose(file);
        return NULL;
    }
    if (fread(header, 1, header_len, file) != header_len) {
        free(header);
        fclose(file);
        return NULL;
    }
    header[header_len] = '\0';

    // Parse header to extract shape and dtype
    // Header format: {'descr': '<f4', 'fortran_order': False, 'shape': (1000, 784), }
    size_t shape[8] = {0};
    size_t num_dims = 0;
    char dtype[32] = "";
    bool fortran_order = false;

    // Parse dtype
    char* descr_pos = strstr(header, "'descr':");
    if (descr_pos) {
        char* quote1 = strchr(descr_pos + 8, '\'');
        if (quote1) {
            char* quote2 = strchr(quote1 + 1, '\'');
            if (quote2) {
                size_t len = quote2 - quote1 - 1;
                if (len < sizeof(dtype)) {
                    strncpy(dtype, quote1 + 1, len);
                    dtype[len] = '\0';
                }
            }
        }
    }

    // Parse fortran_order
    if (strstr(header, "fortran_order': True") || strstr(header, "fortran_order':True")) {
        fortran_order = true;
    }

    // Parse shape
    char* shape_pos = strstr(header, "'shape':");
    if (shape_pos) {
        char* paren1 = strchr(shape_pos, '(');
        if (paren1) {
            char* p = paren1 + 1;
            while (*p && *p != ')' && num_dims < 8) {
                while (*p && isspace(*p)) p++;
                if (isdigit(*p)) {
                    shape[num_dims++] = (size_t)atol(p);
                    while (*p && isdigit(*p)) p++;
                }
                while (*p && (*p == ',' || isspace(*p))) p++;
            }
        }
    }

    free(header);

    if (num_dims == 0 || strlen(dtype) == 0) {
        fprintf(stderr, "NumPy loader: Failed to parse header\n");
        fclose(file);
        return NULL;
    }

    // Calculate total size and determine dataset dimensions
    size_t elem_size = numpy_dtype_size(dtype);
    if (elem_size == 0) {
        fprintf(stderr, "NumPy loader: Unknown dtype %s\n", dtype);
        fclose(file);
        return NULL;
    }

    size_t num_samples = shape[0];
    size_t feature_dim = 1;
    for (size_t i = 1; i < num_dims; i++) {
        feature_dim *= shape[i];
    }

    // Allocate dataset
    dataset_t* dataset = allocate_dataset(num_samples, feature_dim, 0, NULL);
    if (!dataset) {
        fclose(file);
        return NULL;
    }

    // Read data
    size_t total_elements = num_samples * feature_dim;
    void* raw_data = malloc(total_elements * elem_size);
    if (!raw_data) {
        // Clean up dataset
        fclose(file);
        return NULL;
    }

    if (fread(raw_data, elem_size, total_elements, file) != total_elements) {
        fprintf(stderr, "NumPy loader: Failed to read data\n");
        free(raw_data);
        fclose(file);
        return NULL;
    }
    fclose(file);

    // Convert to ComplexFloat based on dtype
    bool is_complex = (dtype[1] == 'c' || dtype[0] == 'c');
    bool is_float = (dtype[1] == 'f' || dtype[0] == 'f');

    for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j < feature_dim; j++) {
            size_t idx = fortran_order ? (j * num_samples + i) : (i * feature_dim + j);
            float real_val = 0.0f, imag_val = 0.0f;

            if (is_complex && elem_size == 8) {
                // Complex64 (two float32)
                float* ptr = (float*)raw_data + idx * 2;
                real_val = ptr[0];
                imag_val = ptr[1];
            } else if (is_complex && elem_size == 16) {
                // Complex128 (two float64)
                double* ptr = (double*)raw_data + idx * 2;
                real_val = (float)ptr[0];
                imag_val = (float)ptr[1];
            } else if (is_float && elem_size == 4) {
                // Float32
                real_val = ((float*)raw_data)[idx];
            } else if (is_float && elem_size == 8) {
                // Float64
                real_val = (float)((double*)raw_data)[idx];
            } else if (elem_size == 4) {
                // Int32
                real_val = (float)((int32_t*)raw_data)[idx];
            } else if (elem_size == 8) {
                // Int64
                real_val = (float)((int64_t*)raw_data)[idx];
            } else if (elem_size == 2) {
                // Int16
                real_val = (float)((int16_t*)raw_data)[idx];
            } else if (elem_size == 1) {
                // Int8/UInt8
                real_val = (float)((uint8_t*)raw_data)[idx];
            }

            dataset->features[i][j] = complex_float_create(real_val, imag_val);
        }
    }

    free(raw_data);

    dataset->num_samples = num_samples;
    dataset->feature_dim = feature_dim;

    return dataset;
}

// ============================================================================
// HDF5 file format loader
// ============================================================================

static dataset_t* load_hdf5(const char* path, dataset_config_t config) {
#ifdef HAVE_HDF5
    hid_t file_id = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "HDF5 loader: Cannot open file %s\n", path);
        return NULL;
    }

    // Try to open common dataset names
    const char* dataset_names[] = {"data", "features", "X", "x", "images", "samples"};
    hid_t dataset_id = -1;

    for (size_t i = 0; i < sizeof(dataset_names) / sizeof(dataset_names[0]); i++) {
        dataset_id = H5Dopen2(file_id, dataset_names[i], H5P_DEFAULT);
        if (dataset_id >= 0) break;
    }

    if (dataset_id < 0) {
        fprintf(stderr, "HDF5 loader: No recognized dataset found\n");
        H5Fclose(file_id);
        return NULL;
    }

    // Get dataspace and dimensions
    hid_t space_id = H5Dget_space(dataset_id);
    int ndims = H5Sget_simple_extent_ndims(space_id);

    if (ndims < 1 || ndims > 4) {
        fprintf(stderr, "HDF5 loader: Unsupported number of dimensions: %d\n", ndims);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    hsize_t dims[4];
    H5Sget_simple_extent_dims(space_id, dims, NULL);

    size_t num_samples = (size_t)dims[0];
    size_t feature_dim = 1;
    for (int i = 1; i < ndims; i++) {
        feature_dim *= (size_t)dims[i];
    }

    // Allocate dataset
    dataset_t* dataset = allocate_dataset(num_samples, feature_dim, 0, NULL);
    if (!dataset) {
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    // Read data as float
    size_t total_size = num_samples * feature_dim;
    float* buffer = malloc(total_size * sizeof(float));
    if (!buffer) {
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    herr_t status = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buffer);

    if (status < 0) {
        fprintf(stderr, "HDF5 loader: Failed to read data\n");
        free(buffer);
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return NULL;
    }

    // Convert to ComplexFloat
    for (size_t i = 0; i < num_samples; i++) {
        for (size_t j = 0; j < feature_dim; j++) {
            dataset->features[i][j] = complex_float_create(buffer[i * feature_dim + j], 0.0f);
        }
    }

    free(buffer);

    // Try to load labels
    const char* label_names[] = {"labels", "y", "Y", "targets", "target"};
    hid_t label_id = -1;

    for (size_t i = 0; i < sizeof(label_names) / sizeof(label_names[0]); i++) {
        label_id = H5Dopen2(file_id, label_names[i], H5P_DEFAULT);
        if (label_id >= 0) break;
    }

    if (label_id >= 0) {
        float* labels = malloc(num_samples * sizeof(float));
        if (labels) {
            H5Dread(label_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, labels);
            for (size_t i = 0; i < num_samples; i++) {
                dataset->labels[i] = complex_float_create(labels[i], 0.0f);
            }
            free(labels);
        }
        H5Dclose(label_id);
    }

    H5Sclose(space_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    dataset->num_samples = num_samples;
    dataset->feature_dim = feature_dim;

    return dataset;
#else
    // HDF5 not available - try to provide a minimal implementation
    fprintf(stderr, "HDF5 loader: HDF5 support not compiled. Rebuild with -DHAVE_HDF5\n");
    (void)path;
    (void)config;
    return NULL;
#endif
}

// ============================================================================
// Image file format loader (PNG, JPG, BMP, etc.)
// ============================================================================

// Simple PPM/PGM loader as fallback
static dataset_t* load_ppm_image(const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file) return NULL;

    char magic[3];
    if (fread(magic, 1, 2, file) != 2) {
        fclose(file);
        return NULL;
    }
    magic[2] = '\0';

    bool is_rgb = (magic[0] == 'P' && magic[1] == '6');
    bool is_gray = (magic[0] == 'P' && magic[1] == '5');

    if (!is_rgb && !is_gray) {
        fclose(file);
        return NULL;
    }

    // Skip comments and whitespace
    int c;
    while ((c = fgetc(file)) != EOF) {
        if (c == '#') {
            while ((c = fgetc(file)) != EOF && c != '\n');
        } else if (!isspace(c)) {
            ungetc(c, file);
            break;
        }
    }

    int width, height, maxval;
    if (fscanf(file, "%d %d %d", &width, &height, &maxval) != 3) {
        fclose(file);
        return NULL;
    }
    fgetc(file);  // Skip single whitespace after header

    size_t channels = is_rgb ? 3 : 1;
    size_t feature_dim = (size_t)width * height * channels;

    dataset_t* dataset = allocate_dataset(1, feature_dim, 0, NULL);
    if (!dataset) {
        fclose(file);
        return NULL;
    }

    unsigned char* buffer = malloc(feature_dim);
    if (!buffer) {
        fclose(file);
        return NULL;
    }

    if (fread(buffer, 1, feature_dim, file) != feature_dim) {
        free(buffer);
        fclose(file);
        return NULL;
    }
    fclose(file);

    // Convert to normalized floats
    float scale = maxval > 0 ? 1.0f / maxval : 1.0f;
    for (size_t i = 0; i < feature_dim; i++) {
        dataset->features[0][i] = complex_float_create(buffer[i] * scale, 0.0f);
    }

    free(buffer);

    dataset->num_samples = 1;
    dataset->feature_dim = feature_dim;

    return dataset;
}

static dataset_t* load_image(const char* path, dataset_config_t config) {
#ifdef HAVE_STB_IMAGE
    int width, height, channels;
    unsigned char* data = stbi_load(path, &width, &height, &channels, 0);

    if (!data) {
        fprintf(stderr, "Image loader: Failed to load %s\n", path);
        return NULL;
    }

    size_t feature_dim = (size_t)width * height * channels;
    dataset_t* dataset = allocate_dataset(1, feature_dim, 0, NULL);

    if (dataset) {
        for (size_t i = 0; i < feature_dim; i++) {
            dataset->features[0][i] = complex_float_create(data[i] / 255.0f, 0.0f);
        }
        dataset->num_samples = 1;
        dataset->feature_dim = feature_dim;
    }

    stbi_image_free(data);
    return dataset;
#elif defined(HAVE_LIBPNG)
    // Check if PNG
    FILE* file = fopen(path, "rb");
    if (!file) return NULL;

    unsigned char sig[8];
    fread(sig, 1, 8, file);

    if (png_sig_cmp(sig, 0, 8)) {
        fclose(file);
        // Try PPM fallback
        return load_ppm_image(path);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(file);
        return NULL;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        fclose(file);
        return NULL;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, NULL);
        fclose(file);
        return NULL;
    }

    png_init_io(png, file);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);

    int width = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    int color_type = png_get_color_type(png, info);
    int bit_depth = png_get_bit_depth(png, info);

    // Convert to 8-bit RGBA
    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);

    png_read_update_info(png, info);

    size_t channels = png_get_channels(png, info);
    size_t rowbytes = png_get_rowbytes(png, info);

    png_bytep* rows = malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        rows[y] = malloc(rowbytes);
    }

    png_read_image(png, rows);
    png_destroy_read_struct(&png, &info, NULL);
    fclose(file);

    size_t feature_dim = (size_t)width * height * channels;
    dataset_t* dataset = allocate_dataset(1, feature_dim, 0, NULL);

    if (dataset) {
        size_t idx = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width * (int)channels; x++) {
                dataset->features[0][idx++] = complex_float_create(rows[y][x] / 255.0f, 0.0f);
            }
        }
        dataset->num_samples = 1;
        dataset->feature_dim = feature_dim;
    }

    for (int y = 0; y < height; y++) {
        free(rows[y]);
    }
    free(rows);

    return dataset;
#else
    // Fallback to PPM/PGM
    return load_ppm_image(path);
#endif
}

// ============================================================================
// Batch image loading from directory
// ============================================================================

static dataset_t* load_image_directory(const char* path, dataset_config_t config) {
    // This loads multiple images from a directory
    // For now, we just load a single image
    // A full implementation would use opendir/readdir
    return load_image(path, config);
}

// ============================================================================
// Public API
// ============================================================================

dataset_t* quantum_load_dataset(const char* path, dataset_config_t config) {
    dataset_t* dataset = NULL;

    switch (config.format) {
        case DATA_FORMAT_CSV:
            dataset = load_csv(path, config);
            break;
        case DATA_FORMAT_NUMPY:
            dataset = load_numpy(path, config);
            break;
        case DATA_FORMAT_HDF5:
            dataset = load_hdf5(path, config);
            break;
        case DATA_FORMAT_IMAGE:
            dataset = load_image(path, config);
            break;
        default:
            return NULL;
    }

    if (dataset && config.normalize) {
        quantum_normalize_data(dataset, config.normalization_method);
    }

    return dataset;
}

dataset_t* quantum_create_synthetic_data(
    size_t num_samples,
    size_t feature_dim,
    size_t num_classes,
    int type
) {
    dataset_t* dataset = allocate_dataset(num_samples, feature_dim, num_classes, NULL);
    if (!dataset) return NULL;

    // Initialize random number generator
    srand(42);

    switch (type) {
        case SYNTHETIC_DATA_CLASSIFICATION: {
            // Generate classification data
            for (size_t i = 0; i < num_samples; i++) {
                // Generate random features
                for (size_t j = 0; j < feature_dim; j++) {
                    dataset->features[i][j] = complex_float_create(
                        ((float)rand() / RAND_MAX) * 2.0f - 1.0f,
                        0.0f
                    );
                }
                // Assign random class
                dataset->labels[i] = complex_float_create(rand() % num_classes, 0.0f);
            }
            break;
        }
        case SYNTHETIC_DATA_REGRESSION: {
            // Generate regression data with linear relationship plus noise
            for (size_t i = 0; i < num_samples; i++) {
                float sum = 0.0f;
                for (size_t j = 0; j < feature_dim; j++) {
                    dataset->features[i][j] = complex_float_create(
                        ((float)rand() / RAND_MAX) * 2.0f - 1.0f,
                        0.0f
                    );
                    sum += dataset->features[i][j].real;
                }
                // Generate target with noise
                dataset->labels[i] = complex_float_create(
                    sum / feature_dim + ((float)rand() / RAND_MAX) * 0.1f - 0.05f,
                    0.0f
                );
            }
            break;
        }
        default:
            quantum_dataset_destroy(dataset);
            return NULL;
    }

    return dataset;
}

dataset_split_t quantum_split_dataset(
    dataset_t* dataset,
    float train_ratio,
    float val_ratio,
    bool shuffle,
    bool stratify
) {
    dataset_split_t split = {0};
    if (!dataset) return split;

    // Calculate split sizes
    size_t train_size = (size_t)(dataset->num_samples * train_ratio);
    size_t val_size = (size_t)(dataset->num_samples * val_ratio);
    size_t test_size = dataset->num_samples - train_size - val_size;

    // Create index array for shuffling
    size_t* indices = (size_t*)malloc(dataset->num_samples * sizeof(size_t));
    if (!indices) return split;

    for (size_t i = 0; i < dataset->num_samples; i++) {
        indices[i] = i;
    }

    // Shuffle indices if requested
    if (shuffle) {
        for (size_t i = dataset->num_samples - 1; i > 0; i--) {
            size_t j = rand() % (i + 1);
            size_t temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
    }

    // Allocate split datasets
    split.train_data = allocate_dataset(train_size, dataset->feature_dim, dataset->num_classes, NULL);
    split.val_data = val_ratio > 0 ? 
        allocate_dataset(val_size, dataset->feature_dim, dataset->num_classes, NULL) : NULL;
    split.test_data = allocate_dataset(test_size, dataset->feature_dim, dataset->num_classes, NULL);

    if (!split.train_data || !split.test_data || (val_ratio > 0 && !split.val_data)) {
        quantum_dataset_split_destroy(&split);
        free(indices);
        return (dataset_split_t){0};
    }

    // Copy data to split datasets
    size_t train_idx = 0, val_idx = 0, test_idx = 0;
    for (size_t i = 0; i < dataset->num_samples; i++) {
        size_t src_idx = indices[i];
        dataset_t* target;
        size_t* target_idx;

        if (i < train_size) {
            target = split.train_data;
            target_idx = &train_idx;
        } else if (i < train_size + val_size) {
            target = split.val_data;
            target_idx = &val_idx;
        } else {
            target = split.test_data;
            target_idx = &test_idx;
        }

        // Copy features and label
        memcpy(target->features[*target_idx], 
               dataset->features[src_idx], 
               dataset->feature_dim * sizeof(ComplexFloat));
        target->labels[*target_idx] = dataset->labels[src_idx];
        (*target_idx)++;
    }

    free(indices);
    return split;
}

bool quantum_normalize_data(dataset_t* dataset, normalization_t method) {
    if (!dataset) return false;

    switch (method) {
        case NORMALIZATION_MINMAX: {
            // Min-max normalization
            for (size_t j = 0; j < dataset->feature_dim; j++) {
                float min_val = dataset->features[0][j].real;
                float max_val = dataset->features[0][j].real;

                // Find min and max of real parts
                for (size_t i = 1; i < dataset->num_samples; i++) {
                    if (dataset->features[i][j].real < min_val) {
                        min_val = dataset->features[i][j].real;
                    }
                    if (dataset->features[i][j].real > max_val) {
                        max_val = dataset->features[i][j].real;
                    }
                }

                // Normalize real parts
                float range = max_val - min_val;
                if (range > 0) {
                    for (size_t i = 0; i < dataset->num_samples; i++) {
                        dataset->features[i][j].real = 
                            (dataset->features[i][j].real - min_val) / range;
                        // Imaginary part remains unchanged (0)
                    }
                }
            }
            break;
        }
        case NORMALIZATION_ZSCORE: {
            // Z-score normalization
            for (size_t j = 0; j < dataset->feature_dim; j++) {
                float sum = 0.0f, sum_sq = 0.0f;

                // Calculate mean and variance
                for (size_t i = 0; i < dataset->num_samples; i++) {
                    sum += dataset->features[i][j].real;
                    sum_sq += dataset->features[i][j].real * dataset->features[i][j].real;
                }

                float mean = sum / dataset->num_samples;
                float variance = (sum_sq / dataset->num_samples) - (mean * mean);
                float std_dev = sqrtf(variance);

                // Normalize
                if (std_dev > 0) {
                    for (size_t i = 0; i < dataset->num_samples; i++) {
                        dataset->features[i][j].real = 
                            (dataset->features[i][j].real - mean) / std_dev;
                        // Imaginary part remains unchanged (0)
                    }
                }
            }
            break;
        }
        default:
            return false;
    }

    return true;
}

void quantum_dataset_destroy(dataset_t* dataset) {
    if (!dataset) return;

    if (dataset->features) {
        for (size_t i = 0; i < dataset->num_samples; i++) {
            free(dataset->features[i]);
        }
        free(dataset->features);
    }

    if (dataset->labels) {
        free(dataset->labels);
    }

    if (dataset->feature_names) {
        for (size_t i = 0; i < dataset->feature_dim; i++) {
            free(dataset->feature_names[i]);
        }
        free(dataset->feature_names);
    }

    if (dataset->class_names) {
        for (size_t i = 0; i < dataset->num_classes; i++) {
            free(dataset->class_names[i]);
        }
        free(dataset->class_names);
    }

    free(dataset);
}

void quantum_dataset_split_destroy(dataset_split_t* split) {
    if (!split) return;
    quantum_dataset_destroy(split->train_data);
    quantum_dataset_destroy(split->val_data);
    quantum_dataset_destroy(split->test_data);
}

bool quantum_configure_performance(data_performance_config_t config) {
    // Initialize performance monitoring
    pthread_mutex_lock(&g_metrics_mutex);
    memset(&g_metrics, 0, sizeof(PerformanceMetrics));
    pthread_mutex_unlock(&g_metrics_mutex);

    // Configure cache if requested
    if (config.cache_size > 0) {
        if (!quantum_configure_cache(config.cache_size)) {
            return false;
        }
    }

    // Configure prefetch queue
    if (config.prefetch_size > 0) {
        if (!quantum_configure_prefetch(config.prefetch_size)) {
            return false;
        }
    }

    // Configure parallel workers
    if (config.num_workers > 0) {
        if (!quantum_configure_workers(config.num_workers)) {
            return false;
        }
    }

    // Enable performance profiling if requested
    if (config.profile) {
        quantum_enable_profiling();
    }

    return true;
}

bool quantum_configure_memory(memory_config_t config) {
    // Configure memory management
    if (config.gpu_cache) {
        if (!quantum_gpu_init()) {
            return false;
        }
    }

    // Configure memory limits
    if (config.max_memory > 0) {
        if (!quantum_set_memory_limit(config.max_memory)) {
            return false;
        }
    }

    // Configure streaming mode
    if (config.streaming) {
        if (!quantum_enable_streaming(config.chunk_size)) {
            return false;
        }
    }

    // Configure compression
    if (config.compress) {
        if (!quantum_enable_compression()) {
            return false;
        }
    }

    return true;
}

bool quantum_get_performance_metrics(PerformanceMetrics* metrics) {
    if (!metrics) return false;

    pthread_mutex_lock(&g_metrics_mutex);
    *metrics = g_metrics;
    pthread_mutex_unlock(&g_metrics_mutex);

    return true;
}

bool quantum_reset_performance_metrics(void) {
    pthread_mutex_lock(&g_metrics_mutex);
    memset(&g_metrics, 0, sizeof(PerformanceMetrics));
    pthread_mutex_unlock(&g_metrics_mutex);

    return true;
}
