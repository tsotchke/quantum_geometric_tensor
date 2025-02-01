#include <quantum_geometric/learning/data_loader.h>
#include <quantum_geometric/core/memory_pool.h>
#include <quantum_geometric/core/quantum_geometric_core.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <curl/curl.h>
#include <zlib.h>

// MNIST file URLs
#define MNIST_TRAIN_IMAGES "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
#define MNIST_TRAIN_LABELS "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
#define MNIST_TEST_IMAGES "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
#define MNIST_TEST_LABELS "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

// CIFAR-10 file URL
#define CIFAR10_URL "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

// Helper struct for memory buffer
typedef struct {
    char* data;
    size_t size;
} memory_buffer_t;

// CURL write callback
static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t realsize = size * nmemb;
    memory_buffer_t* mem = (memory_buffer_t*)userp;

    char* ptr = realloc(mem->data, mem->size + realsize + 1);
    if (!ptr) return 0;  // Out of memory

    mem->data = ptr;
    memcpy(&(mem->data[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->data[mem->size] = 0;

    return realsize;
}

// Download file using CURL
static memory_buffer_t* download_file(const char* url) {
    CURL* curl = curl_easy_init();
    if (!curl) return NULL;

    memory_buffer_t* buffer = malloc(sizeof(memory_buffer_t));
    if (!buffer) {
        curl_easy_cleanup(curl);
        return NULL;
    }

    buffer->data = malloc(1);
    buffer->size = 0;

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)buffer);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        free(buffer->data);
        free(buffer);
        return NULL;
    }

    return buffer;
}

// Decompress gzipped data
static memory_buffer_t* decompress_gzip(const memory_buffer_t* compressed) {
    z_stream strm = {0};
    if (inflateInit2(&strm, 16 + MAX_WBITS) != Z_OK) return NULL;

    memory_buffer_t* decompressed = malloc(sizeof(memory_buffer_t));
    if (!decompressed) {
        inflateEnd(&strm);
        return NULL;
    }

    size_t chunk_size = 16384;
    decompressed->data = malloc(chunk_size);
    decompressed->size = 0;

    strm.next_in = (Bytef*)compressed->data;
    strm.avail_in = compressed->size;

    do {
        strm.avail_out = chunk_size;
        strm.next_out = (Bytef*)(decompressed->data + decompressed->size);

        int ret = inflate(&strm, Z_NO_FLUSH);
        if (ret != Z_OK && ret != Z_STREAM_END) {
            free(decompressed->data);
            free(decompressed);
            inflateEnd(&strm);
            return NULL;
        }

        decompressed->size += chunk_size - strm.avail_out;
        decompressed->data = realloc(decompressed->data, decompressed->size + chunk_size);

    } while (strm.avail_out == 0);

    inflateEnd(&strm);
    return decompressed;
}

dataset_t* quantum_load_mnist(dataset_config_t config) {
    // Download and decompress training images
    memory_buffer_t* compressed = download_file(MNIST_TRAIN_IMAGES);
    if (!compressed) return NULL;

    memory_buffer_t* decompressed = decompress_gzip(compressed);
    free(compressed->data);
    free(compressed);
    if (!decompressed) return NULL;

    // Parse MNIST format
    uint32_t magic = *((uint32_t*)decompressed->data);
    magic = ((magic & 0xFF000000) >> 24) | 
            ((magic & 0x00FF0000) >> 8) |
            ((magic & 0x0000FF00) << 8) |
            ((magic & 0x000000FF) << 24);

    if (magic != 0x803) {
        free(decompressed->data);
        free(decompressed);
        return NULL;
    }

    uint32_t num_images = *((uint32_t*)(decompressed->data + 4));
    uint32_t num_rows = *((uint32_t*)(decompressed->data + 8));
    uint32_t num_cols = *((uint32_t*)(decompressed->data + 12));

    num_images = ((num_images & 0xFF000000) >> 24) |
                ((num_images & 0x00FF0000) >> 8) |
                ((num_images & 0x0000FF00) << 8) |
                ((num_images & 0x000000FF) << 24);

    num_rows = ((num_rows & 0xFF000000) >> 24) |
               ((num_rows & 0x00FF0000) >> 8) |
               ((num_rows & 0x0000FF00) << 8) |
               ((num_rows & 0x000000FF) << 24);

    num_cols = ((num_cols & 0xFF000000) >> 24) |
               ((num_cols & 0x00FF0000) >> 8) |
               ((num_cols & 0x0000FF00) << 8) |
               ((num_cols & 0x000000FF) << 24);

    // Create dataset
    dataset_t* dataset = allocate_dataset(num_images, num_rows * num_cols, 10);
    if (!dataset) {
        free(decompressed->data);
        free(decompressed);
        return NULL;
    }

    // Copy image data
    unsigned char* image_data = (unsigned char*)(decompressed->data + 16);
    for (size_t i = 0; i < num_images; i++) {
        for (size_t j = 0; j < num_rows * num_cols; j++) {
            dataset->features[i][j] = image_data[i * num_rows * num_cols + j] / 255.0f;
        }
    }

    free(decompressed->data);
    free(decompressed);

    // Download and process labels
    compressed = download_file(MNIST_TRAIN_LABELS);
    if (!compressed) {
        quantum_dataset_destroy(dataset);
        return NULL;
    }

    decompressed = decompress_gzip(compressed);
    free(compressed->data);
    free(compressed);
    if (!decompressed) {
        quantum_dataset_destroy(dataset);
        return NULL;
    }

    // Copy label data
    unsigned char* label_data = (unsigned char*)(decompressed->data + 8);
    for (size_t i = 0; i < num_images; i++) {
        dataset->labels[i] = (float)label_data[i];
    }

    free(decompressed->data);
    free(decompressed);

    // Apply normalization if requested
    if (config.normalize) {
        quantum_normalize_data(dataset, config.normalization_method);
    }

    return dataset;
}

dataset_t* quantum_load_cifar10(dataset_config_t config) {
    // Download CIFAR-10 dataset
    memory_buffer_t* compressed = download_file(CIFAR10_URL);
    if (!compressed) return NULL;

    // Create dataset structure
    const size_t num_images = 50000;  // Training set size
    const size_t image_size = 32 * 32 * 3;  // 32x32 RGB images
    dataset_t* dataset = allocate_dataset(num_images, image_size, 10);
    if (!dataset) {
        free(compressed->data);
        free(compressed);
        return NULL;
    }

    // Extract and process data
    // Note: This is a simplified version. In practice, you'd need to properly
    // extract the tar.gz file and process the binary data format.
    
    // Process each batch file
    unsigned char* data_ptr = (unsigned char*)compressed->data;
    size_t image_offset = 0;
    
    for (size_t batch = 0; batch < 5; batch++) {
        for (size_t i = 0; i < 10000; i++) {
            // First byte is the label
            dataset->labels[image_offset + i] = (float)*data_ptr++;
            
            // Next 3072 bytes are the image data (32x32x3)
            for (size_t j = 0; j < image_size; j++) {
                dataset->features[image_offset + i][j] = *data_ptr++ / 255.0f;
            }
        }
        image_offset += 10000;
    }

    free(compressed->data);
    free(compressed);

    // Apply normalization if requested
    if (config.normalize) {
        quantum_normalize_data(dataset, config.normalization_method);
    }

    return dataset;
}

dataset_t* quantum_load_uci(const char* name, dataset_config_t config) {
    // Construct UCI dataset URL
    char url[1024];
    snprintf(url, sizeof(url), "https://archive.ics.uci.edu/ml/machine-learning-databases/%s/%s.data",
             name, name);

    // Download dataset
    memory_buffer_t* buffer = download_file(url);
    if (!buffer) return NULL;

    // Count rows and columns
    size_t num_rows = 0;
    size_t num_cols = 0;
    char* data = buffer->data;
    
    // Count columns in first row
    char* ptr = strchr(data, '\n');
    if (ptr) {
        char* token = strtok(data, ",");
        while (token) {
            num_cols++;
            token = strtok(NULL, ",");
        }
        num_cols--; // Last column is label
    }

    // Count rows
    while ((ptr = strchr(data, '\n'))) {
        num_rows++;
        data = ptr + 1;
    }

    // Create dataset
    dataset_t* dataset = allocate_dataset(num_rows, num_cols - 1, 0);
    if (!dataset) {
        free(buffer->data);
        free(buffer);
        return NULL;
    }

    // Parse data
    data = buffer->data;
    for (size_t i = 0; i < num_rows; i++) {
        char* line = strtok(data, "\n");
        if (!line) break;

        // Parse features
        char* token = strtok(line, ",");
        for (size_t j = 0; j < num_cols - 1; j++) {
            if (!token) break;
            dataset->features[i][j] = atof(token);
            token = strtok(NULL, ",");
        }

        // Parse label
        if (token) {
            dataset->labels[i] = atof(token);
        }

        data = NULL; // For subsequent strtok calls
    }

    free(buffer->data);
    free(buffer);

    // Apply normalization if requested
    if (config.normalize) {
        quantum_normalize_data(dataset, config.normalization_method);
    }

    return dataset;
}
