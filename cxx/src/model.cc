#include "model.h"
#include "tensor.h"
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <fcntl.h>

using std::unique_ptr;
using std::runtime_error;
using std::static_cast;
using std::move;
using std::unordered_map;
using std::string;
using neuron_rs::data::IoTensors;

namespace neuron_rs::model {

    // RAII wrapper for mmap
    struct ModelMappedMemory {
        void *data;
        size_t size;
        
        ModelMappedMemory(const string& path) : data(nullptr), size(0) {
            int fd = open(path.c_str(), O_RDONLY);
            struct stat sb;
            if (fstat(fd, &sb) != 0) {
                close(fd);
                throw runtime_error("Cannot stat file: " + path);
            }
            size = sb.st_size;
            data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (data == MAP_FAILED) {
                close(fd);
                throw runtime_error("Cannot mmap file: " + path);
            }
            close(fd);
        }
        
        ~ModelMappedMemory() {
            if (data != nullptr && data != MAP_FAILED) {
                munmap(data, size);
            }
        }
    };

    // Factory method - load from file
    unique_ptr<Model> Model::from_neff_file(
        const string& path,
        int32_t start_nc,
        int32_t nc_count
    ) {
        ModelMappedMemory buffer{path};        
        return from_neff_buffer(buffer.data, buffer.size, start_nc, nc_count);
    }

    // Factory method - load from memory
    unique_ptr<Model> Model::from_neff_buffer(
        const void* data,
        size_t size,
        int32_t start_nc,
        int32_t nc_count
    ) {
        nrt_model_t *handle = nullptr;
        nrt_tensor_info_array_t *tensor_info = nullptr;
        NRT_STATUS status = nrt_load(
            data,
            size,
            start_nc,
            nc_count,
            &handle
        );
        if (status != NRT_SUCCESS) {
            if(handle != nullptr) {
                nrt_unload(handle);
            }
            throw runtime_error("Model loading failed");
        }
        unique_ptr<nrt_model_t, ModelHandleDestructor> handle_ptr{handle};
        status = nrt_get_tensor_info_array(handle_ptr.get(),&tensor_info);
        if (status != NRT_SUCCESS) {
            if(handle != nullptr) {
                nrt_unload(handle);
            }
            throw runtime_error("Failed to get tensor info array");
        }
        unique_ptr<IoTensors> io_tensors{IoTensors::empty_from_info_array(tensor_info)};
        return unique_ptr<Model>(new Model(move(handle_ptr),move(io_tensors)));
    }

    // Bind buffer to model i/o tensor to the model
    uint32_t Model::bind(const string &name, nrt_tensor_usage_t usage, void *buffer) {
        NRT_STATUS status = NRT_FAILURE;
        Tensor* tensor;
        if (usage == NRT_TENSOR_USAGE_INPUT) {
            tensor = _io_tensors->inputs()->get_tensor(name);
        } else {
            tensor = _io_tensors->outputs()->get_tensor(name);
        }
        if (!tensor) {
            return static_cast<uint32_t>(NRT_INVALID);
        }
        status = tensor->attach_buffer(buffer);      
        return static_cast<uint32_t>(status);
    }

    // Execute inference
    uint32_t Model::execute() {
        return static_cast<uint32_t>(nrt_execute(
            _handle.get(),
            _io_tensors->inputs()->handle(),
            _io_tensors->outputs()->handle()
        ));
    }

}