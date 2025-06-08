#include "model.h"
#include <fcntl.h>         // For open, O_RDONLY
#include <sys/mman.h>      // For mmap, munmap
#include <sys/stat.h>      // For fstat
#include <unistd.h>        // For close

using std::unique_ptr;
using std::static_cast;
using std::move;
using std::string;

namespace neuronx_rs::model {
    // RAII wrapper for mmap
    struct ModelMappedMemory {
        static uint32_t SUCCESS{0};
        static uint32_t STAT_FAIL{1};
        static uint32_t MMAP_FAIL{2};
        uint32_t status;
        void *data;
        size_t size;
        
        ModelMappedMemory(const string& path) : data(nullptr), size(0) {
            int fd{open(path.c_str(), O_RDONLY)};
            struct stat sb;
            if (fstat(fd, &sb) != 0) {
                close(fd);
                status = STAT_FAIL;
            }
            size = sb.st_size;
            data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (data == MAP_FAILED) {
                close(fd);
                status = MMAP_FAIL;
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
    ModelResult Model::from_neff_file(
        const string& path,
        int32_t start_nc,
        int32_t nc_count
    ) {
        ModelMappedMemory buffer{path};
        if (buffer.status != ModelMappedMemory::SUCCESS) {
            return ModelResult{
                nullptr,
                static_cast<uint32_t>(NRT_FAILURE);
            };
        }     
        return from_neff_buffer(buffer.data, buffer.size, start_nc, nc_count);
    }

    // Factory method - load from memory
    ModelResult Model::from_neff_buffer(
        const void* data,
        size_t size,
        int32_t start_nc,
        int32_t nc_count
    ) {
        nrt_model_t *raw_handle{nullptr};
        NRT_STATUS status{
                nrt_load(
                data,
                size,
                start_nc,
                nc_count,
                &raw_handle
            )
        };
        unique_ptr<nrt_model_t, ModelHandleDestructor> handle{raw_handle};
        nrt_tensor_info_array_t *tensor_info{nullptr};
        status = nrt_get_tensor_info_array(handle.get(),&tensor_info);
        return ModelResult{
            unique_ptr<Model>(new Model(move(handle))),
            static_cast<uint32_t>(status)
        };
    }

    // Create IoTensors from model's tensor info
    IoTensorsResult Model::get_new_io_tensors() {
        nrt_tensor_info_array_t *tensor_info{nullptr};
        NRT_STATUS status{nrt_get_tensor_info_array(_handle.get(), &tensor_info)};
        if (status != NRT_SUCCESS) {
            return IoTensorsResult{nullptr, static_cast<uint32_t>(status)};
        }
        IoTensorsResult io_tensors_result{IoTensors::empty_from_info_array(tensor_info)};
        return io_tensors_result;
    }

    // Execute inference
    uint32_t Model::execute(IoTensors *io_tensors) {
        return static_cast<uint32_t>(nrt_execute(
            _handle.get(),
            io_tensors->inputs()->handle(),
            io_tensors->outputs()->handle()
        ));
    }

}