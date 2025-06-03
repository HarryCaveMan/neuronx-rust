#pragma once

#include <nrt/nrt.h>
#include <cstdint>  // For uint32_t
#include <stdexcept>  // For std::runtime_error
#include "rust/cxx.h"  // For rust::Str, rust::Slice

using std::static_cast;

namespace neuronx_rs::runtime {
    class NrtVersion {
        public:
            static std::unique_ptr<Version> get() {
                auto version = std::unique_ptr<Version>(new Version());
                return version;
            }            
            uint64_t major() const { return _version.rt_major; }
            uint64_t minor() const { return _version.rt_minor; }
            uint64_t patch() const { return _version.rt_patch; }
            uint64_t maintenance() const { return _version.rt_maintenance; }
            std::string detail() const { return std::string(_version.rt_detail); }
            std::string git_hash() const { return std::string(_version.git_hash); }
            
        private:
            Version() {
                nrt_version_t *version{nullptr};
                NRT_STATUS status = nrt_get_version(version, sizeof(nrt_version_t));
                if (status != NRT_SUCCESS) {
                    throw std::runtime_error("Failed to get Neuron runtime version");
                }
                _version = *version;
            }
            nrt_version_t _version;
    };
    // Initialize neuron runtime
    // Returns NRT status code (0 for success)
    uint32_t neuronx_init() {return static_cast<uint32_t>(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW,"",""));}
    // Close neuron runtime
    // Returns NRT status code (0 for success)
    uint32_t neuronx_close() {return static_cast<uint32_t>(nrt_close());}
    // Get the version of the neuron runtime
    unique_ptr<NrtVersion> neuronx_version() {
        return uniqe_ptr<NrtVersion>(new NrtVersion());
    }
    // Get the number of neuron cores on this hardware
    uint32_t get_nc_count() {
        uint32_t count{0};
        NRT_STATUS status = nrt_get_nc_count(&count);
        if (status != NRT_SUCCESS) {
            throw std::runtime_error("Failed to get Neuron core count");
        }
        return count;
    }
    // Get the number of visible neuron cores
    uint32_t get_visible_nc_count() {
        uint32_t count{0};
        NRT_STATUS status = nrt_get_visible_nc_count(&count);
        if (status != NRT_SUCCESS) {
            throw std::runtime_error("Failed to get visible Neuron core count");
        }
        return count;
    }
    
}