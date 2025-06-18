/*
 * Copyright 2021, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */

#pragma once

#include <stddef.h>
#include <stdint.h>
// Use quoted includes in nrt headers including other nrt headers. Most clients
// (ptxla, jax, etc.) build with bazel, and bazel has issue with angle-brackets.
// See https://bazel.build/docs/bazel-and-cpp#include-paths for details.
#include "nrt/nrt_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Major and minor version of runtime. */
#define NRT_MAJOR_VERSION 2
#define NRT_MINOR_VERSION 0

typedef struct nrt_model nrt_model_t;

typedef struct nrt_tensor nrt_tensor_t;


/**
 * WARNING: Do not change the value of existing enums!
 * These values will be used by libnrt consumers, we
 * cannot change the defines under them, only append.
 */
typedef enum {
    NRT_TENSOR_PLACEMENT_DEVICE,
    NRT_TENSOR_PLACEMENT_HOST,
    NRT_TENSOR_PLACEMENT_VIRTUAL,
} nrt_tensor_placement_t;

typedef enum {
    NRT_FRAMEWORK_TYPE_INVALID = 0,             // Invalid
    NRT_FRAMEWORK_TYPE_NO_FW = 1,               // Framework less execution
    NRT_FRAMEWORK_TYPE_TENSORFLOW,              // Tensorflow
    NRT_FRAMEWORK_TYPE_PYTORCH,                 // Pytorch
    NRT_FRAMEWORK_TYPE_MXNET,                   // Mxnet
} nrt_framework_type_t;

enum {
    NRT_INSTANCE_UNKNOWN    = 0,
    NRT_INSTANCE_INF1       = 1,
    NRT_INSTANCE_TRN1       = 2,
    NRT_INSTANCE_TRN1N      = 3,
    NRT_INSTANCE_INF2       = 4,
    NRT_INSTANCE_TRN2       = 5,
    NRT_INSTANCE_TRN2N      = 6,
    NRT_INSTANCE_INF2E      = 7,
    NRT_INSTANCE_TRN2P      = 8,
    NRT_INSTANCE_TRN2U      = 9,
#ifdef MARIANA
    NRT_INSTANCE_TRN3       = 10
#endif
};

enum {
    NRT_INSTANCE_SIZE_1XL,
    NRT_INSTANCE_SIZE_2XL,
    NRT_INSTANCE_SIZE_4XL,
    NRT_INSTANCE_SIZE_6XL,
    NRT_INSTANCE_SIZE_8XL,
    NRT_INSTANCE_SIZE_24XL,
    NRT_INSTANCE_SIZE_32XL,
    NRT_INSTANCE_SIZE_48XL,
    NRT_INSTANCE_SIZE_3XL,
    // Note: Add new sizes right above this line to prevent breaking backward compatibility

    NRT_INSTANCE_SIZE_UNKNOWN,
    NRT_INSTANCE_SIZE_NUM = NRT_INSTANCE_SIZE_UNKNOWN,
};

typedef struct nrt_instance_info {
    uint32_t family;
    uint32_t size;
} nrt_instance_info_t;

NRT_STATUS nrt_get_instance_info(nrt_instance_info_t *info, size_t instance_info_len);

/** Initialize neuron runtime.
 *
 * @param framework[in]      - Type of the framework.
 * @param fw_version[in]     - Framework version as string. (eg 2.1)
 * @param fal_version[in]    - Framework Abstraction Layer version as string.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_init(nrt_framework_type_t framework, const char *fw_version, const char *fal_version);

/** Closes all the devices and cleans up the runtime state.
 */
void nrt_close();

/** Load given NEFF and place it in one or more neuron cores.
 *
 * @param neff_bytes[in]    - Pointer to NEFF data.
 * @param size[in]          - Length of the NEFF data.
 * @param start_vnc[in]     - Starting VNC index where the NEFF should be loaded(-1 means runtime would automatically load in first free VNC).
 * @param vnc_count[in]     - Number of VNCs to use(-1 means runtime would automatically determine the need).
 * @param model[out]        - Resulting model would be stored here.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_load(const void *neff_bytes, size_t size, int32_t start_vnc, int32_t vnc_count, nrt_model_t **model);

/** Load given NEFF for collective operations and place it in one or more neuron cores.
 *
 * Global NCCL communicator is created inside the API according to g_device_id and g_device_count.
 *
 * @param neff_bytes[in]        - Pointer to NEFF data.
 * @param size[in]              - Length of the NEFF data.
 * @param start_vnc[in]         - Starting VNC index where the NEFF should be loaded(-1 means runtime would automatically load in first free VNC).
 * @param vnc_count[in]         - Number of VNCs to use(-1 means runtime would automatically determine the need).
 * @param g_device_id[in]       - Global device ID participating collective operations
 * @param g_device_count[in]    - Number of devices participating collective operations
 * @param model[out]            - Resulting model would be stored here.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_load_collectives(const void *neff_bytes, size_t size, int32_t start_vnc, int32_t vnc_count,
                                uint32_t g_device_id, uint32_t g_device_count, nrt_model_t **model);

/** Unload given model and free up device and host resources.
 *
 * @param model - Model to unload.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_unload(nrt_model_t *model);

/** Get the number of VNCs used by a loaded model (deprecated)
 *
 * @param model[in] - Model.
 * @param vnc_count[out] - The number of VNCs used by the model.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_model_nc_count(const nrt_model_t *model, uint32_t *vnc_count);

/** Get the number of VNCs used by a loaded model
 *
 * @param model[in] - Model.
 * @param vnc_count[out] - The number of VNCs used by the model.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_model_vnc_count(const nrt_model_t *model, uint32_t *vnc_count);

/** Returns VirtualNeuronCores available in instance. (deprecated)
 *
 * @param vnc_count[out] - VirtualNeuronCores available in instance.
 *
 * @note This API can be called before nrt_init().
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_total_nc_count(uint32_t *vnc_count);

/** Returns VirtualNeuronCores available in instance.
 *
 * @param vnc_count[out] - VirtualNeuronCores available in instance.
 *
 * @note This API can be called before nrt_init().
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_total_vnc_count(uint32_t *vnc_count);

/** Returns VirtualNeuronCores visible to the application. (deprecated)
 *
 * @param vnc_count[out] - VirtualNeuronCores visible to the application.
 *
 * @note This API can be called before nrt_init().
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_visible_nc_count(uint32_t *vnc_count);

/** Returns VirtualNeuronCores visible to the application.
 *
 * @param vnc_count[out] - VirtualNeuronCores visible to the application.
 *
 * @note This API can be called before nrt_init().
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_visible_vnc_count(uint32_t *vnc_count);

/** A container to hold multiple tensors */
typedef void nrt_tensor_set_t;

/** Allocates a new tensor set.
 *
 * @param result[out]       - Pointer to newly allocated tensor set would be stored here.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_allocate_tensor_set(nrt_tensor_set_t **result);

/** Destroys given tensor_set and frees memory.
 *
 * @param tensor_set[in]    - Tensors set to be freed.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
void nrt_destroy_tensor_set(nrt_tensor_set_t **tensor_set);

/** Add/replace given tensor to tensor set
 *
 * @param tensor_set[in]    - Tensor set to which the tensor is added.
 * @param tensor_name[in]   - Name of the tensor.
 * @param tensor[in]        - Pointer to tensor. This pointer should be valid till nrt_destroy_tensor_set() is called.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_add_tensor_to_tensor_set(nrt_tensor_set_t *tensor_set, const char *tensor_name, nrt_tensor_t *tensor);

/** Get a tensor's info from a tensor set.
 *
 * @param tensor_set[in]    - Tensor set.
 * @param tensor_name[in]   - Name of the tensor.
 * @param tensor[out]       - Pointer to tensor would be stored here.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_get_tensor_from_tensor_set(nrt_tensor_set_t *tensor_set, const char *tensor_name, nrt_tensor_t **tensor);

/** Execute given model with given inputs and collect outputs.
 *
 * @param model[in] - Model to execute.
 * @param input_set[in] - Set of input tensors.
 * @param output_set[in] - Set of output tensors.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_execute(nrt_model_t *model, const nrt_tensor_set_t *input_set, nrt_tensor_set_t *output_set);

/** Execute given model with given inputs, repeat execution specified number of times and collect outputs.
 *
 * @param model[in] - Model to execute.
 * @param input_set[in] - Set of input tensors.
 * @param output_set[in] - Set of output tensors.
 * @param repeat_count[in] - Number of to repeat execution.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_execute_repeat(nrt_model_t *model, const nrt_tensor_set_t *input_set, nrt_tensor_set_t *output_set, int repeat_count);

/** Allocates a tensor that can be passed and used by a model for compute.
 *
 * @param tensor_placement[in]  - Where the tensor would be allocated (device, host, or virtual memory)
 * @param vnc[in]               - Virutal Neuron Core id to allocate the tensor on. Pass in -1 if allocating tensors on host memory.
 * @param size[in]              - Size in bytes of the tensor to allocate.
 * @param name[in]              - OPTIONAL. Name of the tensor.
 * @param tensor[out]           - Pointer to newly created tensor will be stored here.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_tensor_allocate(nrt_tensor_placement_t tensor_placement, int vnc, size_t size, const char *name, nrt_tensor_t **tensor);

/** Deallocates a tensor created by "nrt_tensor_allocate".
 *
 * @param tensor[in]    - Deallocates given tensor.
 *
 * @return None
 */
void nrt_tensor_free(nrt_tensor_t **tensor);

/** Copies data from tensor to passed in buffer.
 *
 * @param tensor[in]    - Tensor used to reference the tensor to read from.
 * @param buf[out]      - Buffer used to store data read from the tensor.
 * @param offset[in]    - Offset into the tensor to read from.
 * @param size[in]      - Number of bytes to read.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_tensor_read(const nrt_tensor_t *tensor, void *buf, size_t offset, size_t size);

/** Copies data from passed in buffer to tensor.
 *
 * @param tensor[in/out]    - Tensor used to reference the tensor to write to.
 * @param buf[in]           - Buffer used to store data to write to the tensor.
 * @param offset[in]        - Offset into the tensor to write to.
 * @param size[in]          - Number of bytes to write.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_tensor_write(nrt_tensor_t *tensor, const void *buf, size_t offset, size_t size);


/** Copies data between tensors.
 *
 * When copying between two device tensors, they must both be allocated on the SAME Neuron Core.
 * A NRT_INVALID will be returned in the failing case.
 *
 * @param src[in]           - Tensor to copy from.
 * @param src_offset[in]    - Offset into the source tensor to copy from.
 * @param dst[out]          - Tensor to copy to.
 * @param dst_offset[in]    - Offset into the destination tensor to copy to.
 * @param size[in]          - Number of bytes to copy.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_tensor_copy(const nrt_tensor_t *src, size_t src_offset, nrt_tensor_t *dst, size_t dst_offset, size_t size);

/** Gets the size of the passed in tensor.
 *
 * @param tensor[in]    - Tensor used to reference the tensor to get size of.
 *
 * @return Size of the tensor.
 */
size_t nrt_tensor_get_size(const nrt_tensor_t *tensor);

/** Set the memory + offset pointed to by tensor to value
 *
 * @param tensor[in]        - allocated tensor
 * @param offset[in]        - offset within the tensor
 * @param value[in]         - value to set with
 * @param size[in]          - size of memory to set
 *
 * @return 0 on success.
 */
NRT_STATUS nrt_tensor_memset(nrt_tensor_t *tensor, uint64_t offset, int value, size_t size);

/** Allocates an empty tensor, i.e. the tensor structure w/o any attached storage
 *
 * @param name[in]              - OPTIONAL. Name of the tensor.
 * @param tensor[out]           - Pointer to newly created tensor will be stored here.
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_tensor_allocate_empty(const char *name, nrt_tensor_t **tensor);

/** Attaches caller supplied buffer to a tensor.  Any storage previously attached to the tensor is detached
 *  and freed if was owned by the tensor.
 *  The buffer is supplied by the caller and must persist through the entire lifetime of the tensor.
 *
 * @param tensor[in]            - Tensor
 * @param buffer[in]            - Caller supplied buffer to use as tensor's storage
 * @param size[in]              - Buffer Size
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
NRT_STATUS nrt_tensor_attach_buffer(nrt_tensor_t *tensor, void *buffer, size_t size);

/** Creates a tensor to point to a slice of another tensor
 *  does not do a deep copy, just points the "slice" tensor storage to the "source" tensor storage
 *
 * @param tensor_source[in] - Tensor to point at
 * @param offset[in]        - Offset from the beginning of the source tensor to point at
 * @param size[in]          - Size of the slice
 * @param name[in]          - Optional name for the new tensor
 * @param tensor_slice[in]  - Newly allocated tensor to point to the storage of the source tensor
 *
 */
NRT_STATUS nrt_tensor_allocate_slice( const nrt_tensor_t *tensor_source, size_t offset, size_t size, const char *name, nrt_tensor_t **tensor_slice);

/** Given a tensor get the virtual address.
 *
 * @param tensor[in]        - Tensor for which the VA needs to be obtained
 *
 * @return va on success, NULL on failure.
 */
void *nrt_tensor_get_va(const nrt_tensor_t *tensor);

/** Returns on device allocation info for a tensor
 *
 * @param tensor[in]        - Tensor for which the information needs to be obtained
 * @param alloc_info[out]   - On device allocation information
 *
 * @return NRT_STATUS_SUCCESS on success.
 */
typedef struct nrt_tensor_device_allocation_info {
    uint64_t physical_address; // physical address in device memory space
    size_t size;               // allocation size, could be larger than the tensor size
    int hbm_index;             // which of the HBMs the tensor is placed
} nrt_tensor_device_allocation_info_t;
NRT_STATUS nrt_tensor_get_device_allocation_info(const nrt_tensor_t *tensor, nrt_tensor_device_allocation_info_t *alloc_info);

/**
 * @brief Get the anonymous file-descriptor of dma-buf associated with
 * a Neuron device memory region if it was registered for EFA peer direct
 *
 * @param addr[in]          - Device buffer virtual address
 * @param size[in]          - Device buffer size (in bytes)
 * @param fd[out]           - dma-buf fd
 *
 * @return NRT_SUCCESS on success
 */
NRT_STATUS nrt_get_dmabuf_fd(uint64_t va, uint64_t size, int* fd);


/**  Get the host based device id from the device id presented to runtime (which may container based device id)
 * @param neuron_dev[in]      - device id
 * @param host_device_id[out] - host device id
 * @return NRT_SUCCESS if call was successful, NRT_INVALID otherwise
 */
NRT_STATUS nrt_host_device_id_get( int neuron_dev, uint32_t *host_device_id);

/**  Return array of routing IDs indexed by host device ID. This is the definitive routing ID mapping provided from the driver 
 * @param coutn[in/out]           - [in] number of entries in the mapping table provided. [out] count of entries returned
 * @param host_did_to_rid_map[in] - table/map of routing IDs indexed by host device ID
 * @return NRT_SUCCESS if call was successful, NRT_INVALID otherwise
 */
NRT_STATUS nrt_host_device_id_rid_map_get(uint32_t *count, uint32_t *host_did_to_rid_map);

/**
 * Get the HBM virtual address and size for a specific HBM index.
 * @param device_id[in]         - Device ID
 * @param hbm_idx[in]           - HBM index
 * @param addr[out]             - Pointer to store the virtual address
 * @param size[out]             - Pointer to store the size of the HBM region
 * @return NRT_SUCCESS if call was successful and HBM region was mapped
 *         NRT_INVALID_HANDLE if there are no more HBM regions to map for this device
 *         NRT_INVALID if the interface isn't supported or for invalid parameters
 *         NRT_FAILURE for other errors
 */
NRT_STATUS nrt_get_hbm_mmap_va(int device_id, int hbm_idx, void **addr, size_t *size);

#ifdef __cplusplus
}
#endif
