/*
 * Copyright 2025, Amazon.com, Inc. or its affiliates. All Rights Reserved
 */
#pragma once
#include <nrt/nrt.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
This is a public interface used by both the fetch api which allows near real-time querying of captured events, and inspect profiling which saves captured events to disk as well as other profiling functions.
*/

//------------------------------------------------
// Section: System Trace Capture
//------------------------------------------------

typedef struct nrt_sys_trace_config nrt_sys_trace_config_t;

/** Allocate memory for the options structure which is needed to
 * start profiling using nrt_sys_trace_start. This will set all options to defaults. The reason we use an _allocate function is so that users don't need to know the size or implementation details of the config struct.
 * 
 * @param options[in] - pointer to a pointer to options nrt_inspect_config struct
 * 
 */
NRT_STATUS nrt_sys_trace_config_allocate(nrt_sys_trace_config_t **options);

/** Set all fields of the nrt_inspect_config structure to their default values.
 * 
 * @param options[in,out] - Pointer to an nrt_inspect_config structure.
 */
void nrt_sys_trace_config_set_defaults(nrt_sys_trace_config_t *options);

/** Free up memory allocated for the options structure which is needed to
 * start profiling using nrt_sys_trace_start
 * 
 * @param options[in] - pointer to an options nrt_inspect_config struct
 * 
 */
void nrt_sys_trace_config_free(nrt_sys_trace_config_t *options);

/**
 * @brief Sets max number of events that can be stored across all ring buffers
 *
 * @param[in,out] options Pointer to the options structure.
 * @param[in] max_events_per_nc Max number of events that can be stored in each ring buffer.
 */
void nrt_sys_trace_config_set_max_events_per_nc(nrt_sys_trace_config_t *options, uint64_t max_events_per_nc);

// Initiailzation for system trace capture including allocating memory for event ring buffers
NRT_STATUS nrt_sys_trace_start(nrt_sys_trace_config_t *options);

// Teardown for system trace capture including freeing allocated memory for event ring buffers
NRT_STATUS nrt_sys_trace_stop();

//------------------------------------------------
// Section: System Trace Fetch
//------------------------------------------------

typedef struct nrt_sys_trace_fetch_options nrt_sys_trace_fetch_options_t;
NRT_STATUS nrt_sys_trace_fetch_options_allocate(nrt_sys_trace_fetch_options_t **options);
void nrt_sys_trace_fetch_options_set_defaults(nrt_sys_trace_fetch_options_t *options);
void nrt_sys_trace_fetch_options_free(nrt_sys_trace_fetch_options_t *options);
// Max number of events to fetch per NeuronCore
void nrt_sys_trace_fetch_options_set_max_events_per_nc(nrt_sys_trace_fetch_options_t *options, uint64_t max_events_per_nc);
// Fetch events only for specified NeuronCore
void nrt_sys_trace_fetch_options_set_nc_idx(nrt_sys_trace_fetch_options_t *options, uint64_t nc_idx);

/** Fetches system trace events from process memory and returns them as a JSON-formatted string. Once events are fetched, they cannot be fetched again.
 *
 * @param[out] buffer       On successful return, will point to a dynamically allocated, null-terminated
 *                          JSON string containing the trace events. Memory for the output buffer is
 *                          allocated internally; therefore, the caller should not allocate or initialize
 *                          the buffer before calling this function. Instead, the caller must initialize
 *                          the buffer pointer to NULL and, after a successful call, is responsible for
 *                          freeing the allocated memory by calling nrt_sys_trace_buffer_free(buffer).
 *
 * @param[out] written_size A pointer to a size_t variable that will be set to the number of bytes written
 *                          into the allocated buffer.
 *
 * @param[in] options       Pointer to options such as max number of events to fetch.
 *
 * @return NRT_SUCCESS on success.
 *
 * Usage example:
 *     char *buffer;
 *     size_t written_size;
 *     nrt_sys_trace_fetch_options_t *options;
 *     nrt_sys_trace_fetch_options_allocate(&options);
 *     nrt_sys_trace_fetch_options_set_nc_idx(options, 0); // Fetch events from NeuronCore 0 only instead of all
 *     nrt_sys_trace_fetch_options_set_max_events_per_nc(options, 10000); // Fetch up to 10,000 events instead of all
 *     nrt_sys_trace_fetch_events(&buffer, &written_size, options);
 *     // or if you want to use the default options:
 *     nrt_sys_trace_fetch_events(&buffer, &written_size, NULL);
 *     // finally free the buffer when the events are no longer needed:
 *     nrt_sys_trace_buffer_free(buffer)
 */
NRT_STATUS nrt_sys_trace_fetch_events(char **buffer, size_t *written_size, const nrt_sys_trace_fetch_options_t *options);

/** Free the buffer allocated by nrt_sys_trace_fetch_events. Should be called after the events are no longer needed.
 *
 * @param buffer [in]        - Pointer to buffer to be freed.
 *
 * @return NRT_SUCCESS on success.
 */
void nrt_sys_trace_buffer_free(char *buffer);

#ifdef __cplusplus
}
#endif
