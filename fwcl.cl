// Called with three blocks, where block_i_j is updated with the new weights 
// i -> j. Variables block_i_k and block_k_j contain costs of paths i -> k and
// k -> j. All block must be of dimensions block_size x block_size. Blocks 
// should be as large as possible within the constraints of the maximum global
// memory allocation.
kernel void floyd_warshall_block(global float *block_i_j, global read_only float *block_i_k, global read_only float *block_k_j) {
    // The block is further subdivided into blocks of LOCAL_SIZE x LOCAL_SIZE.
    size_t xg = get_global_id(0);
    size_t yg = get_global_id(1);
    size_t xl = get_local_id(0);
    size_t yl = get_local_id(1);
    
    // Cache areas in local memory for the target data (i.e. i -> j) and
    // and i -> k, k -> j (updated for each k).
    float cost_i_j = block_i_j[yg * BLOCK_SIZE + xg];
    local float cache_i_k[LOCAL_SIZE];
    local float cache_k_j[LOCAL_SIZE];

    for (size_t k = 0; k < BLOCK_SIZE; k++) {

        // Copy the relevant weights i -> k and k -> i to local memory for
        // faster access.
        if (xl == 0) {
            cache_i_k[yl] = block_i_k[yg * BLOCK_SIZE + k];
        }
        if (yl == 0) {
            cache_k_j[xl] = block_k_j[k * BLOCK_SIZE + xg];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        cost_i_j = min(cost_i_j, cache_i_k[yl] + cache_k_j[xl]);
    }
 
    block_i_j[yg * BLOCK_SIZE + xg] = cost_i_j;
}