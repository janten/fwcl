/*
Matrix layout:
This is the block matrix. Each block consists of BLOCK_SIZE * BLOCK_SIZE
elements. The group id at dimension 0 of each thread is the index of the
corresponding block. Generally, i, j, and k refer to indices in the adjacency 
matrix while x, y, and w are indices in the block matrix.

      x →
     ┏━━━┳━━━┳━━━┳━━━┳━━━┓
   y ┃   ┃   ┃   ┃   ┃   ┃ ⇠ Current strip. Since the matrix is symmetric, the
   ↓ ┡━━━╇━━━╇━━━╇━━━╇━━━┩    n-th row is identical to the n-th column and we
     │   │   │   │   │   │    can therefore compute the whole matrix using only
     ├───┼───┼───┼───┼───┤    row-based access patterns.
     │   │   │   │   │   │
     ┢━━━╈━━━╈━━━╈━━━╈━━━┪
     ┃   ┃   ┃   ┃ w ┃   ┃ ⇠ Current diagonal element and containing strip. The
     ┡━━━╇━━━╇━━━╇━━━╇━━━┩    strip also contains the (i, k) and (k, j) weights
     │   │   │   │   │   │    for all elements in block (x, y) at the indices x
     └───┴───┴───┴───┴───┘    and y.

*/

void copy_block(global float *strip, size_t x, local float *block_data);
void sync_block(global float *strip, size_t x, local float *block_data);

void copy_block(global float *strip, size_t x, local float *block_data) {
    size_t row_length = get_global_size(0);
    size_t i = get_local_id(0);
    size_t j = get_local_id(1);
    size_t strip_index = j * row_length + x * BLOCK_SIZE + i;
    block_data[j * BLOCK_SIZE + i] = strip[strip_index];
}

void sync_block(global float *strip, size_t x, local float *block_data) {
    size_t row_length = get_global_size(0);
    size_t i = get_local_id(0);
    size_t j = get_local_id(1);
    size_t strip_index = j * row_length + x * BLOCK_SIZE + i;
    strip[strip_index] = block_data[j * BLOCK_SIZE + i];
}

kernel void floyd_warshall_block(global float *x_strip, global float *k_strip, size_t w) {
    size_t y = get_group_id(0);
    size_t i = get_local_id(0);
    size_t j = get_local_id(1);
    
    local float block_i_j[BLOCK_SIZE * BLOCK_SIZE];
    local float block_i_k[BLOCK_SIZE * BLOCK_SIZE];
    local float block_k_j[BLOCK_SIZE * BLOCK_SIZE];
    copy_block(x_strip, y, block_i_j);
    copy_block(x_strip, w, block_i_k);
    copy_block(k_strip, y, block_k_j);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (unsigned int k = 0; k < BLOCK_SIZE; k++) {
        float old = block_i_j[j * BLOCK_SIZE + i];
        float new = block_i_k[k * BLOCK_SIZE + i]
                  + block_k_j[j * BLOCK_SIZE + k];
        block_i_j[j * BLOCK_SIZE + i] = min(new, old);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    sync_block(x_strip, y, block_i_j);
}