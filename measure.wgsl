struct Complex {
    real: f32;
    imag: f32;
};

[[block]]
struct State {
  state : [[stride(8)]] array<Complex>;
};

[[block]]
struct MatrixIndices {
    indices : [[stride(4)]] array<u32>;
};


[[group(0), binding(0)]]
var<storage, read> src : State;
[[group(0), binding(1)]]
var<storage, read_write> dst : State;
[[group(0), binding(2)]]
var<storage, read> mat_indices : MatrixIndices;


[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let total = arrayLength(&src.state);
    let matn = mat_indices.indices[0];
    let index = global_id.x;

    if (index >= (1u << matn)) {
        return;
    }

    var n: u32 = 1u;
    loop {
        if (((total >> n) & 1u) == 1u) { break; }
        n = n + 1u;
    }
    var remaining_n = n - matn;

    // The measured bits are all set and in position.
    var positioned_mask: u32 = 0u;
    var positioned_bits: u32 = 0u;
    for(var j: u32 = 0u; j < matn; j = j + 1u) {
        var picked = (index >> j) & 1u;
        var positioned_bit = picked << mat_indices.indices[j+1u];
        positioned_bits = positioned_bits | positioned_bit;
        positioned_mask = positioned_mask | (1u << mat_indices.indices[j+1u]);
    }

    var real_acc: f32 = 0.0;
    for(var remaining_bits: u32 = 0u; remaining_bits < (1u << remaining_n); remaining_bits = remaining_bits + 1u) {
        var source_index = positioned_bits;

        var s: u32 = 0u;
        for(var j: u32 = 0u; j < n; j = j + 1u) {
            if (((positioned_mask >> s) & 1u) == 0u) {
                var next_bit = (remaining_bits >> s) & 1u;
                source_index = source_index | (next_bit << j);
                s = s + 1u;
            }
        }

        real_acc = real_acc + (src.state[source_index].real * src.state[source_index].real) + (src.state[source_index].imag * src.state[source_index].imag);
    }

    dst.state[index].real = real_acc;
    dst.state[index].imag = 0.0;











    var matrix_mask: u32 = 0u;
    for(var j: u32 = 0u; j < matn; j = j + 1u) {
        matrix_mask = matrix_mask | (1u << mat_indices.indices[j+1u]);
    }
    var template_index: u32 = index & !matrix_mask;

    var matrix_row: u32 = 0u;
    for(var j: u32 = 0u; j < matn; j = j + 1u) {
        matrix_row = matrix_row << 1u;
        matrix_row = matrix_row | (index >> mat_indices.indices[j+1u]) & 1u;
    }

    var real_acc: f32 = 0.0;
    for(var matrix_column: u32 = 0u; matrix_column < (1u << matn); matrix_column = matrix_column + 1u) {
        var matrix_index: u32 = matrix_row*(1u << matn) + matrix_column;

        var source_index = template_index;
        for(var j: u32 = 0u; j < matn; j = j + 1u) {
            source_index = source_index | ((matrix_column >> j) & 1u) << mat_indices.indices[j+1u];
        }

        real_acc = real_acc + (src.state[source_index].real * src.state[source_index].real) + (src.state[source_index].imag * src.state[source_index].imag);
    }

    dst.state[index].real = real_acc;
    dst.state[index].imag = 0.0;
}