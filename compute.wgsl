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

[[block]]
struct MatrixData {
    data : [[stride(8)]] array<Complex>;
};

[[group(0), binding(0)]]
var<storage, read> src : State;
[[group(0), binding(1)]]
var<storage, read_write> dst : State;
[[group(0), binding(2)]]
var<storage, read> mat_indices : MatrixIndices;
[[group(0), binding(3)]]
var<storage, read> mat_data : MatrixData;


[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let total = arrayLength(&src.state);
    let matn = arrayLength(&mat_indices.indices);
    let index = global_id.x;
    if (index >= total) {
        return;
    }

    var matrix_mask: u32 = 0u;
    for(var j: u32 = 0u; j < matn; j = j + 1u) {
        matrix_mask = matrix_mask | (1u << mat_indices.indices[j]);
    }
    var template_index: u32 = index & !matrix_mask;

    var matrix_row: u32 = 0u;
    for(var j: u32 = 0u; j < matn; j = j + 1u) {
        matrix_row = matrix_row << 1u;
        matrix_row = matrix_row | (index >> mat_indices.indices[j]) & 1u;
    }

    var real_acc: f32 = 0.0;
    var imag_acc: f32 = 0.0;
    for(var matrix_column: u32 = 0u; matrix_column < (1u << matn); matrix_column = matrix_column + 1u) {
        var matrix_index: u32 = matrix_row*(1u << matn) + matrix_column;

        var source_index = template_index;
        for(var j: u32 = 0u; j < matn; j = j + 1u) {
            source_index = source_index | ((matrix_column >> j) & 1u) << mat_indices.indices[j];
        }

        real_acc = real_acc + (mat_data.data[matrix_index].real * src.state[source_index].real) - (mat_data.data[matrix_index].imag * src.state[source_index].imag);
        imag_acc = imag_acc + (mat_data.data[matrix_index].real * src.state[source_index].imag) + (mat_data.data[matrix_index].imag * src.state[source_index].real);
    }

    dst.state[index].real = real_acc;
    dst.state[index].imag = imag_acc;
}