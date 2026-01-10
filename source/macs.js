
const macs = {};

macs.Calculator = class {

    constructor() {
    }

    calculate(node) {
        if (!node || !node.type || !node.type.name) {
            return null;
        }

        const type = node.type.name;
        // Check for PyTorch Conv2d, Linear
        // Check for ONNX Conv, Gemm, MatMul

        // Handle "aten::conv2d", "Conv", etc.
        const name = type.split('::').pop().toLowerCase();

        if (name === 'conv2d' || name === 'conv' || name === 'convolution' || name === 'conv1d' || name === 'conv3d') {
            return this._computeConv(node);
        } else if (name === 'linear' || name === 'gemm' || name === 'matmul') {
            return this._computeLinear(node);
        }

        return null;
    }

    _computeConv(node) {
        // PyTorch: input, weight, bias
        // ONNX: X, W, B
        const inputs = node.inputs;

        // inputs is array of Argument.
        // Argument has value which is array of Value.
        // Value has type -> shape.

        if (!inputs || inputs.length < 2) {
            return null;
        }

        // Assuming 2nd input is weight.
        // In Netron, inputs are named. We can check names or just indices for common ops.
        // PyTorch Conv2d: input, weight, bias
        // ONNX Conv: X, W, B

        // Input is usually index 0
        // Weight is usually index 1

        const weight = this._getTensorShape(inputs[1]);
        const outputs = node.outputs;
        if (!outputs || outputs.length < 1) {
            return null;
        }
        const output = this._getTensorShape(outputs[0]);

        if (!weight || !output) {
            return null;
        }

        // Weight shape: [OutChannels, InChannels/Groups, KernelH, KernelW] (PyTorch/ONNX)
        // Output shape: [N, OutChannels, OutH, OutW]

        if (output.length < 3 || weight.length < 3) {
            return null;
        }

        // Output elements count
        const outputElements = output.reduce((a, b) => a * b, 1);

        // Weight elements count
        const weightElements = weight.reduce((a, b) => a * b, 1);

        // Weight[0] is typically OutChannels (or M for ONNX)
        const outChannels = weight[0];

        if (outChannels === 0) return null;

        // Kernel size per output channel = Total Weights / OutChannels
        // This accounts for groups automatically.
        const kernelSize = weightElements / outChannels;

        const total = outputElements * kernelSize;

        return total;
    }

    _computeLinear(node) {
        // Linear(PyTorch): input, weight, bias. Weight: [OutFeatures, InFeatures]
        // Gemm(ONNX): A, B, C.
        // MatMul: A, B

        const inputs = node.inputs;
        const outputs = node.outputs;

        if (!inputs || inputs.length < 2 || !outputs || outputs.length < 1) {
            return null;
        }

        // const input0 = this._getTensorShape(inputs[0]);
        const input1 = this._getTensorShape(inputs[1]); // Weight
        const output = this._getTensorShape(outputs[0]);

        if (!output) return null;

        // For Linear/Gemm/MatMul, simpler estimation:
        // Output elements * "inner dimension size"

        // If we have weights (input1), we can infer the operation cost.
        if (input1) {
            // For PyTorch Linear: y = xA^T + b. A is [out_features, in_features]
            // Output is [..., out_features]
            // Each output element is a dot product of size in_features.
            // in_features is A.shape[1] (if transposed) or A.shape[0] if not?
            // PyTorch Linear weight is [out_features, in_features].
            // ONNX Gemm B is typically [in, out] but transB attribute changes it.

            // Heuristic:
            // The operation is basically a dot product.
            // If we assume standard matrix multiplication behavior involved in Linear/Dense layers.
            // Weight matrix has N elements.
            // Output matrix has M elements.
            // If this is a simple Linear layer, Weight is [Out, In]. Output is [Batch, Out].
            // MACs = Batch * Out * In = Output_Elements * In.
            // In = Weight_Elements / Out = Weight_Elements / Output_Last_Dim.

            // Let's try to deduce 'In' dimension from Weight.
            // If Weight is 2D [D1, D2].
            // If node is Linear (PyTorch), weight is [OutFeatures, InFeatures].
            // Output shape last dim is OutFeatures.

            const name = node.type.name.split('::').pop().toLowerCase();

            if (name === 'linear') {
                 if (input1.length >= 2) {
                     // PyTorch Linear: weight is [out_features, in_features]
                     // We count MACs.
                     const inFeatures = input1[1];
                     const outputElements = output.reduce((a, b) => a * b, 1);
                     return outputElements * inFeatures;
                 }
            }

            // For general MatMul/Gemm, it's harder without attributes (transA, transB).
            // But usually one dimension matches.

            if (name === 'matmul' || name === 'gemm') {
                // Approximate: Output elements * Common Dimension.
                // Common Dimension can be estimated from weight size / output channel size?
                // Or just use input0 last dim if available.

                const input0 = this._getTensorShape(inputs[0]);
                if (input0 && input0.length > 0) {
                    const K = input0[input0.length - 1]; // Last dim of input
                    const outputElements = output.reduce((a, b) => a * b, 1);
                    return outputElements * K;
                }
            }
        }

        return null;
    }

    _getTensorShape(parameter) {
        // parameter is an Argument (view.js/model.js terminology)
        // It has 'value' which is an array of Value objects.
        if (!parameter || !parameter.value) return null;

        // In view.js, node.inputs elements are Arguments.
        // argument.value is array of Value objects.

        const values = parameter.value;
        if (!Array.isArray(values) || values.length === 0) return null;

        const value = values[0];
        // Value has 'type' which is a TensorType (usually)
        if (!value || !value.type) return null;

        const type = value.type;
        if (!type.shape || !type.shape.dimensions) return null;

        const dims = type.shape.dimensions;

        // Filter and sanitize dimensions
        const cleanDims = dims.map(d => {
            if (typeof d === 'number') return d;
            if (d && typeof d === 'object' && d.toNumber) return d.toNumber(); // Long.js
            // If string or null (dynamic), treat as 1 for estimation per unit batch/dynamic-dim
            return 1;
        });

        return cleanDims;
    }
};

export const Calculator = macs.Calculator;
