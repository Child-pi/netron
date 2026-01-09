
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

        if (type === 'Conv2d' || type === 'Conv' || type === 'Convolution' || type === 'Conv1d' || type === 'Conv3d') {
            return this._computeConv2d(node);
        } else if (type === 'Linear' || type === 'Gemm' || type === 'MatMul') {
            return this._computeLinear(node);
        }

        return null;
    }

    _computeConv2d(node) {
        // PyTorch: input, weight, bias
        // ONNX: X, W, B
        const inputs = node.inputs;
        const outputs = node.outputs;

        if (inputs.length < 2 || outputs.length < 1) {
            return null;
        }

        // Assuming 2nd input is weight
        const weight = this._getTensorShape(inputs[1]);
        const output = this._getTensorShape(outputs[0]);

        if (!weight || !output) {
            return null;
        }

        // Weight shape: [OutChannels, InChannels/Groups, KernelH, KernelW] (PyTorch)
        // ONNX: [M, C/group, kH, kW]

        // Output shape: [N, OutChannels, OutH, OutW]

        if (output.length < 3 || weight.length < 3) {
            return null;
        }

        // Groups attribute
        let groups = 1;
        const attributes = node.attributes || [];
        const groupAttr = attributes.find(a => a.name === 'groups' || a.name === 'group');
        if (groupAttr) {
            groups = parseInt(groupAttr.value, 10);
        }

        // MACs = OutputElements * (WeightElements / OutChannels)
        // Or roughly: OutputN * OutputH * OutputW * OutChannels * (InChannels/Groups * KernelH * KernelW)
        // Wait, WeightElements = OutChannels * InChannels/Groups * KernelH * KernelW
        // So WeightElements / OutChannels = InChannels/Groups * KernelH * KernelW
        // This is the number of MACs per output pixel per output channel.

        // Total Output Elements = N * OutChannels * OutH * OutW

        // So Total MACs = TotalOutputElements * (WeightElements / OutChannels) ?
        // Let's verify.
        // One output pixel (one channel) is result of dot product of Kernel sized volume.
        // Kernel volume size = (InChannels/Groups) * KernelH * KernelW

        // Correct.

        // However, Weight shape[0] is usually OutChannels.
        const outChannels = weight[0];
        const weightElements = weight.reduce((a, b) => a * b, 1);
        const outputElements = output.reduce((a, b) => a * b, 1);

        if (outChannels === 0) return null;

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

        if (inputs.length < 2 || outputs.length < 1) {
            return null;
        }

        const input0 = this._getTensorShape(inputs[0]);
        const input1 = this._getTensorShape(inputs[1]);
        const output = this._getTensorShape(outputs[0]);

        if (!output) return null;

        // For Linear layer: y = xA^T + b.
        // x: [N, *, in_features]
        // A: [out_features, in_features]
        // y: [N, *, out_features]

        // MACs = OutputElements * in_features

        if (input1) {
            // Assume input1 is weight.
            // For PyTorch Linear, weight is [out_features, in_features]
            // in_features is input1[1].

            // For MatMul [N, M] x [M, K] -> [N, K]
            // MACs = N * K * M = OutputElements * M
            // M is the common dimension.

            // If we know it's a Linear layer where weights are constant and shape is known.
            if (node.type.name === 'Linear') {
                 if (input1.length >= 2) {
                     const inFeatures = input1[1];
                     const outputElements = output.reduce((a, b) => a * b, 1);
                     return outputElements * inFeatures;
                 }
            }

            if (node.type.name === 'Gemm' || node.type.name === 'MatMul') {
                // Determine common dimension M.
                // It is hard to be generic without knowing transposes etc.
                // But usually for MatMul A(..., M) * B(M, ...), the MACs is OutputElements * M.

                // Let's try to infer M from inputs if shapes are available.
                // MatMul: A[... M], B[M ...] -> Out
                if (input0 && input0.length > 0) {
                    const M = input0[input0.length - 1];
                    const outputElements = output.reduce((a, b) => a * b, 1);
                    return outputElements * M;
                }
            }
        }

        return null;
    }

    _getTensorShape(argument) {
        if (!argument || !argument.value) return null;

        let value = argument.value;
        if (Array.isArray(value)) {
            if (value.length === 0) return null;
            value = value[0]; // Assume first tensor
        }

        if (!value || !value.type || !value.type.shape || !value.type.shape.dimensions) {
            return null;
        }

        const dims = value.type.shape.dimensions;

        // If dimensions contain non-numbers (e.g. '?'), we can't calculate exact MACs.
        // But maybe we can treat '?' as 1 or ignore?
        // Usually batch size is dynamic. If we treat it as 1, we get MACs per sample.
        // But if user wants total MACs for the model inference as described by shapes...

        // "Mac數" usually implies for a single inference pass given the input shapes.
        // If input shape has '?', we can't compute.

        // However, many models have fixed shapes or '?' for batch size.
        // If I encounter '?', I will substitute 1 for batch dimension (usually first), or fail?
        // Let's substitute 1 for any unknown dimension to provide an estimate per unit.

        const cleanDims = dims.map(d => {
            if (typeof d === 'number') return d;
            if (d && typeof d === 'object' && d.toNumber) return d.toNumber(); // Long.js
            return 1; // Default to 1 for dynamic/unknown
        });

        return cleanDims;
    }
};

export const Calculator = macs.Calculator;
