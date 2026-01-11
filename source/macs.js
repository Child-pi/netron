
const macs = {};

macs.Calculator = class {

    static calculate(node) {
        const type = node.type ? node.type.name : null;
        if (!type) {
            return null;
        }

        if (type === 'Conv' || type === 'Conv2d' || type === 'Convolution') {
            return macs.Calculator._conv(node);
        }
        if (type === 'Gemm' || type === 'MatMul' || type === 'Linear') {
            return macs.Calculator._matmul(node);
        }

        return null;
    }

    static _conv(node) {
        if (node.inputs.length < 2 || node.outputs.length < 1) {
            return null;
        }

        const input = macs.Calculator._tensor(node.inputs[0]);
        const weight = macs.Calculator._tensor(node.inputs[1]);
        const output = macs.Calculator._tensor(node.outputs[0]);

        if (!input || !output || !weight) {
            return null;
        }

        /*
        let groups = 1;
        if (node.attributes) {
            for (const attr of node.attributes) {
                if (attr.name === 'group' || attr.name === 'groups') {
                    groups = attr.value;
                    break;
                }
            }
        }
        */

        const outputSize = macs.Calculator._numel(output.shape);
        if (outputSize === 0) {
            return null;
        }

        const weightShape = weight.shape;
        if (!weightShape || weightShape.length < 2) {
            return null;
        }
        const weightNumel = macs.Calculator._numel(weightShape);

        // output.shape usually [N, Cout, H, W] or [N, H, W, Cout]
        // weight.shape usually [Cout, Cin/g, kH, kW] or [kH, kW, Cin, Cout]

        // Try to identify Cout to calculate kernel size per output channel.
        let cout = 0;

        // Heuristic 1: If weight has 4 dims, and output has 4 dims.
        if (output.shape.length === 4 && weightShape.length === 4) {
            // Check NCHW vs NHWC
            // NCHW: Cout is shape[1]
            // NHWC: Cout is shape[3]
            // Weight NCHW: Cout is shape[0]
            // Weight NHWC: Cout is shape[3] or shape[0]?
            // TF: [H, W, In, Out]

            // Hypothesis: if output dimension matches a weight dimension, that's likely Cout.
            const candidates = [];
            if (output.shape[1] === weightShape[0]) {
                candidates.push({ dim: 1, val: output.shape[1] }); // NCHW
            }
            if (output.shape[3] === weightShape[3]) {
                candidates.push({ dim: 3, val: output.shape[3] }); // NHWC (TF) or TFLite
            }
            if (output.shape[3] === weightShape[0]) {
                candidates.push({ dim: 3, val: output.shape[3] }); // NHWC (output) but NCHW (weight)? Rare.
            }

            if (candidates.length > 0) {
                cout = candidates[0].val;
            }
        }

        // If we found Cout, we can calculate MACs.
        // MACs = OutputElementCount * (WeightElementCount / Cout)
        if (cout > 0) {
            return outputSize * (weightNumel / cout);
        }

        return null;
    }

    static _matmul(node) {
        const input = macs.Calculator._tensor(node.inputs[0]);
        const weight = macs.Calculator._tensor(node.inputs[1]);
        const output = macs.Calculator._tensor(node.outputs[0]);

        if (!output) {
            return null;
        }

        const outputSize = macs.Calculator._numel(output.shape);

        // Standard Gemm/Linear: Output * InnerDim
        // InnerDim is the common dimension between input and weight.
        // Input: [..., M, K]
        // Weight: [K, N] (if not transposed) or [N, K] (if transposed)
        // Output: [..., M, N]

        // If we have weight shape, we can guess K.
        // If weight is 2D: [A, B]
        // If output last dim is B, then K is A.
        // If output last dim is A, then K is B.

        if (weight && weight.shape && weight.shape.length === 2 && output.shape && output.shape.length >= 2) {
            const outLast = output.shape[output.shape.length - 1];
            if (weight.shape[0] === outLast) {
                const k = weight.shape[1];
                return outputSize * k;
            } else if (weight.shape[1] === outLast) {
                const k = weight.shape[0];
                return outputSize * k;
            }
        }

        // If input shape is known, use last dim of input as K (assuming standard matmul).
        if (input && input.shape && input.shape.length >= 1) {
            const k = input.shape[input.shape.length - 1];
            return outputSize * k;
        }

        return null;
    }

    static _tensor(argument) {
        if (!argument || !argument.value) {
            return null;
        }
        // In Netron, argument.value is array of Value objects.
        // We usually care about the first one.
        const values = argument.value;
        if (!values || values.length === 0) {
            return null;
        }
        const value = values[0];
        if (!value || !value.type) {
            return null;
        }

        // value.type is usually TensorType which has shape
        const type = value.type;
        if (!type.shape || !type.shape.dimensions) {
            return null;
        }

        return {
            shape: type.shape.dimensions.map((d) => (d && d.toNumber) ? d.toNumber() : d) // Handle Long/BigInt if any
        };
    }

    static _numel(dims) {
        if (!dims) {
            return 0;
        }
        let p = 1;
        for (const d of dims) {
            if (typeof d !== 'number') {
                return 0; // Unknown dimension '?'
            }
            p *= d;
        }
        return p;
    }

    static format(value) {
        if (value === null || value === undefined) {
            return '';
        }
        if (value < 1000) {
            return value.toString();
        }
        if (value < 1000000) {
            return `${(value / 1000).toFixed(2)}K`;
        }
        if (value < 1000000000) {
            return `${(value / 1000000).toFixed(2)}M`;
        }
        return `${(value / 1000000000).toFixed(2)}G`;
    }
};

export const Calculator = macs.Calculator;
