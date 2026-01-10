
const macs = {};

macs.Calculator = class {

    constructor() {
    }

    calculate(node) {
        if (!node || !node.type || !node.type.name) {
            return null;
        }

        const type = node.type.name;
        const name = type.split('::').pop().toLowerCase();

        if (name === 'conv2d' || name === 'conv' || name === 'convolution' || name === 'conv1d' || name === 'conv3d') {
            return this._computeConv(node);
        } else if (name === 'linear' || name === 'gemm' || name === 'matmul') {
            return this._computeLinear(node);
        }

        return null;
    }

    _computeConv(node) {
        const inputs = node.inputs;
        if (!inputs || inputs.length < 2) {
            return null;
        }

        let weight = null;
        const weightInput = inputs.find(i => i.name && (i.name.toLowerCase().endsWith('weight') || i.name === 'W'));
        if (weightInput) {
            weight = this._getTensorShape(weightInput);
        }

        if (!weight) {
             weight = this._getTensorShape(inputs[1]);
        }

        const outputs = node.outputs;
        if (!outputs || outputs.length < 1) {
            return null;
        }

        let output = this._getTensorShape(outputs[0]);

        if (!output) {
            const input = this._getTensorShape(inputs[0]);

            if (input && weight) {
                // Assuming NCHW
                // input: [N, Cin, Hin, Win]
                // weight: [Cout, Cin, Kh, Kw]
                if (input.length === 4 && weight.length === 4) {
                    const N = input[0];
                    const Cout = weight[0];
                    const Hin = input[2];
                    const Win = input[3];
                    const Kh = weight[2];
                    const Kw = weight[3];

                    const getAttr = (name, def) => {
                        const attr = (node.attributes || []).find(a => a.name === name);
                        if (attr) {
                            if (Array.isArray(attr.value)) return attr.value;
                            return [attr.value];
                        }
                        return def;
                    };

                    const strides = getAttr('stride', [1, 1]);
                    const pads = getAttr('padding', [0, 0]); // [pad_h, pad_w]
                    const dilations = getAttr('dilation', [1, 1]);

                    let padH = 0;
                    let padW = 0;
                    if (pads.length === 2) {
                        padH = pads[0] * 2;
                        padW = pads[1] * 2;
                    } else if (pads.length === 4) {
                        padH = pads[0] + pads[2];
                        padW = pads[1] + pads[3];
                    }

                    const strideH = strides[0];
                    const strideW = strides.length > 1 ? strides[1] : strides[0];
                    const dilationH = dilations[0];
                    const dilationW = dilations.length > 1 ? dilations[1] : dilations[0];

                    const Hout = Math.floor((Hin + padH - dilationH * (Kh - 1) - 1) / strideH + 1);
                    const Wout = Math.floor((Win + padW - dilationW * (Kw - 1) - 1) / strideW + 1);

                    output = [N, Cout, Hout, Wout];
                }
            }
        }

        if (!weight || !output) {
            return null;
        }

        if (output.length < 3 || weight.length < 3) {
            return null;
        }

        const outputElements = output.reduce((a, b) => a * b, 1);
        const weightElements = weight.reduce((a, b) => a * b, 1);
        const outChannels = weight[0];

        if (outChannels === 0) return null;

        const kernelSize = weightElements / outChannels;
        const total = outputElements * kernelSize;

        return total;
    }

    _computeLinear(node) {
        const inputs = node.inputs;
        const outputs = node.outputs;

        if (!inputs || inputs.length < 2 || !outputs || outputs.length < 1) {
            return null;
        }

        let input1 = null;
        const weightInput = inputs.find(i => i.name && (i.name.toLowerCase().endsWith('weight') || i.name === 'B'));
        if (weightInput) {
            input1 = this._getTensorShape(weightInput);
        }
        if (!input1) {
            input1 = this._getTensorShape(inputs[1]);
        }

        let output = this._getTensorShape(outputs[0]);

        if (!output && input1) {
             const input0 = this._getTensorShape(inputs[0]);
             if (input0) {
                 const Out = input1[0];
                 const batch = input0.slice(0, -1);
                 output = [...batch, Out];
             }
        }

        if (!output) return null;

        if (input1) {
            const name = node.type.name.split('::').pop().toLowerCase();

            if (name === 'linear') {
                 if (input1.length >= 2) {
                     const inFeatures = input1[1];
                     const outputElements = output.reduce((a, b) => a * b, 1);
                     return outputElements * inFeatures;
                 }
            }

            if (name === 'matmul' || name === 'gemm') {
                const input0 = this._getTensorShape(inputs[0]);
                if (input0 && input0.length > 0) {
                    const K = input0[input0.length - 1];
                    const outputElements = output.reduce((a, b) => a * b, 1);
                    return outputElements * K;
                }
            }
        }

        return null;
    }

    _getTensorShape(parameter) {
        if (!parameter || !parameter.value) return null;

        const values = parameter.value;
        if (!Array.isArray(values) || values.length === 0) return null;

        const value = values[0];
        if (!value) return null;

        let type = value.type;
        if (!type && value.initializer && value.initializer.type) {
            type = value.initializer.type;
        }

        if (!type) {
            return null;
        }

        if (!type.shape || !type.shape.dimensions) {
            return null;
        }

        const dims = type.shape.dimensions;

        const cleanDims = dims.map(d => {
            if (typeof d === 'number') return d;
            if (d && typeof d === 'object' && d.toNumber) return d.toNumber();
            return 1;
        });

        return cleanDims;
    }
};

export const Calculator = macs.Calculator;
