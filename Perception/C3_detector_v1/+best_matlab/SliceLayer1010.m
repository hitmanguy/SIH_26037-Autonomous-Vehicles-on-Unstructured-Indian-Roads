classdef SliceLayer1010 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
    end

    properties (State)
    end

    properties
        Vars
        NumDims
    end


    methods(Static, Hidden)
        % Specify the path to the class that will be used for codegen
        function name = matlabCodegenRedirect(~)
            name = 'best_matlab.coder.SliceLayer1010';
        end
    end


    methods
        function this = SliceLayer1010(name)
            this.Name = name;
            this.OutputNames = {'x_model_15_Slice_1_o'};
        end

        function [x_model_15_Slice_1_o] = predict(this, x_model_15_cv1_act_M)
            if isdlarray(x_model_15_cv1_act_M)
                x_model_15_cv1_act_M = stripdims(x_model_15_cv1_act_M);
            end
            x_model_15_cv1_act_MNumDims = 4;
            x_model_15_cv1_act_M = best_matlab.ops.permuteInputVar(x_model_15_cv1_act_M, [4 3 1 2], 4);

            [x_model_15_Slice_1_o, x_model_15_Slice_1_oNumDims] = SliceGraph1020(this, x_model_15_cv1_act_M, x_model_15_cv1_act_MNumDims, false);
            x_model_15_Slice_1_o = best_matlab.ops.permuteOutputVar(x_model_15_Slice_1_o, [3 4 2 1], 4);

            x_model_15_Slice_1_o = dlarray(single(x_model_15_Slice_1_o), 'SSCB');
        end

        function [x_model_15_Slice_1_o] = forward(this, x_model_15_cv1_act_M)
            if isdlarray(x_model_15_cv1_act_M)
                x_model_15_cv1_act_M = stripdims(x_model_15_cv1_act_M);
            end
            x_model_15_cv1_act_MNumDims = 4;
            x_model_15_cv1_act_M = best_matlab.ops.permuteInputVar(x_model_15_cv1_act_M, [4 3 1 2], 4);

            [x_model_15_Slice_1_o, x_model_15_Slice_1_oNumDims] = SliceGraph1020(this, x_model_15_cv1_act_M, x_model_15_cv1_act_MNumDims, true);
            x_model_15_Slice_1_o = best_matlab.ops.permuteOutputVar(x_model_15_Slice_1_o, [3 4 2 1], 4);

            x_model_15_Slice_1_o = dlarray(single(x_model_15_Slice_1_o), 'SSCB');
        end

        function [x_model_15_Slice_1_o, x_model_15_Slice_1_oNumDims1021] = SliceGraph1020(this, x_model_15_cv1_act_M, x_model_15_cv1_act_MNumDims, Training)

            % Execute the operators:
            % Slice:
            [Indices, x_model_15_Slice_1_oNumDims] = best_matlab.ops.prepareSliceArgs(x_model_15_cv1_act_M, this.Vars.x_model_2_Mul_1_outp, this.Vars.x_model_4_Mul_1_outp, this.Vars.x_model_2_Constant_o, '', x_model_15_cv1_act_MNumDims);
            x_model_15_Slice_1_o = x_model_15_cv1_act_M(Indices{:});

            % Set graph output arguments
            x_model_15_Slice_1_oNumDims1021 = x_model_15_Slice_1_oNumDims;

        end

    end

end