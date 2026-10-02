classdef SliceLayer1003 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'best_matlab.coder.SliceLayer1003';
        end
    end


    methods
        function this = SliceLayer1003(name)
            this.Name = name;
            this.OutputNames = {'x_model_4_Slice_outp'};
        end

        function [x_model_4_Slice_outp] = predict(this, x_model_4_cv1_act_Mu)
            if isdlarray(x_model_4_cv1_act_Mu)
                x_model_4_cv1_act_Mu = stripdims(x_model_4_cv1_act_Mu);
            end
            x_model_4_cv1_act_MuNumDims = 4;
            x_model_4_cv1_act_Mu = best_matlab.ops.permuteInputVar(x_model_4_cv1_act_Mu, [4 3 1 2], 4);

            [x_model_4_Slice_outp, x_model_4_Slice_outpNumDims] = SliceGraph1006(this, x_model_4_cv1_act_Mu, x_model_4_cv1_act_MuNumDims, false);
            x_model_4_Slice_outp = best_matlab.ops.permuteOutputVar(x_model_4_Slice_outp, [3 4 2 1], 4);

            x_model_4_Slice_outp = dlarray(single(x_model_4_Slice_outp), 'SSCB');
        end

        function [x_model_4_Slice_outp] = forward(this, x_model_4_cv1_act_Mu)
            if isdlarray(x_model_4_cv1_act_Mu)
                x_model_4_cv1_act_Mu = stripdims(x_model_4_cv1_act_Mu);
            end
            x_model_4_cv1_act_MuNumDims = 4;
            x_model_4_cv1_act_Mu = best_matlab.ops.permuteInputVar(x_model_4_cv1_act_Mu, [4 3 1 2], 4);

            [x_model_4_Slice_outp, x_model_4_Slice_outpNumDims] = SliceGraph1006(this, x_model_4_cv1_act_Mu, x_model_4_cv1_act_MuNumDims, true);
            x_model_4_Slice_outp = best_matlab.ops.permuteOutputVar(x_model_4_Slice_outp, [3 4 2 1], 4);

            x_model_4_Slice_outp = dlarray(single(x_model_4_Slice_outp), 'SSCB');
        end

        function [x_model_4_Slice_outp, x_model_4_Slice_outpNumDims1007] = SliceGraph1006(this, x_model_4_cv1_act_Mu, x_model_4_cv1_act_MuNumDims, Training)

            % Execute the operators:
            % Slice:
            [Indices, x_model_4_Slice_outpNumDims] = best_matlab.ops.prepareSliceArgs(x_model_4_cv1_act_Mu, this.Vars.x_model_2_Constant_1, this.Vars.x_model_2_Mul_1_outp, this.Vars.x_model_2_Constant_o, '', x_model_4_cv1_act_MuNumDims);
            x_model_4_Slice_outp = x_model_4_cv1_act_Mu(Indices{:});

            % Set graph output arguments
            x_model_4_Slice_outpNumDims1007 = x_model_4_Slice_outpNumDims;

        end

    end

end