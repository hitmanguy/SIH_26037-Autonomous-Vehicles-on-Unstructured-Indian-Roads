classdef SliceLayer1004 < nnet.layer.Layer & nnet.layer.Formattable
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
            name = 'best_matlab.coder.SliceLayer1004';
        end
    end


    methods
        function this = SliceLayer1004(name)
            this.Name = name;
            this.OutputNames = {'x_model_6_Slice_1_ou'};
        end

        function [x_model_6_Slice_1_ou] = predict(this, x_model_6_cv1_act_Mu)
            if isdlarray(x_model_6_cv1_act_Mu)
                x_model_6_cv1_act_Mu = stripdims(x_model_6_cv1_act_Mu);
            end
            x_model_6_cv1_act_MuNumDims = 4;
            x_model_6_cv1_act_Mu = best_matlab.ops.permuteInputVar(x_model_6_cv1_act_Mu, [4 3 1 2], 4);

            [x_model_6_Slice_1_ou, x_model_6_Slice_1_ouNumDims] = SliceGraph1008(this, x_model_6_cv1_act_Mu, x_model_6_cv1_act_MuNumDims, false);
            x_model_6_Slice_1_ou = best_matlab.ops.permuteOutputVar(x_model_6_Slice_1_ou, [3 4 2 1], 4);

            x_model_6_Slice_1_ou = dlarray(single(x_model_6_Slice_1_ou), 'SSCB');
        end

        function [x_model_6_Slice_1_ou] = forward(this, x_model_6_cv1_act_Mu)
            if isdlarray(x_model_6_cv1_act_Mu)
                x_model_6_cv1_act_Mu = stripdims(x_model_6_cv1_act_Mu);
            end
            x_model_6_cv1_act_MuNumDims = 4;
            x_model_6_cv1_act_Mu = best_matlab.ops.permuteInputVar(x_model_6_cv1_act_Mu, [4 3 1 2], 4);

            [x_model_6_Slice_1_ou, x_model_6_Slice_1_ouNumDims] = SliceGraph1008(this, x_model_6_cv1_act_Mu, x_model_6_cv1_act_MuNumDims, true);
            x_model_6_Slice_1_ou = best_matlab.ops.permuteOutputVar(x_model_6_Slice_1_ou, [3 4 2 1], 4);

            x_model_6_Slice_1_ou = dlarray(single(x_model_6_Slice_1_ou), 'SSCB');
        end

        function [x_model_6_Slice_1_ou, x_model_6_Slice_1_ouNumDims1009] = SliceGraph1008(this, x_model_6_cv1_act_Mu, x_model_6_cv1_act_MuNumDims, Training)

            % Execute the operators:
            % Slice:
            [Indices, x_model_6_Slice_1_ouNumDims] = best_matlab.ops.prepareSliceArgs(x_model_6_cv1_act_Mu, this.Vars.x_model_4_Mul_1_outp, this.Vars.x_model_6_Mul_1_outp, this.Vars.x_model_2_Constant_o, '', x_model_6_cv1_act_MuNumDims);
            x_model_6_Slice_1_ou = x_model_6_cv1_act_Mu(Indices{:});

            % Set graph output arguments
            x_model_6_Slice_1_ouNumDims1009 = x_model_6_Slice_1_ouNumDims;

        end

    end

end