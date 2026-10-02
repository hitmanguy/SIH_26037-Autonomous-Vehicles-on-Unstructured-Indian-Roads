classdef SliceLayer1011 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

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
        % Specify the properties of the class that will not be modified
        % after the first assignment.
        function p = matlabCodegenNontunableProperties(~)
            p = {
                % Constants, i.e., Vars, NumDims and all learnables and states
                'Vars'
                'NumDims'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = best_matlab.coder.SliceLayer1011(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = best_matlab.SliceLayer1011(cgInstance.Name);
            if isstruct(cgInstance.Vars)
                names = fieldnames(cgInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this_ml.Vars.(fieldname) = dlarray(cgInstance.Vars.(fieldname));
                end
            else
                this_ml.Vars = [];
            end
            this_ml.NumDims = cgInstance.NumDims;
        end
    end

    methods
        function this = SliceLayer1011(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_15_Slice_out'};
            if isstruct(mlInstance.Vars)
                names = fieldnames(mlInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this.Vars.(fieldname) = best_matlab.coder.ops.extractIfDlarray(mlInstance.Vars.(fieldname));
                end
            else
                this.Vars = [];
            end

            this.NumDims = mlInstance.NumDims;
        end

        function [x_model_15_Slice_out] = predict(this, x_model_15_cv1_act_M__)
            if isdlarray(x_model_15_cv1_act_M__)
                x_model_15_cv1_act_M_ = stripdims(x_model_15_cv1_act_M__);
            else
                x_model_15_cv1_act_M_ = x_model_15_cv1_act_M__;
            end
            x_model_15_cv1_act_MNumDims = 4;
            x_model_15_cv1_act_M = best_matlab.coder.ops.permuteInputVar(x_model_15_cv1_act_M_, [4 3 1 2], 4);

            [x_model_15_Slice_out__, x_model_15_Slice_outNumDims__] = SliceGraph1022(this, x_model_15_cv1_act_M, x_model_15_cv1_act_MNumDims, false);
            x_model_15_Slice_out_ = best_matlab.coder.ops.permuteOutputVar(x_model_15_Slice_out__, [3 4 2 1], 4);

            x_model_15_Slice_out = dlarray(single(x_model_15_Slice_out_), 'SSCB');
        end

        function [x_model_15_Slice_out, x_model_15_Slice_outNumDims1023] = SliceGraph1022(this, x_model_15_cv1_act_M, x_model_15_cv1_act_MNumDims, Training)

            % Execute the operators:
            % Slice:
            [indices1011, x_model_15_Slice_outNumDims] = best_matlab.coder.ops.prepareSliceArgs(x_model_15_cv1_act_M, this.Vars.x_model_2_Constant_1, this.Vars.x_model_2_Mul_1_outp, this.Vars.x_model_2_Constant_o, '', coder.const(x_model_15_cv1_act_MNumDims));
            x_model_15_Slice_out = x_model_15_cv1_act_M(indices1011{:});

            % Set graph output arguments
            x_model_15_Slice_outNumDims1023 = coder.const(x_model_15_Slice_outNumDims);

        end

    end

end