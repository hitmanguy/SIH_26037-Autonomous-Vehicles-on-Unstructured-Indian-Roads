classdef SliceLayer1004 < nnet.layer.Layer & nnet.layer.Formattable
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
            this_cg = best_matlab.coder.SliceLayer1004(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = best_matlab.SliceLayer1004(cgInstance.Name);
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
        function this = SliceLayer1004(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'x_model_6_Slice_1_ou'};
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

        function [x_model_6_Slice_1_ou] = predict(this, x_model_6_cv1_act_Mu__)
            if isdlarray(x_model_6_cv1_act_Mu__)
                x_model_6_cv1_act_Mu_ = stripdims(x_model_6_cv1_act_Mu__);
            else
                x_model_6_cv1_act_Mu_ = x_model_6_cv1_act_Mu__;
            end
            x_model_6_cv1_act_MuNumDims = 4;
            x_model_6_cv1_act_Mu = best_matlab.coder.ops.permuteInputVar(x_model_6_cv1_act_Mu_, [4 3 1 2], 4);

            [x_model_6_Slice_1_ou__, x_model_6_Slice_1_ouNumDims__] = SliceGraph1008(this, x_model_6_cv1_act_Mu, x_model_6_cv1_act_MuNumDims, false);
            x_model_6_Slice_1_ou_ = best_matlab.coder.ops.permuteOutputVar(x_model_6_Slice_1_ou__, [3 4 2 1], 4);

            x_model_6_Slice_1_ou = dlarray(single(x_model_6_Slice_1_ou_), 'SSCB');
        end

        function [x_model_6_Slice_1_ou, x_model_6_Slice_1_ouNumDims1009] = SliceGraph1008(this, x_model_6_cv1_act_Mu, x_model_6_cv1_act_MuNumDims, Training)

            % Execute the operators:
            % Slice:
            [indices1004, x_model_6_Slice_1_ouNumDims] = best_matlab.coder.ops.prepareSliceArgs(x_model_6_cv1_act_Mu, this.Vars.x_model_4_Mul_1_outp, this.Vars.x_model_6_Mul_1_outp, this.Vars.x_model_2_Constant_o, '', coder.const(x_model_6_cv1_act_MuNumDims));
            x_model_6_Slice_1_ou = x_model_6_cv1_act_Mu(indices1004{:});

            % Set graph output arguments
            x_model_6_Slice_1_ouNumDims1009 = coder.const(x_model_6_Slice_1_ouNumDims);

        end

    end

end