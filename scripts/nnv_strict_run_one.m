function nnv_strict_run_one(category, onnx, vnnlib, outputfile)
% NNV STRICT no-helper per-instance runner.
%
% Designed to be invoked once per VNN-COMP 2025 instance via:
%   timeout --kill-after=10 ${TO}s matlab -nodisplay -nosplash -batch \
%     "addpath('/data1/Kane/ACT/scripts'); nnv_strict_run_one('CAT','ONNX','VNN','OUT');"
%
% Pre-conditions:
%   - NNV source patched with /data1/Kane/ACT/scripts/nnv_patches/run_vnncomp_instance.m.patch
%     (adds env-gated NNV_STRICT_NO_HELPER mode).
%   - env NNV_STRICT_NO_HELPER is set to '1' BEFORE this function runs.
%     We set it defensively here too, in case the caller forgot.
%
% Behavior:
%   - Adds NNV codebase to path.
%   - Calls run_vnncomp_instance(category, onnx, vnnlib, outputfile) which writes
%     one of {sat, unsat, unknown, unsupported_strict} to outputfile.
%   - On any exception, writes 'error' to outputfile so downstream sweep logic
%     can classify the verdict instead of seeing missing_result.
%
% External wrapper enforces wall-clock timeout via the 'timeout' command.

    setenv('NNV_STRICT_NO_HELPER', '1');

    nnv_root = '/data1/Kane/nnv/code/nnv';
    submission_dir = fullfile(nnv_root, 'examples/Submission/VNN_COMP2025');
    addpath(submission_dir);
    addpath(genpath(nnv_root));
    cd(submission_dir);

    fprintf('=== NNV STRICT instance start  category=%s\n', category);
    fprintf('=== onnx=%s\n', onnx);
    fprintf('=== vnnlib=%s\n', vnnlib);
    fprintf('=== outputfile=%s\n', outputfile);
    fprintf('=== NNV_STRICT_NO_HELPER=%s\n', getenv('NNV_STRICT_NO_HELPER'));

    t0 = tic;
    try
        [status, ~] = run_vnncomp_instance(category, onnx, vnnlib, outputfile);
        fprintf('=== NNV STRICT status=%d wall=%.3fs\n', status, toc(t0));
    catch err
        fprintf(2, '=== NNV STRICT error: %s\n%s\n', err.message, err.getReport);
        % Best-effort: write 'error' so the sweep classifies this honestly.
        try
            fid = fopen(outputfile, 'w');
            if fid > 0
                fprintf(fid, 'error\n');
                fclose(fid);
            end
        catch
        end
    end
end
