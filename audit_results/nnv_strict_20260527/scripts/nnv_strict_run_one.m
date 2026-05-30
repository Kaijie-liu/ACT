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
    % Cap NNV's parpool size for exact-star reach. With N lanes, each lane
    % requests this many workers; total = N * NNV_NUMCORES. Keep 3*5 < 20 cores.
    if isempty(getenv('NNV_NUMCORES'))
        setenv('NNV_NUMCORES', '5');
    end

    % Clean any stale parpool state from a previous interrupted MATLAB session.
    % NNV's exact-star calls parpool('local', N) which fails on stale Jobs in
    % ~/.matlab/local_cluster_jobs/R2026a/ with "Failed to locate and destroy
    % old interactive jobs" / "Unable to use a value of type cell as an index".
    % Three-layer defence:
    %   1) Close any pool in this process
    %   2) Delete all known jobs in the local cluster
    %   3) Nuke the on-disk job storage so even corrupt entries are gone
    try
        delete(gcp('nocreate'));
    catch
    end
    try
        c = parcluster('Processes');
        if ~isempty(c.Jobs)
            delete(c.Jobs);
        end
    catch
    end
    try
        prefdir_local = fullfile(prefdir, '..', 'local_cluster_jobs', version('-release'));
        if isfolder(prefdir_local)
            d = dir(fullfile(prefdir_local, 'Job*'));
            for k = 1:length(d)
                try, rmdir(fullfile(prefdir_local, d(k).name), 's'); catch, end
                try, delete(fullfile(prefdir_local, [d(k).name '*'])); catch, end
            end
        end
    catch
    end
    % CRITICAL: with 3 lanes running concurrently, the shared default
    % JobStorageLocation produces "Failed to locate and destroy old interactive
    % jobs" race. Give THIS MATLAB process its own private job storage so the
    % parcluster never shares state with sibling processes.
    try
        c = parcluster('Processes');
        priv_dir = fullfile(tempdir, sprintf('nnv_strict_jobs_%d_%d', feature('getpid'), round(rand()*1e9)));
        if ~isfolder(priv_dir), mkdir(priv_dir); end
        c.JobStorageLocation = priv_dir;
        % DO NOT cap c.NumWorkers — NNV hardcodes numCores=feature('numcores')
        % and calls parpool('local', N), which errors if N > NumWorkers.
        saveProfile(c);
    catch
    end

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
