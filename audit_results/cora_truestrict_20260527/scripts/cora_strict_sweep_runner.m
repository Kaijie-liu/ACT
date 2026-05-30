function cora_strict_sweep_runner(bench_root, only_bench, results_root, timeout_cap, gpu_free_gb_min, gpu_check_interval_s)
% STRICT "pure verifier" CORA sweep on VNN-COMP 2025 benchmarks.
% Pre-conditions enforced:
%   - prepare_instance.m has been patched to use 'center' + 'naive'
%     (no falsification shortcut, no random sampling, no gradient).
%
% Parallel-safety: before each benchmark, waits until GPU has
%   gpu_free_gb_min GB free (default 40). Polls every gpu_check_interval_s
%   seconds (default 30). Inside a benchmark, instances run sequentially.
%
% Args (all optional):
%   bench_root             default /data1/Kane/data/vnncomp2025_benchmarks/benchmarks
%   only_bench             ''  -> use built-in light-to-heavy order
%   results_root           default /data1/Kane/ACT/audit_results/cora_strict_20260526
%   timeout_cap            0   -> use instances.csv timeout per row
%   gpu_free_gb_min        40  -> wait until at least this much GPU is free
%   gpu_check_interval_s   30
%
% Light->heavy order is chosen so the heavy CORA benchmarks (cifar100,
% tinyimagenet, vit, safenlp) run LAST, after the ACT stream-3 heavy CNN
% load has had time to drain.

    if nargin < 1 || isempty(bench_root)
        bench_root = '/data1/Kane/data/vnncomp2025_benchmarks/benchmarks';
    end
    if nargin < 2, only_bench = ''; end
    if nargin < 3 || isempty(results_root)
        results_root = '/data1/Kane/ACT/audit_results/cora_strict_20260526';
    end
    if nargin < 4 || isempty(timeout_cap), timeout_cap = 0; end
    if nargin < 5 || isempty(gpu_free_gb_min), gpu_free_gb_min = 40; end
    if nargin < 6 || isempty(gpu_check_interval_s), gpu_check_interval_s = 30; end

    addpath(genpath('/data1/Kane/cora-vnncomp2025'));
    cd('/data1/Kane/cora-vnncomp2025');

    % LIGHT -> HEAVY order. The heaviest CORA benchmarks at the END.
    % (light = small MLPs / few instances; heavy = ResNet/Transformer / large input)
    bench_order = {
        'test', ...                                                 % 5    smoke
        'cersyve', ...                                              % 12   small
        'lsnc_relu', ...                                            % 80   small
        'soundnessbench', ...                                       % 50   small
        'sat_relu', ...                                             % 100  small
        'cgan_2023', ...                                            % 21   small GAN
        'tllverifybench_2023', ...                                  % 32   small MLP
        'malbeware', ...                                            % 150  small
        'traffic_signs_recognition_2023', ...                       % 45   small QConv
        'collins_rul_cnn_2022', ...                                 % 62   small
        'linearizenn_2024', ...                                     % 60   small
        'dist_shift_2023', ...                                      % 72   small
        'ml4acopf_2024', ...                                        % 69   medium
        'metaroom_2023', ...                                        % 100  medium
        'cctsdb_yolo_2023', ...                                     % 39   medium
        'yolo_2023', ...                                            % 72   medium
        'relusplitter', ...                                         % 220  medium
        'cora_2024', ...                                            % 180  medium
        'acasxu_2023', ...                                          % 186  medium
        'nn4sys', ...                                               % 194  medium
        'safenlp_2024', ...                                         % 1080 (large count, small per-inst)
        'vit_2023', ...                                             % 200  HEAVY
        'tinyimagenet_2024', ...                                    % 200  HEAVY
        'cifar100_2024', ...                                        % 200  HEAVY
        'vggnet16_2022', ...                                        % 18   HEAVY (VGG)
        'collins_aerospace_benchmark', ...                          % 6    HEAVY (YOLO)
    };

    if ~isempty(only_bench)
        bench_order = strsplit(strtrim(only_bench));
    end

    if ~isfolder(results_root), mkdir(results_root); end
    driver_log = fullfile(results_root, '_run.log');
    meta_path  = fullfile(results_root, '_run.meta.json');

    log_fid = fopen(driver_log, 'a');
    L = @(s) lprint(log_fid, s);

    fid = fopen(meta_path, 'w');
    fprintf(fid, ['{\n' ...
        '  "tool": "CORA (vnncomp2025, TRUESTRICT pure verifier)",\n' ...
        '  "tool_dir": "/data1/Kane/cora-vnncomp2025",\n' ...
        '  "config": {\n' ...
        '    "falsification_method": "none",\n' ...
        '    "refinement_method": "naive",\n' ...
        '    "note": "TRUESTRICT: PATCHED verify.m + validateNNoptions.m to skip ALL falsification (no center-of-box, no FGSM, no zonotack/random sampling). CORA proves only V via over-approximative reachability; A is IMPOSSIBLE under this config (over-approximation cannot witness counter-examples)."\n' ...
        '  },\n' ...
        '  "matlab_version": "%s",\n' ...
        '  "started_at": "%s",\n' ...
        '  "host": "%s",\n' ...
        '  "bench_root": "%s",\n' ...
        '  "results_root": "%s",\n' ...
        '  "flags": {\n' ...
        '    "GPU_FREE_GB_MIN": %g,\n' ...
        '    "GPU_CHECK_INTERVAL_S": %g,\n' ...
        '    "TIMEOUT_CAP_SEC": %g\n' ...
        '  }\n' ...
        '}\n'], version, datestr(now,'yyyy-mm-ddTHH:MM:SS'), getenv('HOSTNAME'), bench_root, results_root, gpu_free_gb_min, gpu_check_interval_s, timeout_cap);
    fclose(fid);

    L(sprintf('=== CORA TRUESTRICT sweep (none+naive, falsification block PATCHED out); results -> %s', results_root));
    L(sprintf('=== GPU gate: wait until %g GB free per benchmark', gpu_free_gb_min));

    for bi = 1:numel(bench_order)
        bench  = bench_order{bi};
        bdir   = fullfile(bench_root, bench);
        csv    = fullfile(bdir, 'instances.csv');
        outdir = fullfile(results_root, bench);
        sumcsv = fullfile(outdir, '_summary.csv');
        if ~isfile(csv)
            L(sprintf('[%s] SKIP — instances.csv not found', bench));
            continue
        end
        if ~isfolder(outdir), mkdir(outdir); end
        if ~isfile(sumcsv)
            fid = fopen(sumcsv, 'w');
            fprintf(fid, 'idx,onnx,vnnlib,csv_timeout,used_timeout,wall_sec,verdict,result_file,log_file\n');
            fclose(fid);
        end

        % ---- GPU gate: wait for free memory ----
        gpu_wait_for_free(gpu_free_gb_min, gpu_check_interval_s, bench, L);

        T = readtable(csv, 'Delimiter', ',', 'ReadVariableNames', false, ...
                            'TextType', 'string', 'NumHeaderLines', 0);
        n_total = height(T);
        L(sprintf('[%s] instances=%d (start)', bench, n_total));

        n_done=0; n_skip=0; n_sat=0; n_unsat=0; n_to=0; n_err=0;

        for k = 1:n_total
            onnx_rel    = strtrim(char(T{k,1}));
            vnnlib_rel  = strtrim(char(T{k,2}));
            csv_timeout = str2double(strtrim(string(T{k,3})));
            if isnan(csv_timeout), csv_timeout = 60; end

            onnx_p   = fullfile(bdir, onnx_rel);
            vnnlib_p = fullfile(bdir, vnnlib_rel);
            [~, onnx_tag, ~]   = fileparts(onnx_rel);
            [~, vnn_tag,  ~]   = fileparts(vnnlib_rel);
            stem = sprintf('%04d__%s__%s', k, onnx_tag, vnn_tag);
            res_file = fullfile(outdir, [stem '.result']);
            log_file = fullfile(outdir, [stem '.log']);
            json_file = fullfile(outdir, [stem '.json']);

            if isfile(res_file)
                info = dir(res_file);
                if info.bytes > 0
                    n_skip = n_skip + 1;
                    continue
                end
            end

            used_to = csv_timeout;
            if timeout_cap > 0 && used_to > timeout_cap, used_to = timeout_cap; end

            t0 = tic;
            verdict = 'missing_result';
            % Capture full MATLAB stdout for THIS instance into its .log file
            % (CORA's verify.m prints the iteration table + final result via fprintf
            % to MATLAB stdout — diary() teeing routes it to the file).
            diary(log_file); diary on;
            fprintf('=== CORA STRICT: bench=%s idx=%d  timeout=%g  start=%s\n', ...
                bench, k, used_to, datestr(now, 'yyyy-mm-dd HH:MM:SS'));
            fprintf('=== onnx=%s\n=== vnnlib=%s\n', onnx_rel, vnnlib_rel);
            try
                prepare_instance(bench, onnx_p, vnnlib_p);
                [resStr, ~] = run_instance(bench, onnx_p, vnnlib_p, res_file, used_to, false);
                verdict = char(resStr);
                fprintf('=== CORA STRICT: verdict=%s wall=%.3fs\n', verdict, toc(t0));
            catch err
                fprintf('=== CORA STRICT error on %s: %s\n%s\n', stem, err.message, err.getReport);
                verdict = 'error';
            end
            diary off; diary('');  % reset diary target
            wall = toc(t0);

            switch lower(verdict)
                case {'unsat','verified','holds','safe'},        cat_ver = 'unsat';   n_unsat=n_unsat+1;
                case {'sat','violated','falsified','unsafe'},    cat_ver = 'sat';     n_sat=n_sat+1;
                case {'timeout','timed_out'},                     cat_ver = 'timeout'; n_to=n_to+1;
                case 'unknown',                                   cat_ver = 'unknown'; n_to=n_to+1;
                otherwise,                                        cat_ver = verdict;   n_err=n_err+1;
            end
            n_done = n_done + 1;

            sfid = fopen(sumcsv, 'a');
            fprintf(sfid, '%d,"%s","%s",%g,%g,%.2f,"%s","%s","%s"\n', ...
                k, onnx_rel, vnnlib_rel, csv_timeout, used_to, wall, cat_ver, res_file, log_file);
            fclose(sfid);

            jfid = fopen(json_file, 'w');
            fprintf(jfid, '{"idx":%d,"benchmark":"%s","onnx":"%s","vnnlib":"%s","csv_timeout":%g,"used_timeout":%g,"wall_sec":%.2f,"verdict_raw":"%s","verdict":"%s","attack_path_exists":false,"falsification_method":"center","refinement_method":"naive","tool":"CORA_STRICT"}\n', ...
                k, bench, onnx_rel, vnnlib_rel, csv_timeout, used_to, wall, verdict, cat_ver);
            fclose(jfid);

            if mod(k, 5) == 0 || k == n_total
                L(sprintf('[%s] %d/%d sat=%d unsat=%d timeout=%d err=%d resumed=%d', ...
                    bench, k, n_total, n_sat, n_unsat, n_to, n_err, n_skip));
            end
        end
        L(sprintf('[%s] DONE -- total=%d new=%d resumed=%d sat=%d unsat=%d timeout=%d err=%d', ...
            bench, n_total, n_done, n_skip, n_sat, n_unsat, n_to, n_err));
    end

    L('=== STRICT sweep complete');
    fclose(log_fid);
end

function gpu_wait_for_free(gb_min, interval_s, bench, L)
    bytes_needed = gb_min * 1024^3;
    first_wait = true;
    while true
        try
            [~, out] = system('nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits');
            free_mib = str2double(strtrim(out));
            free_gb  = free_mib / 1024;
        catch
            free_gb = 0;
        end
        if free_gb >= gb_min || ~isfinite(free_gb)
            if ~first_wait
                L(sprintf('[GPU gate] free=%.1f GB >= %.1f GB; starting [%s]', free_gb, gb_min, bench));
            end
            return
        end
        if first_wait
            L(sprintf('[GPU gate] free=%.1f GB < %.1f GB; waiting before [%s] (poll %ds)', free_gb, gb_min, bench, interval_s));
            first_wait = false;
        end
        pause(interval_s);
    end
end

function lprint(fid, s)
    ts = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    line = sprintf('[%s] %s\n', ts, s);
    fprintf('%s', line);
    fprintf(fid, '%s', line);
end
