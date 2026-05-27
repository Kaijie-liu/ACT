function cora_sweep_runner(bench_root, only_bench, results_root, timeout_cap, kill_grace)
% In-MATLAB sweep runner for CORA on VNN-COMP 2025 benchmarks.
% Runs prepare_instance + run_instance for every instance, in a single
% MATLAB session (avoids per-instance startup cost).
%
% Defaults:
%   bench_root   = /data1/Kane/data/vnncomp2025_benchmarks/benchmarks
%   only_bench   = ''  (empty -> all in BENCH_ORDER)
%   results_root = /data1/Kane/ACT/audit_results/cora_noattack_20260526
%   timeout_cap  = 0    (0 = use whatever instances.csv says)
%   kill_grace   = 90s  (extra time on top of CSV timeout before we abort)
%
% CORA has no attack/PGD/falsification path in its codebase; this run is
% attack-free by construction.

    if nargin < 1 || isempty(bench_root)
        bench_root = '/data1/Kane/data/vnncomp2025_benchmarks/benchmarks';
    end
    if nargin < 2, only_bench = ''; end
    if nargin < 3 || isempty(results_root)
        results_root = '/data1/Kane/ACT/audit_results/cora_noattack_20260526';
    end
    if nargin < 4 || isempty(timeout_cap), timeout_cap = 0; end
    if nargin < 5 || isempty(kill_grace), kill_grace = 90; end %#ok<NASGU>

    addpath(genpath('/data1/Kane/cora-vnncomp2025'));
    cd('/data1/Kane/cora-vnncomp2025');

    bench_order = {
        'test', 'cersyve', 'cgan_2023', ...
        'tllverifybench_2023', 'cctsdb_yolo_2023', 'traffic_signs_recognition_2023', ...
        'collins_rul_cnn_2022', 'linearizenn_2024', 'ml4acopf_2024', 'dist_shift_2023', ...
        'yolo_2023', 'lsnc_relu', 'soundnessbench', ...
        'metaroom_2023', 'cora_2024', 'relusplitter', 'sat_relu', 'malbeware', ...
        'acasxu_2023', 'nn4sys', ...
        'vit_2023', 'tinyimagenet_2024', 'cifar100_2024', ...
        'safenlp_2024', ...
        'vggnet16_2022', 'collins_aerospace_benchmark'
    };

    if ~isempty(only_bench)
        bench_order = strsplit(strtrim(only_bench));
    end

    if ~isfolder(results_root), mkdir(results_root); end
    driver_log = fullfile(results_root, '_run.log');
    meta_path  = fullfile(results_root, '_run.meta.json');

    log_fid = fopen(driver_log, 'a');
    L = @(s) lprint(log_fid, s);

    % write meta once
    fid = fopen(meta_path, 'w');
    fprintf(fid, '{\n  "tool": "CORA (vnncomp2025)",\n  "tool_dir": "/data1/Kane/cora-vnncomp2025",\n  "matlab_version": "%s",\n  "started_at": "%s",\n  "host": "%s",\n  "bench_root": "%s",\n  "results_root": "%s",\n  "flags": {\n    "attack_path_exists": false,\n    "note": "CORA has no PGD/attack/falsification path; default reachability-only run is attack-free by construction.",\n    "use_gpu": true,\n    "TIMEOUT_CAP_SEC": %d,\n    "KILL_GRACE_SEC": %d\n  }\n}\n', ...
        version, datetime("now",'Format','uuuu-MM-dd HH:mm:ssZ','TimeZone','local'), getenv('HOSTNAME'), bench_root, results_root, timeout_cap, kill_grace);
    fclose(fid);

    L(sprintf('=== CORA VNN-COMP 2025 sweep (attack-free by construction), results -> %s', results_root));

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

        % read CSV (no header, 3 cols)
        T = readtable(csv, 'Delimiter', ',', 'ReadVariableNames', false, ...
                            'TextType', 'string', 'NumHeaderLines', 0);
        n_total = height(T);
        L(sprintf('[%s] instances=%d', bench, n_total));

        n_done=0; n_skip=0; n_sat=0; n_unsat=0; n_to=0; n_err=0;

        for k = 1:n_total
            onnx_rel    = char(T{k,1});
            vnnlib_rel  = char(T{k,2});
            csv_timeout = str2double(T{k,3});
            if isnan(csv_timeout), csv_timeout = 60; end

            onnx_p   = fullfile(bdir, onnx_rel);
            vnnlib_p = fullfile(bdir, vnnlib_rel);
            [~, onnx_tag, ~]   = fileparts(onnx_rel);
            [~, vnn_tag,  ~]   = fileparts(vnnlib_rel);
            stem = sprintf('%04d__%s__%s', k, onnx_tag, vnn_tag);
            res_file = fullfile(outdir, [stem '.result']);
            log_file = fullfile(outdir, [stem '.log']);
            json_file = fullfile(outdir, [stem '.json']);

            % resume: skip if .result exists with content
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
            try
                % prepare creates a .mat file (network + spec)
                prepare_instance(bench, onnx_p, vnnlib_p);
                % run reads the .mat and verifies
                [resStr, ~] = run_instance(bench, onnx_p, vnnlib_p, res_file, used_to, false);
                verdict = char(resStr);
            catch err
                % write error info to log
                lfid = fopen(log_file, 'a');
                fprintf(lfid, 'CORA error on %s: %s\n%s\n', stem, err.message, err.getReport);
                fclose(lfid);
                verdict = 'error';
            end
            wall = toc(t0);

            % normalise: CORA writes 'unsat'/'sat'/'unknown'/'timeout'/'verified'
            switch lower(verdict)
                case {'unsat','verified','holds','safe'},      cat_ver = 'unsat';   n_unsat = n_unsat+1;
                case {'sat','violated','falsified','unsafe'}, cat_ver = 'sat';     n_sat   = n_sat+1;
                case {'timeout','timed_out'},                  cat_ver = 'timeout'; n_to    = n_to+1;
                case 'unknown',                                cat_ver = 'unknown'; n_to    = n_to+1;
                otherwise,                                     cat_ver = verdict;   n_err   = n_err+1;
            end
            n_done = n_done + 1;

            % append summary
            sfid = fopen(sumcsv, 'a');
            fprintf(sfid, '%d,"%s","%s",%g,%g,%.2f,"%s","%s","%s"\n', ...
                k, onnx_rel, vnnlib_rel, csv_timeout, used_to, wall, cat_ver, res_file, log_file);
            fclose(sfid);

            % write per-instance JSON
            jfid = fopen(json_file, 'w');
            fprintf(jfid, '{"idx":%d,"benchmark":"%s","onnx":"%s","vnnlib":"%s","csv_timeout":%g,"used_timeout":%g,"wall_sec":%.2f,"verdict_raw":"%s","verdict":"%s","attack_path_exists":false,"tool":"CORA"}\n', ...
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

    L('=== sweep complete');
    fclose(log_fid);
end

function lprint(fid, s)
    ts = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    line = sprintf('[%s] %s\n', ts, s);
    fprintf('%s', line);
    fprintf(fid, '%s', line);
end
