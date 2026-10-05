# Read-only counts of authenticated archived bounds, never a model evaluator.
def insist($ok; $why): if $ok then . else error($why) end;
def choose3:
  if . < 3 then 0 else (. * (.-1) * (.-2) / 6) end;
def state:
  . as $bound
  | insist(type == "array" and length == 2; "bound shape")
  | insist(all(.[]; type == "array" and length == 2
                   and all(.[]; type == "number") and .[1] > 0); "rational shape")
  | .[0][0] as $lo | .[1][0] as $hi
  | insist(($lo <= 0 or $hi > 0) and ($hi >= 0 or $lo < 0); "bound signs")
  | if $lo > 0 then "strictly_active"
    elif $hi < 0 then "strictly_inactive"
    elif $lo < 0 and $hi > 0 then "strictly_crossing"
    else "zero_touch" end;
def unresolved: .state == "strictly_crossing" or .state == "zero_touch";
def model:
  . as $model
  | [.branches[] | select(.admitted)] as $branches
  | insist(.summary.expected_windows == .summary.completed_windows
      and (.windows|length) == .summary.expected_windows
      and .summary.expected_pairs == .summary.completed_pairs
      and .summary.expected_receiver_rows == .summary.completed_receiver_rows;
      "incomplete archived population")
  | [.windows[] | . as $window
      | [$branches[] | select(.index == $window.branch)] as $matches
      | insist(($matches|length) == 1; "window branch")
      | $matches[0] as $branch
      | insist(($branch.positions|any(. == $window.position)); "window position")
      | insist([.pairs[].channels] == $branch.channel_pairs; "original pairs")
      | [.pairs[]
          | {channel:.channels[0], state:(.relation.ordinary_h|state)},
            (if .channels[0] != .channels[1] then
               {channel:.channels[1], state:(.relation.ordinary_w|state)}
             else
               insist(.relation.same_consumer_certified == true
                 and (.relation.ordinary_h|state) == (.relation.ordinary_w|state);
                 "self pair") | empty end)]
        | sort_by(.channel) as $rows
      | insist([$rows[].channel] == [range(0;$branch.receiver_count)]; "all channels")
      | [$rows[]|select(unresolved)|.channel] as $open
      | {branch:$window.branch, position:$window.position,
         pair_count:($window.pairs|length), receiver_count:($rows|length),
         strictly_active:([$rows[]|select(.state=="strictly_active")]|length),
         strictly_inactive:([$rows[]|select(.state=="strictly_inactive")]|length),
         strictly_crossing:([$rows[]|select(.state=="strictly_crossing")]|length),
         zero_touch:([$rows[]|select(.state=="zero_touch")]|length),
         unresolved_channels:$open,
         arbitrary_unresolved_triples:($open|length|choose3),
         consecutive_unresolved_triples:
           ([range(0;(($rows|length)-2)) as $start
             | select(($open|index($start)) != null
                  and ($open|index($start+1)) != null
                  and ($open|index($start+2)) != null)]|length)}]
      as $windows
  | insist(([$windows[].receiver_count]|add)==$model.summary.expected_receiver_rows
      and ([$windows[].pair_count]|add)==$model.summary.expected_pairs
      and ([$windows[]|[.branch,.position]]|unique|length)==($windows|length);
      "full geometry population")
  | {model:$model.source.model_relative_path,
     model_sha256:$model.source.model_sha256,
     spec_sha256:$model.source.spec_sha256,
     actual_phase_column_binding_verified:$model.actual_phase_column_binding_verified,
     windows:$windows,
     totals:{receivers:([$windows[].receiver_count]|add),
       pairs:([$windows[].pair_count]|add),
       strictly_active:([$windows[].strictly_active]|add),
       strictly_inactive:([$windows[].strictly_inactive]|add),
       strictly_crossing:([$windows[].strictly_crossing]|add),
       zero_touch:([$windows[].zero_touch]|add),
       unresolved:([$windows[].unresolved_channels|length]|add),
       within_window_arbitrary_triples:([$windows[].arbitrary_unresolved_triples]|add),
       within_window_consecutive_triples:([$windows[].consecutive_unresolved_triples]|add),
       across_all_recorded_windows_triples:
         ([$windows[].unresolved_channels|length]|add|choose3)}};
insist(type == "array" and length == 2; "exactly the two saved complete archives")
| {schema:"d055_readonly_archive_diagnostic_v1", models:map(model),
   tiny_status:"missing_complete_archive_not_inferred_from_partial_logs",
   source_census_qualified:false, native_HZ_admitted:false,
   gpu_computation_completed:false, formal_gain:0}
