# Sign-only audit of authenticated, previously computed rational intervals.
def require($ok; $message): if $ok then . else error($message) end;
def valid_number: type == "number" and isfinite;
def valid_bound:
  type == "array" and length == 2 and
  all(.[]; type == "array" and length == 2 and
      all(.[]; valid_number) and .[1] > 0);
def category:
  if .[0][0] > 0 then "strict_active"
  elif .[1][0] < 0 then "strict_inactive"
  elif .[0][0] < 0 and .[1][0] > 0 then "outer_crossing"
  else "zero_touch" end;
def coordinate_key: "(" + (map(tostring) | join(", ")) + ")";
def histogram: group_by(.) | map({key: .[0], value: length}) | from_entries;
def occurrences($label): map(select(. == $label)) | length;

map(
  . as $doc
  | require((.source_bounds|length) == 1600 and
            (.residual_groups|length) == 325 and (.windows|length) == 5;
            "unexpected source/group/window population")
  | require(all(.source_bounds|to_entries[];
                .key == (.value.coordinate|coordinate_key) and
                (.value.bounds|valid_bound)); "invalid source bound/reference")
  | [.residual_groups|to_entries[]|
      .value as $g
      | ($g.source_coordinates|length) as $n
      | require(($n == 4 or $n == 5) and
                ($g.knots|length) == $n and
                ($g.source_bounds|length) == $n and
                ($g.residual_bounds|length) == $n - 2;
                "unexpected group shape")
      | require(all($g.source_bounds[], $g.residual_bounds[]; valid_bound);
                "invalid group bound")
      | require(all(range(0; $n);
          . as $i | ($g.source_coordinates[$i]|coordinate_key) as $k |
          $doc.source_bounds[$k] != null and
          $doc.source_bounds[$k].coordinate == $g.source_coordinates[$i] and
          ($doc.source_bounds[$k].bounds|category) ==
          ($g.source_bounds[$i]|category)); "missing/mismatched group source")
      | ($g.source_bounds|map(category)) as $classes
      | {size:$n, crossing:($classes|occurrences("outer_crossing")),
         zero_touch:($classes|occurrences("zero_touch")),
         anchor_pair:([$classes[0],$classes[-1]]|join("/")),
         coordinates:$g.source_coordinates,
         residual_classes:($g.residual_bounds|map(category)),
         exact_zero_residuals:([$g.residual_bounds[]|
                              select(.[0][0] == 0 and .[1][0] == 0)]|length)}
    ] as $groups
  | require(([$groups[].coordinates[]]|length) == 1600 and
            ([$groups[].coordinates[]]|unique|length) == 1600;
            "group population does not cover each source once")
  | {model:$doc.source.model_relative_path,
     source_forms:($doc.source_bounds|length), groups:($groups|length),
     windows:($doc.windows|length), downstream_rows:([$doc.windows[].rows[]]|length),
     source_classes:([$doc.source_bounds[].bounds|category]|histogram),
     group_sizes:([$groups[].size|tostring]|histogram),
     group_crossing_histogram:([$groups[].crossing|tostring]|histogram),
     group_zero_touch_histogram:([$groups[].zero_touch|tostring]|histogram),
     anchor_pairs:([$groups[].anchor_pair]|histogram),
     strict_lemma_eligible_groups:([$groups[]|
                                  select(.zero_touch == 0 and .crossing <= 1)]|length),
     multi_cross_groups:([$groups[]|select(.crossing >= 2)]|length),
     multi_cross_without_zero_touch:([$groups[]|
                                    select(.crossing >= 2 and .zero_touch == 0)]|length),
     residual_classes:([$groups[].residual_classes[]]|histogram),
     exact_zero_residuals:([$groups[].exact_zero_residuals]|add)}
)
| require(length == 3 and (map(.downstream_rows) == [320,640,640]);
          "unexpected file/row population")
