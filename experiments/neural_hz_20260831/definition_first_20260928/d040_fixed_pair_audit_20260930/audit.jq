# Only classify previously computed, authenticated certificate endpoints.
# No division, bound recomputation, candidate import, or phase search.
def require($condition; $message):
  if $condition then . else error($message) end;
def tuple_key:
  if . == null then "None" else "(" + (map(tostring)|join(", ")) + ")" end;
def pair_key: "(" + (map(tuple_key)|join(", ")) + ")";
def state($slot; $bound):
  if $slot == null then "padding"
  elif $bound[0][0] > 0 then "strict_active"
  elif $bound[1][0] < 0 then "strict_inactive"
  elif $bound[0][0] < 0 and $bound[1][0] > 0 then "crossing"
  else "zero_boundary" end;
def counts: group_by(.) | map({key:.[0], value:length}) | from_entries;

. as $data
| .result as $r
| require(($r.windows|length)==5 and ($r.source_forms|length)==1600
    and ($data.box|length)==3072 and $r.first_shape==[1,64,32,32]
    and $r.frame_identity==["5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16","modelInput"]
    and $data.source.model_sha256==$r.frame_identity[0]
    and $data.source.spec_sha256=="caca9ef2245019dd70883661bd5c102544f64152ddc0ea8f967eb8ed0882e996"
    and $data.packet.first_relu.output=="127"
    and ($data.packet.branches|length)==1
    and ($r.receivers|keys)==["0"]; "original frame/source/population mismatch")
| $data.packet.branches[0].conv as $conv
| require($conv.weight_shape==[64,64,3,3] and $conv.strides==[1,1]
    and $conv.pads==[1,1,1,1] and $conv.dilations==[1,1] and $conv.group==1
    and $conv.input_shape==[1,64,32,32] and $conv.output_shape==[1,64,32,32];
    "original convolution geometry mismatch")
| require(($r.receivers["0"]|length)==64
    and all(range(0;64); . as $c | $r.receivers["0"][$c].channel==$c
      and ($r.receivers["0"][$c].weights|length)==576); "receiver population mismatch")
| [[0,0],[0,31],[16,16],[31,0],[31,31]] as $positions
| [range(0;5) as $widx
   | $r.windows[$widx] as $w
   | require($w.branch==0 and $w.position==$positions[$widx]
       and ($w.pair_keys|length)==288 and ($w.pair_bounds|length)==288
       and ($w.source_slots|length)==576 and ($w.source_bounds|length)==576
       and ($w.original_phases|length)==576 and ($w.rows|length)==64;
       "window population mismatch")
   | require(all(range(0;64); . as $c
       | $w.rows[$c].channel==$c
       and $w.rows[$c].receiver_coefficients_ref==[0,$c]
       and $w.rows[$c].all_slot_and_pair_premises_ref==([0]+$w.position));
       "receiver reference mismatch")
   | range(0;288) as $p
   | (2*$p) as $i
   | $w.source_slots[$i:$i+2] as $slots
   | $w.source_bounds[$i:$i+2] as $bounds
   | $w.pair_bounds[$p] as $difference
   | require($w.pair_keys[$p]==$slots
       and $r.pair_bounds[($slots|pair_key)]==$difference
       and $difference[0][1]>0 and $difference[1][1]>0;
       "fixed pair key or bound mismatch")
   | require(all(range(0;2); . as $j
       | ($i+$j) as $s
       | (($s%9)/3|floor) as $ky | ($s%3) as $kx
       | ($w.position[0]+$ky-1) as $y | ($w.position[1]+$kx-1) as $x
       | (if $y>=0 and $y<32 and $x>=0 and $x<32
          then [($s/9|floor),$y,$x] else null end) as $expected
       | $slots[$j]==$expected
       and (if $slots[$j]==null then
          $w.original_phases[$s]==null and $bounds[$j]==[[0,1],[0,1]]
         else ($r.source_forms[($slots[$j]|tuple_key)]) as $source
          | $source!=null and $source.bounds==$bounds[$j]
          and $source.original_phase==(["127"]+$slots[$j])
          and $w.original_phases[$s]==$source.original_phase
          and $bounds[$j][0][1]>0 and $bounds[$j][1][1]>0 end));
       "canonical source, phase, or saved bound mismatch")
   | state($slots[0];$bounds[0]) as $anchor
   | state($slots[1];$bounds[1]) as $member
   | ($slots[0]!=null and $difference[0][0]>=0) as $certified
   | (if $slots[0]==null then "P"
      elif $certified|not then "N"
      elif $anchor=="strict_active" then "A"
      elif $anchor=="strict_inactive" then "I"
      elif $anchor=="zero_boundary" then "Z"
      elif $slots[1]==null or $bounds[1][1][0]<=0 then "T"
      else "C" end) as $code
   | {window:$widx,position:$w.position,pair:$p,source_slots:$slots,
      anchor_state:$anchor,member_state:$member,certified:$certified,code:$code}
  ] as $pairs
| require(($pairs|length)==1440;"incomplete fixed pair population")
| require(([$pairs[].source_slots[]|select(.!=null)]|length)==1600;
    "real source slot population mismatch")
| {schema:"d040_existing_certificate_audit_v1",
   archive_sha256:"fbab84537df071153a7d9362161b2c9aa85b80c8b4c605ff1e1149170e0965e0",
   population:{windows:5,source_forms:1600,source_canonical_slots:2880,
     real_source_slots:1600,padding_source_slots:1280,
     canonical_slot_consumer_positions:184320,
     receiver_rows:320,fixed_pairs:1440,pair_consumer_positions:92160},
   code_counts:([$pairs[].code]|counts),
   state_counts:([$pairs[]|.anchor_state+"/"+.member_state+"/"+(.certified|tostring)]|counts),
   by_window:[range(0;5) as $widx
      | [$pairs[]|select(.window==$widx)] as $records
      | {position:$positions[$widx],pairs:($records|length),
         code_counts:([$records[].code]|counts),
         fixed_pair_codes:([$records[].code]|join(""))}],
   certified_nonstable_anchors:[$pairs[]|select(.certified and
      (.anchor_state=="crossing" or .anchor_state=="zero_boundary"))],
   interpretation:"saved-certificate eligibility only; no new bound, row, solver or native phase-column qualification",
   formal_gain:0,source_census_qualified:false,candidate_qualified:false,gpu_qualified:false}
