BASE=/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/gpu_stream3_20260526T132419Z STREAM=gpu3
Tue May 26 01:24:19 PM UTC 2026
=== Tue May 26 01:24:19 PM UTC 2026 :: gpu3 :: vggnet16_2022 ===
  wall=240s ids=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17...
  -> rc=1
   counts: {'OK': 0, 'UNKNOWN_TIMEOUT': 18, 'UNKNOWN_RESOURCE_LIMIT': 0, 'ERROR': 0}
   verdicts: {}
=== Tue May 26 02:46:04 PM UTC 2026 :: gpu3 :: yolo_2023 ===
  wall=180s ids=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,2...
  -> rc=1
   counts: {'OK': 0, 'UNKNOWN_TIMEOUT': 0, 'UNKNOWN_RESOURCE_LIMIT': 0, 'ERROR': 72}
   verdicts: {'ERROR_NotImplementedError': 72}
=== Tue May 26 02:52:54 PM UTC 2026 :: gpu3 :: cifar100_2024 ===
  wall=120s ids=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,2...
  -> rc=1
   counts: {'OK': 121, 'UNKNOWN_TIMEOUT': 0, 'UNKNOWN_RESOURCE_LIMIT': 0, 'ERROR': 79}
   verdicts: {'UNKNOWN': 121, 'ERROR_OutOfMemoryError': 79}
=== Tue May 26 03:50:31 PM UTC 2026 :: gpu3 :: tinyimagenet_2024 ===
  wall=120s ids=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,2...
  -> rc=1
   counts: {'OK': 198, 'UNKNOWN_TIMEOUT': 0, 'UNKNOWN_RESOURCE_LIMIT': 0, 'ERROR': 2}
   verdicts: {'UNKNOWN': 197, 'FALSIFIED': 1, 'ERROR_OutOfMemoryError': 2}
Tue May 26 04:46:55 PM UTC 2026
=== gpu_stream3 done ===
