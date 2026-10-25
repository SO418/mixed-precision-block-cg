# GPU accuracy and cost

One row per RHS seed. Times are reported only if every repeat reaches the tolerance.

|matrix|seed|method|threshold|RHS exponent|success/runs|updates|setup+solve (s)|vs FP32|
|---|---:|---|---|---:|---|---:|---:|---:|
|Ani|301|call_scaled|—|0|1/1|140|1.7644|—|
|Ani|301|fp16|—|0|0/1|—|—|—|
|Ani|301|fp32|—|0|1/1|140|1.636|—|
|Ani|301|fp64|—|0|1/1|140|1.7832|—|
|Ani|301|local_scaled|—|0|1/1|140|1.7631|—|
|Ani|301|switch|0.0001|0|1/1|140|1.6659|—|
|Const|301|call_scaled|—|0|1/1|85|1.0731|—|
|Const|301|fp16|—|0|0/1|—|—|—|
|Const|301|fp32|—|0|1/1|80|0.93609|—|
|Const|301|fp64|—|0|1/1|80|1.0344|—|
|Const|301|local_scaled|—|0|1/1|90|1.1416|—|
|Const|301|switch|0.0001|0|1/1|90|1.2944|—|
|Dis|301|call_scaled|—|0|1/1|325|4.0906|—|
|Dis|301|fp16|—|0|0/1|—|—|—|
|Dis|301|fp32|—|0|1/1|120|1.4062|—|
|Dis|301|fp64|—|0|1/1|90|1.1449|—|
|Dis|301|local_scaled|—|0|1/1|330|4.1555|—|
|Dis|301|switch|0.0001|0|1/1|145|1.7179|—|
