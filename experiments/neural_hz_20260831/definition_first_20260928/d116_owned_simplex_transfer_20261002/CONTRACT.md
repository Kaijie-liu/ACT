# Owned phase joint budget transfer contract

This default-off mathematical component specializes the D115 two-anchor simplex factor to three actual observations when the old anchor belongs to the first observation's own ReLU. It appends six exact rational rows and no new coordinates. Original HZ factors, signed bits, EQ/LE, frame and decoder obligations remain. The question is whether these rows have a downstream effect after genuine neural source binding, not whether the already rejected D115 local point can be rescued.

## Definition and projection theorem

Retain the nonconvex HZ H(theta) and canonical same-theta observations v>=0, sum(v)<=B, t=alpha*v, u=beta*v, delta=alpha*beta, B>0. Original alpha/beta are authenticated views of distinct signed binary columns. Here v has length three and t1=v1 because alpha is the first output's own phase. The canonical joint amount s1 is consequently u1. Empty observations embed H; all added valid relations preserve its original integer projection and concrete decoder. Query relaxations are not substitutes for the binary concrete domain.

Assume the complete D035 box interfaces for each coordinate and the total, their real certified ranges, both single-phase simplex interfaces, and the own identity. Put A_i=t_i+u_i-v_i and C_i=t_i-u_i. The D115 joint factor projects exactly to the old system plus

```text
u1+A2 <= B*delta
u1+A3 <= B*delta
v1-u1+C2 <= B*(alpha-delta)
v1-u1+C3 <= B*(alpha-delta)
-A2-A3 <= B*(1-alpha-beta+delta)
-C2-C3 <= B*(beta-delta).
```

Proof: expand D115's four sums of positive parts. For coordinate one, A1=u1>=0, C1=v1-u1>=0, their negative parts vanish. Each remaining sum has only two positive parts and therefore four subset choices. The coordinate and total reference already implies the other choices, leaving exactly these six rows. This is exact projection, not a claim that each row is a facet or that every old auxiliary assignment is retained. With only two observations, the coordinate/total reference on this own-face is already exact.

For length three, local nonredundancy survives the own identity. At B=1, alpha=beta=1/2, delta=1/4, choose v=(1/5,1/4,3/10), t=(1/5,1/5,1/20), u=(3/20,1/5,1/20). Coordinate cell witnesses (11,10,01,00) are (6,2,0,0)/40, (7,1,1,1)/40, (1,1,1,9)/40. The total has witness (9,9,7,5)/40; both single-phase simplex hulls hold. The first new row fails by 1/20. This local point is not registered as a neural feasible point. For the fixed network below, additional source conditions rule it out; no local point is imposed on that network.

The small equivalent reference retains s2,s3, sets s1=u1, and enforces four nonnegative cell vectors and four total capacities. The six-row component and that reference must have equal query projections when appended to the same complete reference. This is known perspective/RLT projection, not a new convexification primitive. The project contribution under test is source-bound neural consumption and this explicit finite structural form.

## Physical rows and acceptance contract

On distinct physical v,t,u,alpha,beta,delta coordinates, six rows contain at most 37 coefficient nonzeros and add no continuous or binary columns. Actual signed-bit views and latent readouts change this count; the implementation reports actual row nnz and retained entries. Original rows and bounds remain an unchanged prefix. The existing source, old/new observations, overlap, normalization, inequalities-to-slacks, reconstruction, and all construction/terminal work remain payable. This is not a full-memory or speed claim.

The compiler requires the literal reference rows, source budget and own identity to be present, validates readout/frame/bit metadata, uses only exact rational coefficients capped at 512 bits, and checks a retained-entry cap before creating the six rows. It does not certify that arbitrary caller-provided observations really are original-model products. The registered neural control constructs and checks those premises explicitly; native ONNX binding remains unqualified. Disabled calls return the input without inspecting it. No solver or model is called by the compiler.

## Fixed complete mathematical neural block

Use the unchanged D115 source x,y in [-1,1] and original gates

```text
g1=x+y/4+1/5; q1=ReLU(g1)
g2=-x+y/5+3/10; q2=ReLU(g2)
g3=-y+x/10+1/4; q3=ReLU(g3)
h=q1-3*q2/5+2*q3/5+x/10-1/5; p=ReLU(h)
j=p+2*q1/5-4*q2/5+3*q3/5+y/10-1/4; w=ReLU(j).
```

Keep all five original signed bits and all original gate rows. alpha is q1's bit and beta is p's bit; do not replace them. Certified interval propagation gives each original preactivation range. The proven common budget is B=251/100 for q1+q2+q3. Both anchors retain conditional products of x,y,q1,q2,q3,p. q1's alpha-product is q1; p's beta-product is p. Lift the nonnegative/epigraph/secant rows of q1,q2,q3,p into both anchor cells, together with source bounds and the joint budget. Crucially retain q1=alpha*g1 and p=beta*h, including beta*x; the residual is not free noise. D035 interfaces share one delta and cover x,y,q1,q2,q3,p and the total q1+q2+q3. The known conditional bound alpha*(q1+q2)<=29*alpha/20 is included in every arm.

The final w gate receives the beta-conditional secant over j, using p=0 when beta=0 and p in [0,U_h] when beta=1. The complete y residual uses the same beta*y. This deliberately finite conditional source interface is not a full-prefix hull or full RLT hierarchy.

Compare A, this complete reference; C, A plus the six rows; and E, A plus the explicit small simplex extension. Predeclare both signs of the physical directions p, w, w-p, w-q1, w-q2, w-q3 and w-p+q1-q2. All 42 terminal diagnostic LPs are fixed before execution. Do not inspect LP marginals, choose cuts, change objectives or retry based on results. C/E equality and C no weaker than A are correctness checks, but positive A-C gain is NOT required for test success; a zero gain is an admissible negative research result. Binary columns remain tagged original in the mathematical carrier; terminal LP relaxes them only to measure bounds, not to change concrete semantics.

No production adapter, real trained model, GPU computation, full physical admission, shadow or benchmark replay is authorized by this component contract. No numeric result changes the formal 1870/2413 or independent 61/400. The Goal remains active in every outcome.
