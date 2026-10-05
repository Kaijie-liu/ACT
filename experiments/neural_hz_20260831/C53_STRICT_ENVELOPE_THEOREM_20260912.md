# Strict box normalization prevents a first signed-unit congruence

For each nonzero finite source summand a_i with integer parent power p_i,
frexp gives |a_i| < 2^e_i, where e_i includes p_i. This inequality is strict,
including when a_i is itself an exact power of two.

Set f=max(e_i)-26, T=sum 2^max(e_i-f,0), and
u=max(0,f+ceil(log2(T))). Then

sum |a_i|2^p_i < sum 2^e_i <= 2^f T <= 2^u.

For at most64M summands, T<2^52, so the existing int64 reduction is exact.
C53 recomputes this integer envelope from all original source rows and checks
u against the saved exponent for every MAIN coordinate. Source rows include
continuous, binary and constant summands; operator and sum rows use their
original parent exponents. This uses no rounded L1 sum.

Thus every normalized fresh factor has strict coefficient/constant L1<1.
In a homogeneous binary-free row, a signed-unit defining equality would need
continuous L1>=1. No first C52 merge exists. Initially all roots are distinct;
signed substitutions/coalescing would not increase L1 in any event. Therefore
there can be no later signed-chain or shared-Add merge under unchanged C52
normalization. This conclusion is about fresh MAIN definitions, not every
old predicate, nor a matrix transformed by another quotient algorithm.

The original physical network map can nevertheless have many single-parent
copies or power-of-two scalings. For y=a*2^p*x, a=s*2^k,
the incumbent fresh coordinate is z=s*2^(k+p-u)*x, with k+p-u<0.
Its box is redundant; source-scale-aware forwarding could retain the shared
root and carry the exponent in metadata before allocating z. Along a chain,
signs multiply and integer exponents add. This is an exact-real identity;
it is NOT permission to materialize unchecked floating products, delete UID
or inverse data, omit any consumer/owner check, or assume total storage/work
decreases. A new source-bound generator and full witness/resource proofs are
required. C53 only counts and independently checks this ordinary population.
