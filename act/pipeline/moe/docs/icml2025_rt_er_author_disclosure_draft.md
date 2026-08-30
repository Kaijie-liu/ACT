# Draft author clarification: released RT-ER router training

**Status:** `USER_REPORTED_SENT_EXACT_DATE_NOT_RECORDED`

The user explicitly authorized sending on 2026-08-30 and later reported having
sent the message personally. The exact sent date has not yet been recorded, so
the reminder and public-issue windows are not scheduled. An earlier agent Gmail
search/send attempt failed with `ACCESS_TOKEN_SCOPE_INSUFFICIENT` / HTTP 403;
that failed attempt is a technical note, not Day 0. No agent retry is required.
Any reply belongs in the private, untracked survey directory defined by the
author-contact policy; the address itself is not stored in this public file.

## Subject

Clarification about router optimization in the released CIFAR-10 RT-ER code

## Message

Dear authors,

We are conducting an artifact-centered reproduction of the CIFAR-10 RT-ER
experiments from *Robust Mixture-of-Experts: A Dual Model Approach* using the
official repository at commit
`30ef94d77b5451595b82e739aa8938e1f4c4521f`.

In the released training path, we observe that the router output is converted
to an integer route before the selected expert output enters the training loss.
In a targeted execution audit, both router parameter tensors had no gradient,
neither tensor changed after optimization, and the optimizer created no router
state. The router tensors also have identical hashes in the frozen epoch-10 and
epoch-20 reproduction checkpoints.

Could you please clarify whether a fixed randomly initialized router was the
intended design for the reported CIFAR-10 RT-ER experiments? If a different
training script, estimator, or checkpoint was used to optimize the router, we
would be grateful for its exact provenance so that we can reproduce the
intended configuration accurately.

We are asking only to resolve the released artifact semantics and will record
any clarification factually in the reproduction report.

Best regards,

[Name / project team]

## Evidence boundary

- The draft describes the released CIFAR-10 training path, not every possible
  private or unreleased experiment.
- A static router is not characterized as intrinsically invalid.
- Until the authors respond, the repository classifies the missing
  differentiable estimator as underspecification and the observed released
  training behavior as an artifact-level fact.
- No non-response may be converted into refusal or evidence against the
  theorem; the frozen Day-0/Day-14/Day-21--30 policy applies from the user's
  actual sent date once recorded.
