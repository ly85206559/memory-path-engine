# Production Incident Playbook (Noise-Realistic Draft)

## Detection
1 When p95 latency exceeds the alert threshold for five minutes, notify the incident commander and open an incident channel.
2 If the latest deployment is implicated, roll back the release and verify latency recovery.

## Recovery Branching
3 If rollback does not restore the API and slow queries continue, escalate to the database on-call engineer.
4 After escalation in clause 3, verify query plan regressions before restarting workers.
5 Unless a manual approval exception is granted, do not restart workers before query verification completes.

## Cross-Incident Notes
6 When repeated rollback failures occur across consecutive releases, apply the generalized rule: freeze further deploys until canary soak passes.
7 If payment dispute tickets and incident tickets both reference the same batch job, treat the shared failure mode as a cross-episode abstraction for postmortem clustering.
