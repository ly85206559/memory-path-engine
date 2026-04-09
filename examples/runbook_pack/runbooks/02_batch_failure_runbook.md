# Batch Failure Runbook

## Detection
1 When the nightly batch misses its completion SLA, notify the operations lead and inspect the scheduler logs.
2 If more than 20 percent of jobs are stuck in retry, pause downstream exports.

## Recovery
3 Verify whether the failure was caused by an upstream schema change or expired credentials.
4 If credentials expired, rotate the secret and rerun the failed batch segment.
5 If the schema changed, apply the compatible parser update before resuming exports.

## Escalation
6 Escalate to the data platform owner if rerun fails twice within the same maintenance window.
7 Except for approved maintenance windows, notify customer success before delaying external reports.
