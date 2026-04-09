# API Incident Runbook

## Detection
1 When p95 latency exceeds the alert threshold, notify the incident commander and open an incident channel.
2 If error rate exceeds 5 percent for 10 minutes, page the on-call backend engineer.

## Mitigation
3 Verify whether the latest deployment changed traffic patterns or dependency timeouts.
4 If the latest deployment is implicated, roll back the release and verify latency recovery.
5 If rollback does not recover service, restart the worker service and confirm queue drain behavior.

## Escalation
6 Escalate to the database owner if slow queries continue after rollback and worker restart.
7 Unless customer impact is contained, publish a status update within 15 minutes of escalation.
