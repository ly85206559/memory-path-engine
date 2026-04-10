# Dynamic Memory Priming Runbook

## Alpha Path
1 When alpha cache queue health degrades, verify alpha cache queue health.
2 If alpha cache queue health stays degraded, restart the alpha cache worker.
3 After the alpha cache worker restart, confirm alpha cache recovery.

## Beta Path
4 When beta cache queue health degrades, verify beta cache queue health.
5 If beta cache queue health stays degraded, restart the beta cache worker.
6 After the beta cache worker restart, collect beta cache diagnostics.
7 After beta cache diagnostics are collected, confirm beta cache recovery.
