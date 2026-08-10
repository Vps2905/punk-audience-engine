# Security Policy

## Project status

The `audience-intelligence-agents` branch is the active preproduction integration
branch. The repository contains production-oriented controls, but live provider
delivery and downstream activation are not currently certified or authorized.

## Reporting a vulnerability

Do not disclose a suspected vulnerability in a public issue, discussion, pull
request, log, screenshot, or test artifact.

Use the repository's GitHub private vulnerability reporting or security advisory
channel when available. If private reporting is not enabled, contact the
repository owner privately through GitHub before sharing technical details.

Include only the minimum safe information needed to reproduce the issue:

- affected commit and component;
- expected and observed behavior;
- safe reproduction steps using synthetic/test data;
- security, privacy, tenant, or delivery impact;
- suggested mitigation, if known.

Never include real API keys, database URLs, provider records, MAIDs, device IDs,
email addresses, phone numbers, access tokens, private keys, or production data.
Rotate any credential that may have been exposed before sending the report.

## High-priority security boundaries

Reports are especially important when they involve:

- authentication or authorization bypass;
- cross-tenant data access;
- raw or hashed identifier disclosure;
- k-anonymity or differential-privacy failure;
- terminal-policy or manual-approval bypass;
- stale or uncertified data reaching activation;
- unsafe model/tool execution or capability escalation;
- secret leakage, dependency compromise, container escape, or IAM overreach;
- audit, lineage, retention, deletion, reconciliation, or evidence tampering.

## Responsible testing

Use local or explicitly authorized staging environments and synthetic data. Do
not probe third-party providers, Meta endpoints, cloud accounts, or databases
without written authorization. Do not perform denial-of-service, destructive,
or privacy-invasive testing.

Security fixes must preserve evidence and fail-closed behavior. A passing unit
test alone does not constitute penetration, privacy, or production certification.
